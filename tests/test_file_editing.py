from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from agent import Agent
from agent.llm import LLMClient
from agent.storage import ConversationStore
from config import ConfigManager
from tests.fakes import chat_response, function_call
from tools import ToolContext, registry
from tools.file_io import FileConflictError


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileEditingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "sample.txt"
        self.changes = []
        self.context = ToolContext(
            self.root, frozenset({"file"}), approval_mode="auto", record_file_change=self.changes.append,
        )

    def call(self, name, **arguments):
        return json.loads(registry.dispatch(name, arguments, context=self.context))

    def test_paginated_read_hashes_whole_file_and_preserves_spaces(self):
        data = b"first\r\nsecond  \r\nthird\r\n"
        self.path.write_bytes(data)
        result = self.call("read_file", path="sample.txt", offset=2, limit=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "2|second  ")
        self.assertEqual(result["sha256"], digest(data))
        self.assertEqual(result["total_lines"], 3)

    def test_read_rejects_invalid_ranges_and_directories(self):
        self.path.write_bytes(b"text")
        for arguments in ({"offset": 0}, {"limit": -1}, {"path": "."}):
            with self.subTest(arguments=arguments):
                self.assertFalse(self.call("read_file", **{"path": "sample.txt", **arguments})["ok"])

    def test_all_edit_tools_accept_guard_and_return_new_hash(self):
        for name in ("write_file", "patch", "apply_patch"):
            with self.subTest(tool=name):
                self.path.write_bytes(b"old\n")
                arguments = {"content": "new\n"} if name == "write_file" else {
                    "old_string": "old", "new_string": "new",
                }
                result = self.call(name, path="sample.txt", expected_sha256=digest(b"old\n").upper(), **arguments)
                self.assertTrue(result["ok"])
                self.assertTrue(result["checkpoint_created"])
                self.assertEqual(self.path.read_bytes(), b"new\n")
                self.assertEqual(result["sha256"], digest(b"new\n"))
                self.assertEqual(result["before_sha256"], digest(b"old\n"))

    def test_stale_guard_blocks_all_edits_without_checkpoint(self):
        self.path.write_bytes(b"user change")
        for name in ("write_file", "patch", "apply_patch"):
            with self.subTest(tool=name):
                arguments = {"content": "new"} if name == "write_file" else {
                    "old_string": "user", "new_string": "agent",
                }
                result = self.call(name, path="sample.txt", expected_sha256=digest(b"old"), **arguments)
                self.assertFalse(result["ok"])
                self.assertTrue(result["metadata"]["conflict"])
                self.assertEqual(result["metadata"]["actual_sha256"], digest(b"user change"))
                self.assertEqual(self.path.read_bytes(), b"user change")
        self.assertEqual(self.changes, [])

    def test_deleted_file_is_a_guard_conflict(self):
        result = self.call("patch", path="sample.txt", old_string="old", new_string="new", expected_sha256=digest(b"old"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["metadata"]["conflict"])
        self.assertIsNone(result["metadata"]["actual_sha256"])

    def test_missing_guard_distinguishes_empty_file_from_absent_file(self):
        result = self.call("write_file", path="sample.txt", content="", expected_sha256="missing")
        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertIsNone(self.changes[0]["before_content"])
        result = self.call("write_file", path="sample.txt", content="new", expected_sha256="missing")
        self.assertFalse(result["ok"])
        self.assertEqual(self.path.read_bytes(), b"")
        self.assertEqual(len(self.changes), 1)

    def test_invalid_hashes_and_host_argument_injection_are_rejected(self):
        self.path.write_bytes(b"old")
        for guard in ("", "not-a-hash", "z" * 64, True, {"workspace": "elsewhere"}):
            with self.subTest(guard=guard):
                self.assertFalse(self.call("write_file", path="sample.txt", content="new", expected_sha256=guard)["ok"])
        result = self.call("write_file", path="sample.txt", content="new", expected_sha256=digest(b"old"), context={"approval_mode": "auto"})
        self.assertTrue(result["blocked"])
        self.assertEqual(self.path.read_bytes(), b"old")

    def test_hash_does_not_replace_host_approval(self):
        self.path.write_bytes(b"old")
        context = ToolContext(self.root, frozenset({"file"}), approval_mode="on_request")
        arguments = {"path": "sample.txt", "content": "new", "expected_sha256": digest(b"old")}
        denied = json.loads(registry.dispatch("write_file", arguments, context=context))
        self.assertTrue(denied["blocked"])
        self.assertEqual(self.path.read_bytes(), b"old")
        approved = json.loads(registry.dispatch("write_file", arguments, context=context, approved=True))
        self.assertTrue(approved["ok"])

    def test_no_op_does_not_write_or_create_checkpoint(self):
        self.path.write_bytes(b"unchanged")
        with patch("tools.file_io.os.replace") as replace:
            for name, arguments in (
                ("write_file", {"content": "unchanged"}),
                ("patch", {"old_string": "unchanged", "new_string": "unchanged"}),
            ):
                result = self.call(name, path="sample.txt", **arguments)
                self.assertFalse(result["changed"])
                self.assertFalse(result["checkpoint_created"])
            replace.assert_not_called()
        self.assertEqual(self.changes, [])

    def test_patch_preserves_bom_line_endings_and_missing_final_newline(self):
        for data, old, new, expected in (
            (b"\xef\xbb\xbfalpha\r\nbeta\r\n", "alpha\nbeta", "one\ntwo", b"\xef\xbb\xbfone\r\ntwo\r\n"),
            (b"alpha\nbeta", "alpha", "one", b"one\nbeta"),
            (b"alpha\r\nbeta\n", "beta", "two", b"alpha\r\ntwo\n"),
        ):
            with self.subTest(data=data):
                self.path.write_bytes(data)
                result = self.call("apply_patch", path="sample.txt", old_string=old, new_string=new)
                self.assertTrue(result["ok"])
                self.assertEqual(self.path.read_bytes(), expected)
                self.assertEqual(self.changes[-1]["before_content"].encode("utf-8"), data)

    def test_empty_patch_match_is_rejected_even_with_replace_all(self):
        self.path.write_bytes(b"old")
        result = self.call("patch", path="sample.txt", old_string="", new_string="new", replace_all=True)
        self.assertFalse(result["ok"])
        self.assertEqual(self.path.read_bytes(), b"old")

    def test_replace_failure_preserves_original_and_cleans_temporary(self):
        self.path.write_bytes(b"original")
        with patch("tools.file_io.os.replace", side_effect=OSError("simulated replace failure")):
            result = self.call("write_file", path="sample.txt", content="new")
        self.assertFalse(result["ok"])
        self.assertEqual(self.path.read_bytes(), b"original")
        self.assertEqual(list(self.root.glob(".drudge-edit-*")), [])
        self.assertEqual(self.changes, [])

    def test_write_rechecks_snapshot_after_staging(self):
        self.path.write_bytes(b"original")
        with patch("tools.file_io.os.fsync", side_effect=lambda _: self.path.write_bytes(b"editor change")):
            result = self.call("write_file", path="sample.txt", content="agent change")
        self.assertFalse(result["ok"])
        self.assertTrue(result["metadata"]["conflict"])
        self.assertEqual(self.path.read_bytes(), b"editor change")
        self.assertEqual(list(self.root.glob(".drudge-edit-*")), [])
        self.assertEqual(self.changes, [])

    def test_unreadable_existing_file_is_not_treated_as_new(self):
        self.path.write_bytes(b"original")
        with patch("tools.file_ops.read_bytes", side_effect=PermissionError("simulated read denial")):
            result = self.call("write_file", path="sample.txt", content="new")
        self.assertFalse(result["ok"])
        self.assertEqual(self.path.read_bytes(), b"original")
        self.assertEqual(self.changes, [])

    def test_atomic_write_preserves_permission_bits(self):
        self.path.write_bytes(b"original")
        self.path.chmod(0o744)
        mode = self.path.stat().st_mode
        self.assertTrue(self.call("write_file", path="sample.txt", content="new")["ok"])
        self.assertEqual(self.path.stat().st_mode, mode)

    def test_checkpoint_failure_reports_saved_file_with_warning(self):
        self.context = ToolContext(self.root, frozenset({"file"}), approval_mode="auto", record_file_change=Mock(side_effect=OSError("database full")))
        result = self.call("write_file", path="sample.txt", content="saved")
        self.assertTrue(result["ok"])
        self.assertFalse(result["checkpoint_created"])
        self.assertIn("checkpoint recording failed", result["warnings"][0])
        self.assertEqual(self.path.read_bytes(), b"saved")


class FileUndoTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "sample.txt"
        self.config = ConfigManager()
        self.config.override("storage", "enabled", value=True)
        self.config.override("storage", "path", value=str(self.root / "session.db"))
        self.config.override("security", "workspace_root", value=str(self.root))
        self.config.override("security", "approval_mode", value="auto")
        self.config.override("toolsets", value=["file"])
        self.agent = Agent(self.config)
        self.agent.session_id = self.agent.store.create_session("edits", "offline", cwd=str(self.root))
        self.agent.store.append_message(self.agent.session_id, "user", "edit a file")
        self.agent._refresh_tool_context()

    def edit(self, content):
        result = json.loads(registry.dispatch("write_file", {"path": "sample.txt", "content": content}, context=self.agent.tool_context))
        self.assertTrue(result["ok"], result)
        return result

    def test_byte_exact_checkpoint_survives_resume_and_undo(self):
        original = b"\xef\xbb\xbfbefore\r\nunchanged\nlast"
        self.path.write_bytes(original)
        self.edit("after\r\n")
        revision = self.agent.list_file_revisions()[0]
        self.assertEqual(revision["snapshot_version"], 1)
        self.assertEqual(revision["before_sha256"], digest(original))
        self.assertEqual(revision["after_sha256"], digest(b"after\r\n"))
        resumed = Agent(self.config)
        resumed.resume_session(self.agent.session_id)
        self.assertTrue(resumed.undo_last_file_change()["undone"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_preview_does_not_mutate_file_or_revision(self):
        self.path.write_bytes(b"before")
        self.edit("after")
        preview = self.agent.undo_last_file_change(dry_run=True)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["undo_action"], "restore")
        self.assertIn("-after", preview["undo_diff_summary"])
        self.assertIn("+before", preview["undo_diff_summary"])
        self.assertFalse(preview["undone"])
        self.assertEqual(self.path.read_bytes(), b"after")
        self.assertEqual(len(self.agent.list_file_revisions()), 1)

    def test_undo_preserves_user_edits_and_deletions(self):
        self.path.write_bytes(b"before")
        self.edit("after")
        for external in (b"user edit", None):
            with self.subTest(external=external):
                if external is None:
                    self.path.unlink()
                else:
                    self.path.write_bytes(external)
                for dry_run in (True, False):
                    with self.assertRaises(FileConflictError):
                        self.agent.undo_last_file_change(dry_run=dry_run)
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, external)
                self.assertEqual(len(self.agent.list_file_revisions()), 1)

    def test_new_and_empty_files_undo_differently(self):
        self.edit("created")
        self.assertEqual(self.agent.undo_last_file_change(dry_run=True)["undo_action"], "delete")
        self.agent.undo_last_file_change()
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b"")
        self.edit("changed")
        self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"")

    def test_sequential_undo_reaches_original_bytes(self):
        self.path.write_bytes(b"first\r\n")
        self.edit("second\r\n")
        self.edit("third\n")
        self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"second\r\n")
        self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"first\r\n")
        self.assertEqual(self.agent.list_file_revisions(), [])

    def test_undo_obeys_current_host_policy_but_preview_remains_read_only(self):
        self.path.write_bytes(b"before")
        self.edit("after")
        self.config.override("security", "approval_mode", value="never")
        self.assertTrue(self.agent.undo_last_file_change(dry_run=True)["dry_run"])
        with self.assertRaises(PermissionError):
            self.agent.undo_last_file_change()
        self.config.override("security", "approval_mode", value="auto")
        self.config.override("toolsets", value=[])
        with self.assertRaises(PermissionError):
            self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"after")
        self.assertEqual(len(self.agent.list_file_revisions()), 1)

    def test_undo_blocked_for_credential_and_outside_paths(self):
        for path in (self.root / ".drudge" / "auth.json", self.root.parent / "outside.txt"):
            with self.subTest(path=path):
                self.agent.store.record_file_revision(
                    session_id=self.agent.session_id, run_id=None, path=str(path),
                    operation="write_file", before_content=None, after_content="fictional",
                )
                with patch("tools.file_io.read_bytes") as read:
                    with self.assertRaises(PermissionError):
                        self.agent.undo_last_file_change()
                    read.assert_not_called()

    def test_failed_restore_leaves_checkpoint_reversible(self):
        self.path.write_bytes(b"before")
        self.edit("after")
        with patch("tools.file_io.os.replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"after")
        self.assertEqual(len(self.agent.list_file_revisions()), 1)
        self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"before")

    def test_corrupt_checkpoint_is_not_applied(self):
        self.path.write_bytes(b"before")
        self.edit("after")
        with closing(sqlite3.connect(self.agent.store.path)) as connection, connection:
            connection.execute("UPDATE file_revisions SET before_content = 'corrupt'")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"after")

    def test_legacy_checkpoint_uses_exact_comparison_not_newline_guessing(self):
        self.path.write_bytes(b"before\n")
        self.edit("after\n")
        with closing(sqlite3.connect(self.agent.store.path)) as connection, connection:
            connection.execute("UPDATE file_revisions SET snapshot_version = 0, before_sha256 = NULL, after_sha256 = NULL")
        self.path.write_bytes(b"after\r\n")
        with self.assertRaises(FileConflictError):
            self.agent.undo_last_file_change()
        self.path.write_bytes(b"after\n")
        self.agent.undo_last_file_change()
        self.assertEqual(self.path.read_bytes(), b"before\n")

    def test_already_undone_revision_never_calls_restore_again(self):
        self.edit("created")
        revision = self.agent.undo_last_file_change()
        callback = Mock()
        with self.assertRaises(RuntimeError):
            self.agent.store.apply_file_revision_undo(revision["id"], callback)
        callback.assert_not_called()

    def test_concurrent_undo_consumers_apply_a_revision_once(self):
        self.edit("created")
        revision = self.agent.list_file_revisions()[0]
        second_store = ConversationStore(str(self.agent.store.path))
        calls = []

        def undo(store):
            try:
                store.apply_file_revision_undo(revision["id"], lambda item: calls.append(item["id"]))
                return True
            except RuntimeError:
                return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(undo, (self.agent.store, second_store)))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(calls, [revision["id"]])

    def test_storage_disabled_does_not_advertise_a_checkpoint(self):
        self.config.override("storage", "enabled", value=False)
        agent = Agent(self.config)
        agent._refresh_tool_context()
        self.assertIsNone(agent.tool_context.record_file_change)
        result = json.loads(registry.dispatch("write_file", {"path": "sample.txt", "content": "new"}, context=agent.tool_context))
        self.assertFalse(result["checkpoint_created"])


class FileRevisionMigrationTests(unittest.TestCase):
    def test_legacy_revision_migration_is_non_destructive_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.executescript("""
                    CREATE TABLE file_revisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, run_id TEXT,
                        path TEXT NOT NULL, operation TEXT NOT NULL,
                        before_content TEXT, after_content TEXT, diff_summary TEXT NOT NULL DEFAULT '',
                        undone INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, undone_at TEXT
                    );
                    INSERT INTO file_revisions (id, path, operation, before_content, after_content, undone, undone_at)
                    VALUES (42, 'sample.txt', 'patch', '', 'after', 1, '2026-01-01');
                """)
            for _ in range(2):
                store = ConversationStore(str(path))
                revision = store.get_file_revision(42)
                self.assertEqual(revision["snapshot_version"], 0)
                self.assertIsNone(revision["after_sha256"])
                self.assertEqual(revision["before_content"], "")
                self.assertTrue(revision["undone"])
                self.assertEqual(revision["undone_at"], "2026-01-01")
            created = store.record_file_revision(
                session_id=None, run_id=None, path="new.txt", operation="write_file",
                before_content=None, after_content="line\r\n",
            )
            self.assertEqual(created["snapshot_version"], 1)
            self.assertIsNone(created["before_sha256"])
            self.assertEqual(created["after_sha256"], digest(b"line\r\n"))


class GuardedEditProtocolTests(unittest.TestCase):
    def test_guarded_edits_remain_valid_in_both_api_tool_loops(self):
        class Client(LLMClient):
            def __init__(self, api, stale):
                super().__init__(base_url="https://example.invalid/v1", api_key="offline", model="offline", api_type=api)
                self.bodies = []
                self.api = api
                self.expected = digest(b"stale" if stale else b"original")

            async def _post_json(self, url, body):
                self.bodies.append(body)
                if len(self.bodies) == 1:
                    arguments = json.dumps({"path": "sample.txt", "old_string": "original", "new_string": "agent", "expected_sha256": self.expected})
                    if self.api == "chat":
                        return chat_response(finish_reason="tool_calls", tool_calls=[function_call("guard-1", "apply_patch", arguments)])
                    return {"status": "completed", "output": [{"type": "function_call", "call_id": "guard-1", "name": "apply_patch", "arguments": arguments}]}
                if self.api == "chat":
                    return chat_response("finished")
                return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "finished"}]}]}

        for api, stale in (("chat", False), ("chat", True), ("responses", False), ("responses", True)):
            with self.subTest(api=api, stale=stale), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "sample.txt"
                target.write_bytes(b"original")
                config = ConfigManager()
                config.override("security", "approval_mode", value="auto")
                config.override("storage", "enabled", value=False)
                config.override("security", "workspace_root", value=directory)
                config.override("toolsets", value=["file"])
                config.override("agent", "refusal_review_enabled", value=False)
                config.override("display", "show_tool_calls", value=False)
                agent = Agent(config)
                client = Client(api, stale)
                agent.llm = client
                result = asyncio.run(agent.run("edit sample.txt"))
                self.assertEqual(result, "finished")
                self.assertEqual(target.read_bytes(), b"original" if stale else b"agent")
                if api == "chat":
                    output = next(item for item in client.bodies[1]["messages"] if item["role"] == "tool")
                    payload = json.loads(output["content"])
                    self.assertEqual(output["tool_call_id"], "guard-1")
                    schema = next(tool["function"] for tool in client.bodies[0]["tools"] if tool["function"]["name"] == "apply_patch")
                else:
                    output = next(item for item in client.bodies[1]["input"] if item.get("type") == "function_call_output")
                    payload = json.loads(output["output"])
                    self.assertEqual(output["call_id"], "guard-1")
                    schema = next(tool for tool in client.bodies[0]["tools"] if tool["name"] == "apply_patch")
                self.assertEqual(payload["ok"], not stale)
                if stale:
                    self.assertTrue(payload["metadata"]["conflict"])
                else:
                    self.assertEqual(payload["metadata"]["sha256"], digest(b"agent"))
                self.assertEqual(schema["parameters"]["properties"]["expected_sha256"]["type"], "string")
                self.assertNotIn("expected_sha256", schema["parameters"]["required"])
