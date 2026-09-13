from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from agent import Agent
from agent.llm import LLMClient
from agent.storage import ConversationStore
from config import ConfigManager
from tests.fakes import chat_response, function_call
from tools import OutputToolProvider, ToolContext, ToolResult, registry
from tools.output_capture import MAX_OUTPUT_PAGE_CHARS, OUTPUT_CHUNK_CHARS, OutputCapture, utf8_prefix
from tools.result import limit_tool_result, normalize_tool_payload


class ToolResultBudgetTests(unittest.TestCase):
    def test_small_results_keep_legacy_fields(self):
        result = json.loads(limit_tool_result({"output": "hello", "exit_code": 0}))
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "hello")
        self.assertEqual(result["exit_code"], 0)

    def test_explicit_failure_and_contradictory_success_remain_failures(self):
        for raw in (
            {"ok": False, "content": "failed without an error field"},
            {"ok": True, "content": "", "error": "failed", "metadata": {}},
            {"ok": True, "content": "", "error": None, "metadata": {}, "blocked": True},
        ):
            with self.subTest(raw=raw):
                self.assertFalse(normalize_tool_payload(raw)["ok"])

    def test_unicode_and_control_characters_stay_within_serialized_budget(self):
        for content in ("hello" * 10000, "中文🙂" * 10000, "\x00\x1b\r\n\t\\\"" * 10000):
            for budget in (1024, 2048, 10000):
                with self.subTest(budget=budget, first=repr(content[:5])):
                    encoded = limit_tool_result(ToolResult.failure("error: " + content, content=content, exit_code=7), max_chars=budget)
                    self.assertLessEqual(len(encoded), budget)
                    payload = json.loads(encoded)
                    self.assertFalse(payload["ok"])
                    self.assertIsNotNone(payload["error"])
                    self.assertEqual(payload["metadata"]["exit_code"], 7)
                    self.assertTrue(payload["metadata"]["truncated"])

    def test_persisted_original_is_not_the_preview(self):
        saved = []
        original = {"content": "HEADER" + "x" * 15000 + "FOOTER", "arbitrary": {"nested": [1, 2]}}

        def save(content):
            saved.append(content)
            return {"id": "a" * 32, "complete": True}

        encoded = limit_tool_result(original, save_output=save)
        payload = json.loads(encoded)
        self.assertIn("HEADER", payload["content"])
        self.assertIn("FOOTER", payload["content"])
        self.assertEqual(json.loads(saved[0])["content"], original["content"])
        self.assertEqual(json.loads(saved[0])["arbitrary"], original["arbitrary"])
        self.assertEqual(payload["metadata"]["output_ref"]["id"], "a" * 32)

    def test_huge_metadata_and_receipt_do_not_defeat_budget_or_change_status(self):
        original = ToolResult.failure(
            "\x00" * 10000, blocked=True, conflict=True, checkpoint_created=False,
            content="", enormous={str(index): "large" * 1000 for index in range(50)},
        )
        before = copy.deepcopy(original.to_dict())
        payload = limit_tool_result(original, max_chars=1024, save_output=lambda _: {
            "id": "a" * 32, "kind": "\x00" * 10000, "sha256": "\x00" * 10000, "complete": True,
            "extra": "x" * 100000,
        })
        self.assertLessEqual(len(payload), 1024)
        parsed = json.loads(payload)
        self.assertFalse(parsed["ok"])
        self.assertTrue(parsed["blocked"])
        self.assertTrue(parsed["metadata"]["conflict"])
        self.assertFalse(parsed["metadata"]["checkpoint_created"])
        self.assertEqual(original.to_dict(), before)

    def test_output_save_failure_does_not_invite_repeating_a_successful_mutation(self):
        encoded = limit_tool_result(
            ToolResult.success("saved" * 10000, changed=True, checkpoint_created=True),
            save_output=Mock(side_effect=sqlite3.OperationalError("disk full")),
        )
        payload = json.loads(encoded)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["metadata"]["checkpoint_created"])
        self.assertIn("persistence failed", payload["metadata"]["output_warning"])

    def test_missing_storage_and_invalid_receipts_are_explicit(self):
        for save in (None, lambda _: None):
            result = json.loads(limit_tool_result("x" * 20000, save_output=save))
            self.assertTrue(result["ok"])
            self.assertIsNone(result["metadata"]["output_ref"])
            self.assertIn("output_warning", result["metadata"])

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            limit_tool_result("text", max_chars=100)


class OutputCaptureTests(unittest.TestCase):
    def test_capture_is_bounded_and_keeps_beginning_and_tail(self):
        capture = OutputCapture(persist=True, max_bytes=32, preview_chars=20)
        self.addCleanup(capture.close)
        capture.write("HEAD:")
        for _ in range(1000):
            capture.write("x" * 1000)
        capture.write(":TAIL")
        self.assertLessEqual(len(capture.prefix), 20)
        self.assertLessEqual(len(capture.tail), 10)
        self.assertLessEqual(capture.stored_bytes, 32)
        self.assertIn("HEAD:", capture.preview())
        self.assertIn(":TAIL", capture.preview())
        callback = Mock(return_value={"id": "test"})
        capture.finish(callback, tool_name="terminal.stdout", status="completed")
        self.assertEqual(callback.call_args.args[0], "HEAD:" + "x" * 27)
        self.assertFalse(callback.call_args.kwargs["complete"])
        self.assertEqual(callback.call_args.kwargs["source_chars"], 1000010)

    def test_utf8_cap_is_a_true_prefix_without_skipped_characters(self):
        self.assertEqual(utf8_prefix("🙂abc", 5), "🙂a")
        capture = OutputCapture(persist=True, max_bytes=6, preview_chars=2)
        self.addCleanup(capture.close)
        capture.write("🙂a")
        capture.write("🙂")
        capture.write("b")
        callback = Mock(return_value={"id": "test"})
        capture.finish(callback, tool_name="terminal.stdout", status="completed")
        self.assertEqual(callback.call_args.args[0], "🙂a")
        self.assertTrue(capture.storage_truncated)

    def test_unpersisted_output_retains_bounded_preview_with_warning(self):
        capture = OutputCapture(persist=False, preview_chars=8)
        capture.write("abcdefghijklmnop")
        self.assertIsNone(capture.finish(None, tool_name="terminal.stdout", status="completed"))
        self.assertIn("not persisted", capture.warning)
        self.assertIsNone(capture.spool)

    def test_spool_failure_continues_tracking_and_retains_failure_tail(self):
        capture = OutputCapture(persist=True, preview_chars=8)
        self.addCleanup(capture.close)
        with patch.object(capture.spool, "write", side_effect=OSError("disk full")):
            capture.write("first chunk")
        capture.write("last error")
        self.assertIn("disk full", capture.warning)
        self.assertTrue(capture.storage_truncated)
        self.assertIn("rror", capture.preview())
        self.assertEqual(capture.source_chars, 21)


class OutputFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = ConfigManager()
        self.config.override("storage", "enabled", value=True)
        self.config.override("storage", "path", value=str(self.root / "session.db"))
        self.config.override("security", "workspace_root", value=str(self.root))
        self.config.override("security", "approval_mode", value="auto")
        self.config.override("toolsets", value=["file", "terminal"])
        self.config.override("display", "show_tool_calls", value=False)
        self.config.override("agent", "refusal_review_enabled", value=False)
        self.config.override("agent", "repo_map_enabled", value=False)
        self.config.override("tool_selection", "enabled", value=False)
        self.agent = Agent(self.config)
        self.store = self.agent.store
        self.session_id = self.store.create_session("outputs", "offline", cwd=str(self.root))
        self.store.append_message(self.session_id, "user", "initial request")
        self.agent.resume_session(self.session_id)

    def save(self, content, **kwargs):
        return self.store.save_tool_output(content, session_id=self.session_id, workspace=str(self.root), **kwargs)


class OutputStorageTests(OutputFixture):
    def test_unicode_nul_and_chunk_boundaries_round_trip(self):
        content = ("abcd中文🙂\x00\n" * 1100) + "last line"
        receipt = self.save(content, kind="text")
        for offset in (0, 7, OUTPUT_CHUNK_CHARS - 3, OUTPUT_CHUNK_CHARS, len(content) - 2, len(content), len(content) + 12):
            with self.subTest(offset=offset):
                page = self.agent.read_tool_output(receipt["id"], offset, 41)
                self.assertEqual(page["content"], content[offset:offset + 41])
                self.assertEqual(page["eof"], offset + 41 >= len(content))
        chunks = []
        offset = 0
        while True:
            page = self.agent.read_tool_output(receipt["id"], offset)
            chunks.append(page["content"])
            if page["eof"]:
                break
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
        self.assertEqual("".join(chunks), content)
        self.assertEqual(receipt["sha256"], hashlib.sha256(content.encode("utf-8")).hexdigest())

    def test_capped_json_is_explicitly_partial_text(self):
        receipt = self.save('{"content":"🙂abcdef"}', max_bytes=15)
        self.assertFalse(receipt["complete"])
        self.assertEqual(receipt["kind"], "text")
        self.assertLessEqual(receipt["size_bytes"], 15)
        self.assertEqual(self.agent.read_tool_output(receipt["id"])["content"], utf8_prefix('{"content":"🙂abcdef"}', 15))

    def test_empty_output_and_invalid_pagination(self):
        receipt = self.save("")
        page = self.agent.read_tool_output(receipt["id"])
        self.assertEqual(page["content"], "")
        self.assertTrue(page["eof"])
        for offset, limit in ((-1, 10), (True, 10), (0, 0), (0, 1001), (0, True), ("0", 10)):
            with self.subTest(offset=offset, limit=limit), self.assertRaises(ValueError):
                self.agent.read_tool_output(receipt["id"], offset, limit)
        with self.assertRaises(ValueError):
            self.agent.read_tool_output("../session.db")

    def test_output_scope_does_not_cross_sessions_or_workspaces(self):
        receipt = self.save("session private output")
        other = self.store.create_session("other", "offline")
        with self.assertRaises(KeyError):
            self.store.read_tool_output(receipt["id"], session_id=other, workspace=str(self.root))
        with self.assertRaises(KeyError):
            self.store.read_tool_output(receipt["id"], session_id=self.session_id, workspace=str(self.root / "other"))
        self.agent.fork_session("branch")
        self.assertEqual(self.agent.list_tool_outputs(), [])
        with self.assertRaises(KeyError):
            self.agent.read_tool_output(receipt["id"])
        self.agent.resume_session(self.session_id)
        self.assertEqual(self.agent.read_tool_output(receipt["id"])["content"], "session private output")

    def test_callback_scope_remains_bound_after_agent_switches_sessions(self):
        context = self.agent.tool_context
        self.agent.fork_session("new branch")
        receipt = context.save_tool_output("late output", tool_name="terminal")
        self.assertEqual(context.read_tool_output(receipt["id"])["content"], "late output")
        with self.assertRaises(KeyError):
            self.agent.read_tool_output(receipt["id"])

    def test_output_provider_rejects_model_authorization_arguments(self):
        receipt = self.save("body")
        provider = OutputToolProvider()
        for injected in ({"context": {}}, {"session_id": "other"}, {"workspace": "/"}, {"approved": True}):
            with self.subTest(injected=injected):
                result = asyncio.run(provider.call("read_tool_output", {"output_id": receipt["id"], **injected}, self.agent.tool_context))
                self.assertTrue(json.loads(result)["blocked"])
        context = replace(self.agent.tool_context, approval_mode="never")
        page = json.loads(asyncio.run(provider.call("read_tool_output", {"output_id": receipt["id"]}, context)))
        self.assertTrue(page["ok"])
        self.assertEqual(page["content"], "body")

    def test_page_read_does_not_load_other_chunks_and_checks_integrity(self):
        receipt = self.save("a" * (OUTPUT_CHUNK_CHARS * 3))
        with self.store._connect() as connection:
            connection.execute("UPDATE tool_output_chunks SET content = ? WHERE output_id = ? AND chunk_index = 1", (b"corrupt", receipt["id"]))
        self.assertEqual(self.agent.read_tool_output(receipt["id"], 0, 10)["content"], "a" * 10)
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            self.agent.read_tool_output(receipt["id"], OUTPUT_CHUNK_CHARS, 10)

    def test_failed_chunk_insert_rolls_back_metadata_and_earlier_chunks(self):
        with self.store._connect() as connection:
            connection.executescript("""
                CREATE TRIGGER fail_chunk BEFORE INSERT ON tool_output_chunks
                WHEN NEW.chunk_index = 1 BEGIN SELECT RAISE(ABORT, 'simulated chunk failure'); END;
            """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.save("x" * (OUTPUT_CHUNK_CHARS + 1))
        self.assertEqual(self.agent.list_tool_outputs(), [])
        with self.store._connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tool_output_chunks").fetchone()[0], 0)

    def test_migration_is_additive_and_idempotent(self):
        history = self.store.get_messages(self.session_id, limit=None)
        with self.store._connect() as connection:
            connection.execute("DROP TABLE tool_output_chunks")
            connection.execute("DROP TABLE tool_outputs")
        for _ in range(2):
            store = ConversationStore(str(self.store.path))
            self.assertEqual(store.get_messages(self.session_id, limit=None), history)
            self.assertEqual(store.list_tool_outputs(session_id=self.session_id, workspace=str(self.root)), [])
        self.assertTrue(self.save("after migration")["complete"])

    def test_receipts_survive_resume_and_context_compaction(self):
        receipt = self.save("kept output")
        self.config.override("agent", "context_summary_mode", value="deterministic")
        self.config.override("agent", "compact_keep_recent", value=1)
        for index in range(5):
            self.store.append_message(self.session_id, "user", f"question {index}")
        self.agent.resume_session(self.session_id)
        asyncio.run(self.agent.compact_context())
        resumed = Agent(self.config)
        resumed.resume_session(self.session_id)
        self.assertEqual(resumed.read_tool_output(receipt["id"])["content"], "kept output")


class TerminalOutputTests(OutputFixture):
    def command(self, source):
        script = self.root / "fixture command.py"
        script.write_text(source, encoding="utf-8")
        return f'"{sys.executable}" -u "{script}"'

    def dispatch(self, command, *, timeout=10, context=None):
        return json.loads(asyncio.run(registry.dispatch_async("terminal", {"command": command, "timeout": timeout}, context=context or self.agent.tool_context)))

    def test_nonzero_exit_is_failure_but_stderr_alone_is_not(self):
        command = self.command("import sys\nprint('useful stdout')\nprint('diagnostic', file=sys.stderr)\nsys.exit(7)\n")
        failed = self.dispatch(command)
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["exit_code"], 7)
        self.assertIn("code 7", failed["error"])
        self.assertIn("useful stdout", failed["content"])
        self.assertIn("diagnostic", failed["content"])
        successful = self.dispatch(self.command("import sys\nprint('warning only', file=sys.stderr)\n"))
        self.assertTrue(successful["ok"])
        self.assertEqual(successful["metadata"]["exit_code"], 0)

    def test_large_stdout_and_stderr_are_drained_without_deadlock(self):
        command = self.command("import sys\nsys.stdout.write('OUT-' + 'x' * 200000 + '-OUT-END')\nsys.stderr.write('ERR-' + 'y' * 200000 + '-ERR-END')\n")
        result = self.dispatch(command)
        self.assertTrue(result["ok"])
        self.assertLess(len(result["content"]), 6500)
        refs = result["metadata"]["output_refs"]
        self.assertEqual(set(refs), {"stdout", "stderr"})
        for name, marker in (("stdout", "-OUT-END"), ("stderr", "-ERR-END")):
            self.assertTrue(refs[name]["complete"])
            page = self.agent.read_tool_output(refs[name]["id"], refs[name]["char_count"] - len(marker))
            self.assertEqual(page["content"], marker)

    def test_timeout_returns_partial_output_and_persisted_receipt(self):
        command = self.command("import time\nprint('BEFORE_TIMEOUT', flush=True)\ntime.sleep(30)\n")
        started = time.monotonic()
        result = self.dispatch(command, timeout=1)
        self.assertLess(time.monotonic() - started, 8)
        self.assertFalse(result["ok"])
        self.assertTrue(result["metadata"]["timed_out"])
        self.assertIn("BEFORE_TIMEOUT", result["content"])
        ref = result["metadata"]["output_refs"]["stdout"]
        self.assertFalse(ref["complete"])
        self.assertEqual(ref["status"], "timed_out")
        self.assertIn("BEFORE_TIMEOUT", self.agent.read_tool_output(ref["id"])["content"])

    def test_cancel_saves_partial_logs_and_reraises_cancellation(self):
        command = self.command("import time\nprint('READY_TO_CANCEL', flush=True)\ntime.sleep(30)\n")
        original_write = OutputCapture.write

        async def exercise():
            entered = asyncio.Event()

            def observe(capture, text):
                original_write(capture, text)
                if "READY_TO_CANCEL" in text:
                    entered.set()

            with patch.object(OutputCapture, "write", observe):
                task = asyncio.create_task(registry.dispatch_async("terminal", {"command": command}, context=self.agent.tool_context))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=3)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

        asyncio.run(exercise())
        outputs = self.agent.list_tool_outputs()
        self.assertEqual(outputs[0]["status"], "cancelled")
        self.assertFalse(outputs[0]["complete"])
        self.assertIn("READY_TO_CANCEL", self.agent.read_tool_output(outputs[0]["id"])["content"])

    def test_capture_cap_and_persistence_error_do_not_change_exit_success(self):
        command = self.command("print('x' * 10000 + 'TAIL')\n")
        result = self.dispatch(command, context=replace(self.agent.tool_context, max_output_bytes=64))
        self.assertTrue(result["ok"])
        ref = result["metadata"]["output_refs"]["stdout"]
        self.assertEqual(ref["size_bytes"], 64)
        self.assertFalse(ref["complete"])
        self.assertIn("TAIL", result["content"])
        failed_storage = replace(self.agent.tool_context, save_tool_output=Mock(side_effect=OSError("database full")))
        result = self.dispatch(command, context=failed_storage)
        self.assertTrue(result["ok"])
        self.assertIn("persistence failed", result["metadata"]["warnings"][0])

    def test_invalid_timeout_or_model_capture_override_never_spawns(self):
        with patch("asyncio.create_subprocess_shell") as shell, patch("asyncio.create_subprocess_exec") as execute:
            for arguments in ({"timeout": 0}, {"timeout": -1}, {"max_output_bytes": 100000000}):
                result = json.loads(asyncio.run(registry.dispatch_async("terminal", {"command": "echo example", **arguments}, context=self.agent.tool_context)))
                self.assertFalse(result["ok"])
            shell.assert_not_called()
            execute.assert_not_called()


class OutputProtocolTests(OutputFixture):
    def test_large_result_then_page_read_works_in_both_api_loops(self):
        source = "HEAD_SENTINEL\n" + "x" * 25000 + "TAIL_SENTINEL_98765\n"
        (self.root / "large.txt").write_text(source, encoding="utf-8")
        test = self

        class WireClient(LLMClient):
            def __init__(self, api):
                super().__init__(base_url="https://example.invalid/v1", api_key="offline", model="offline", api_type=api)
                self.api = api
                self.bodies = []
                self.output_id = None

            async def _post_json(self, url, body):
                self.bodies.append(body)
                if len(self.bodies) == 1:
                    name = "read_file"
                    args = {"path": "large.txt"}
                else:
                    if self.api == "chat":
                        tool_results = [item["content"] for item in body["messages"] if item["role"] == "tool"]
                        schemas = [item["function"] for item in body["tools"]]
                    else:
                        tool_results = [item["output"] for item in body["input"] if item.get("type") == "function_call_output"]
                        schemas = body["tools"]
                    test.assertTrue(all(len(result) <= 10000 for result in tool_results))
                    payload = json.loads(tool_results[-1])
                    test.assertTrue(payload["ok"])
                    test.assertIn("read_tool_output", [item["name"] for item in schemas])
                    if len(self.bodies) == 2:
                        ref = payload["metadata"]["output_ref"]
                        self.output_id = ref["id"]
                        name = "read_tool_output"
                        args = {"output_id": self.output_id, "offset": ref["char_count"] - 1000, "limit": 1000}
                    else:
                        test.assertIn("TAIL_SENTINEL_98765", payload["content"])
                        test.assertTrue(payload["metadata"]["eof"])
                        if self.api == "chat":
                            return chat_response("verified output")
                        return {"status": "completed", "output": [{
                            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "verified output"}],
                        }]}
                arguments = json.dumps(args)
                call_id = f"output-call-{len(self.bodies)}"
                if self.api == "chat":
                    return chat_response(finish_reason="tool_calls", tool_calls=[function_call(call_id, name, arguments)])
                return {"status": "completed", "output": [{"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}]}

        for api in ("chat", "responses"):
            with self.subTest(api=api):
                agent = Agent(self.config)
                client = WireClient(api)
                agent.llm = client
                self.assertEqual(asyncio.run(agent.run("read the large file and inspect its tail")), "verified output")
                self.assertEqual(len(client.bodies), 3)
                resumed = Agent(self.config)
                resumed.resume_session(agent.session_id)
                first_page = resumed.read_tool_output(client.output_id)
                self.assertIn("HEAD_SENTINEL", first_page["content"])
                self.assertEqual(len(resumed.list_tool_outputs()), 1)

    def test_paging_tool_remains_in_dynamically_selected_core(self):
        self.agent._tool_selection_active = True
        self.agent._turn_tool_names = {"terminal"}
        schemas = self.agent._selected_tool_schemas()
        self.assertIn("read_tool_output", [item["function"]["name"] for item in schemas])

    def test_storage_disabled_retains_valid_json_without_advertising_a_reader(self):
        self.config.override("storage", "enabled", value=False)
        agent = Agent(self.config)
        agent._refresh_tool_context()
        result = json.loads(agent._bound_tool_result("test", "large" * 10000))
        self.assertTrue(result["ok"])
        self.assertIsNone(result["metadata"]["output_ref"])
        self.assertNotIn("read_tool_output", agent.tool_provider.tool_names())
