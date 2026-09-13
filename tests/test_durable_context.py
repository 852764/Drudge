from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from agent.conversation import encode_context, repair_tool_transactions
from agent.llm import LLMClient
from agent.storage import ContextConflictError
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call


def assistant_calls(*ids):
    return {"role": "assistant", "content": "", "tool_calls": [
        function_call(call_id, "write_file", '{"path":"note.txt","content":"written"}') for call_id in ids
    ]}


def tool_result(call_id, content="recorded result"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class DurableContextTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = ConfigManager()
        self.config.override("storage", "enabled", value=True)
        self.config.override("storage", "path", value=str(self.root / "session.db"))
        self.config.override("security", "workspace_root", value=str(self.root))
        self.config.override("security", "approval_mode", value="auto")
        self.config.override("toolsets", value=["file"])
        self.config.override("display", "show_tool_calls", value=False)
        self.config.override("agent", "refusal_review_enabled", value=False)
        self.config.override("agent", "compact_keep_recent", value=2)
        self.config.override("agent", "context_summary_mode", value="deterministic")
        self.agent = Agent(self.config)
        self.store = self.agent.store

    def seed(self, messages=None):
        if messages is None:
            messages = [{"role": "system", "content": "original system"}]
            for index in range(6):
                messages.extend([
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ])
        session_id = self.store.create_session("original", "offline", cwd=str(self.root))
        for message in messages:
            self.store.append_message(
                session_id, message["role"], message.get("content"),
                tool_call_id=message.get("tool_call_id"),
                metadata={key: message[key] for key in ("tool_calls", "provider_items") if key in message},
            )
        self.agent.resume_session(session_id)
        return session_id

    def test_compacted_context_and_full_history_both_survive_restart(self):
        session_id = self.seed()
        raw_before = self.store.get_messages(session_id, limit=None)
        turns = self.agent.get_token_usage()["turns"]
        result = asyncio.run(self.agent.compact_context())
        compacted = self.agent.get_messages()
        self.assertIsNotNone(result["checkpoint_id"])
        self.assertLess(len(compacted), len(raw_before))
        self.assertEqual(self.store.get_messages(session_id, limit=None), raw_before)
        resumed = Agent(self.config)
        info = resumed.resume_session(session_id)
        self.assertEqual(resumed.get_messages(), compacted)
        self.assertEqual(resumed.get_token_usage()["turns"], turns)
        self.assertEqual(info["context_checkpoint_id"], result["checkpoint_id"])
        self.assertEqual(resumed.get_status()["last_compaction"]["mode"], "fallback")

    def test_resume_appends_only_raw_tail_after_checkpoint(self):
        session_id = self.seed()
        asyncio.run(self.agent.compact_context())
        expected = self.agent.get_messages()
        self.agent.llm = FakeLLM([chat_response("tail answer")])
        asyncio.run(self.agent.run("tail question"))
        resumed = Agent(self.config)
        resumed.resume_session(session_id)
        self.assertEqual(resumed.get_messages()[1:-2], expected[1:])
        self.assertEqual([item["content"] for item in resumed.get_messages()[-2:]], ["tail question", "tail answer"])
        self.assertEqual(resumed.get_token_usage()["turns"], 7)

    def test_repeated_compaction_does_not_reintroduce_old_raw_messages(self):
        session_id = self.seed()
        first = asyncio.run(self.agent.compact_context())
        self.agent.llm = FakeLLM([chat_response("new answer")])
        asyncio.run(self.agent.run("new question"))
        second = asyncio.run(self.agent.compact_context())
        self.assertGreater(second["checkpoint_id"], first["checkpoint_id"])
        resumed = Agent(self.config)
        resumed.resume_session(session_id)
        self.assertEqual(resumed.get_messages()[1:], self.agent.get_messages()[1:])
        self.assertEqual(len(self.store.get_messages(session_id, limit=None)), 15)

    def test_checkpoint_write_failure_keeps_original_working_context(self):
        self.seed()
        before = copy.deepcopy(self.agent.get_messages())
        with patch.object(self.store, "save_context_checkpoint", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "not saved"):
                asyncio.run(self.agent.compact_context())
        self.assertEqual(self.agent.get_messages(), before)
        self.assertIsNone(self.agent.get_status()["context_checkpoint_id"])
        self.assertIsNone(self.agent._active_task)

    def test_corrupt_and_future_checkpoints_fall_back_to_raw_history(self):
        session_id = self.seed()
        asyncio.run(self.agent.compact_context())
        with self.store._connect() as conn:
            original = dict(conn.execute("SELECT * FROM context_checkpoints").fetchone())
        malformed = '{"not":"a message list"}'
        cases = [
            ("messages_json = ?", ("broken json",)),
            ("schema_version = ?", (99,)),
            ("through_message_id = ?", (1000000,)),
            ("messages_json = ?, content_sha256 = ?", (malformed, self.store._context_digest(malformed, original["through_message_id"], original["metadata_json"]))),
            ("metadata_json = ?", ("[]",)),
        ]
        for assignment, parameters in cases:
            with self.subTest(assignment=assignment):
                with self.store._connect() as conn:
                    conn.execute(
                        "UPDATE context_checkpoints SET messages_json=?, content_sha256=?, schema_version=?, through_message_id=?, metadata_json=?",
                        tuple(original[key] for key in ("messages_json", "content_sha256", "schema_version", "through_message_id", "metadata_json")),
                    )
                    conn.execute(f"UPDATE context_checkpoints SET {assignment}", parameters)
                resumed = Agent(self.config)
                info = resumed.resume_session(session_id)
                self.assertIn("restored raw history", info["context_warning"])
                self.assertEqual(len(resumed.get_messages()), 13)
                self.assertEqual(resumed.get_messages()[1]["content"], "question 0")

    def test_stale_compactor_does_not_hide_external_message(self):
        session_id = self.seed()
        before = copy.deepcopy(self.agent.get_messages())
        self.store.append_message(session_id, "user", "external correction")
        with self.assertRaisesRegex(RuntimeError, "another consumer"):
            asyncio.run(self.agent.compact_context())
        self.assertEqual(self.agent.get_messages(), before)
        self.assertIsNone(self.store.load_session_context(session_id)["checkpoint"])
        self.assertEqual(self.store.get_messages(session_id, limit=None)[-1]["content"], "external correction")

    def test_stale_checkpoint_head_is_detected_without_new_raw_messages(self):
        session_id = self.seed()
        other = Agent(self.config)
        other.resume_session(session_id)
        asyncio.run(self.agent.compact_context())
        with self.assertRaisesRegex(RuntimeError, "another consumer"):
            asyncio.run(other.compact_context())
        self.assertEqual(self.store.load_session_context(session_id)["checkpoint_id"], self.agent._context_checkpoint_id)

    def test_stale_turn_stops_before_model_or_tool_execution(self):
        session_id = self.seed()
        self.store.append_message(session_id, "user", "external correction")
        fake = FakeLLM([chat_response("should not run")])
        self.agent.llm = fake
        with self.assertRaises(ContextConflictError):
            asyncio.run(self.agent.run("local stale request"))
        self.assertEqual(fake.requests, [])
        self.assertEqual(self.store.get_messages(session_id, limit=None)[-1]["content"], "external correction")

    def test_recovery_audit_entries_and_checkpoint_commit_together(self):
        session_id = self.seed()
        before = self.store.get_messages(session_id, limit=None)
        with self.assertRaises(ValueError):
            self.store.save_context_checkpoint(
                session_id, self.agent.get_messages(), expected_message_id=self.agent._last_message_id,
                expected_checkpoint_id=None, repair_messages=[tool_result("one"), {"role": "user", "content": "invalid"}],
            )
        self.assertEqual(self.store.get_messages(session_id, limit=None), before)
        self.assertIsNone(self.store.load_session_context(session_id)["checkpoint"])

    def test_manual_compaction_is_cancellable_and_blocks_concurrent_context_changes(self):
        self.seed()
        self.config.override("agent", "context_summary_mode", value="llm")
        before = copy.deepcopy(self.agent.get_messages())

        async def exercise():
            entered = asyncio.Event()
            blocker = asyncio.Event()
            agent = self.agent

            class BlockingLLM(FakeLLM):
                async def chat(self, *args, **kwargs):
                    entered.set()
                    await blocker.wait()
                    return chat_response("summary")

            agent.llm = BlockingLLM([])
            task = asyncio.create_task(agent.compact_context())
            await asyncio.wait_for(entered.wait(), 2)
            for action in (agent.new_session, lambda: agent.resume_session(agent.session_id), agent.fork_session):
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    action()
            with self.assertRaisesRegex(RuntimeError, "busy"):
                await agent.run("concurrent prompt")
            with self.assertRaisesRegex(RuntimeError, "busy"):
                await agent.compact_context()
            agent.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(exercise())
        self.assertEqual(self.agent.get_messages(), before)
        self.assertIsNone(self.store.load_session_context(self.agent.session_id)["checkpoint"])
        self.config.override("agent", "context_summary_mode", value="deterministic")
        self.assertIsNotNone(asyncio.run(self.agent.compact_context())["checkpoint_id"])

    def test_resume_refreshes_project_instructions_not_checkpoint_authorization(self):
        instruction = self.root / "AGENTS.md"
        instruction.write_text("old project rule", encoding="utf-8")
        session_id = self.seed()
        asyncio.run(self.agent.compact_context())
        instruction.write_text("current project rule", encoding="utf-8")
        self.config.override("security", "approval_mode", value="never")
        resumed = Agent(self.config)
        resumed.resume_session(session_id)
        self.assertIn("current project rule", resumed.get_messages()[0]["content"])
        self.assertNotIn("old project rule", resumed.get_messages()[0]["content"])
        self.assertEqual(resumed.tool_context.approval_mode, "never")

    def test_fork_preserves_working_context_but_not_approvals_or_undo(self):
        parent_id = self.seed()
        asyncio.run(self.agent.compact_context())
        self.agent._session_approvals.add(("write_file", "note.txt"))
        self.store.create_task(parent_id, "parent task")
        self.store.record_file_revision(
            session_id=parent_id, run_id=None, path=str(self.root / "note.txt"),
            operation="write_file", before_content=None, after_content="parent change",
        )
        (self.root / "note.txt").write_text("shared workspace", encoding="utf-8")
        parent_context = copy.deepcopy(self.agent.get_messages())
        parent_history = self.store.get_messages(parent_id, limit=None)
        child = self.agent.fork_session("approach B")
        self.assertEqual(child["metadata"]["parent_session_id"], parent_id)
        self.assertNotEqual(self.agent.session_id, parent_id)
        self.assertEqual(child["title"], "approach B")
        self.assertEqual(self.agent.get_messages()[1:], parent_context[1:])
        self.assertEqual(self.agent._session_approvals, set())
        self.assertEqual(self.agent.list_file_revisions(), [])
        self.assertEqual(self.agent.list_tasks(), [])
        self.assertEqual(self.store.get_messages(parent_id, limit=None), parent_history)
        self.assertEqual((self.root / "note.txt").read_text(encoding="utf-8"), "shared workspace")
        self.agent.llm = FakeLLM([chat_response("branch answer")])
        asyncio.run(self.agent.run("branch question"))
        self.assertEqual(self.store.get_messages(parent_id, limit=None), parent_history)
        self.agent.resume_session(parent_id)
        self.assertEqual(self.agent.get_messages()[1:], parent_context[1:])

    def test_fork_keeps_explicitly_active_skills(self):
        self.seed()
        skill = self.root / ".drudge" / "skills" / "review" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("---\nname: review\ndescription: Review\n---\nReview carefully", encoding="utf-8")
        self.agent.activate_skill("review")
        self.agent.fork_session()
        self.assertEqual(self.agent.active_skill_names, ["review"])

    def test_fork_failure_rolls_back_child_without_switching_parent(self):
        parent_id = self.seed()
        original = self.store._insert_context_message
        calls = []

        def fail_second(conn, session_id, message):
            calls.append(message)
            if len(calls) == 2:
                raise OSError("simulated branch failure")
            return original(conn, session_id, message)

        with patch.object(self.store, "_insert_context_message", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.agent.fork_session()
        self.assertEqual(self.agent.session_id, parent_id)
        self.assertEqual(len(self.store.list_sessions()), 1)

    def test_fork_rejects_stale_parent_and_disabled_storage(self):
        session_id = self.seed()
        self.store.append_message(session_id, "user", "new external input")
        with self.assertRaises(ContextConflictError):
            self.agent.fork_session()
        self.assertEqual(len(self.store.list_sessions()), 1)
        self.config.override("storage", "enabled", value=False)
        with self.assertRaises(RuntimeError):
            Agent(self.config).fork_session()

    def test_cancelled_tool_is_repaired_before_next_user_without_reexecution(self):
        self.seed([{"role": "user", "content": "original task"}])
        self.agent.llm = FakeLLM([
            chat_response(finish_reason="tool_calls", tool_calls=assistant_calls("done", "pending")["tool_calls"]),
            chat_response("continued"),
        ])
        executed = []

        async def exercise():
            entered = asyncio.Event()
            blocker = asyncio.Event()

            async def tool_call(name, arguments, **kwargs):
                executed.append(name)
                if len(executed) == 2:
                    (self.root / "side-effect.txt").write_text("already happened", encoding="utf-8")
                    entered.set()
                    await blocker.wait()
                return '{"ok":true,"content":"done","error":null,"metadata":{}}'

            with patch.object(self.agent.tool_provider, "call", side_effect=tool_call):
                task = asyncio.create_task(self.agent.run("perform edits"))
                await asyncio.wait_for(entered.wait(), 2)
                self.agent.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(await self.agent.run("inspect current state"), "continued")

        asyncio.run(exercise())
        self.assertEqual(len(executed), 2)
        request = self.agent.llm.requests[-1]["messages"]
        new_user = next(index for index, item in enumerate(request) if item.get("content") == "inspect current state")
        recovered = request[new_user - 1]
        self.assertEqual(recovered["tool_call_id"], "pending")
        self.assertTrue(json.loads(recovered["content"])["metadata"]["outcome_unknown"])
        self.assertEqual((self.root / "side-effect.txt").read_text(encoding="utf-8"), "already happened")

    def test_context_table_migration_is_additive_and_idempotent(self):
        session_id = self.seed()
        before = self.store.get_messages(session_id, limit=None)
        with self.store._connect() as conn:
            conn.execute("DROP TABLE context_checkpoints")
        for _ in range(2):
            restarted = Agent(self.config)
            info = restarted.resume_session(session_id)
            self.assertIsNone(info["context_checkpoint_id"])
            self.assertEqual(restarted.store.get_messages(session_id, limit=None), before)
        self.assertIsNotNone(asyncio.run(restarted.compact_context())["checkpoint_id"])

    def test_checkpoint_checksum_binds_message_watermark_and_metadata(self):
        session_id = self.seed()
        asyncio.run(self.agent.compact_context())
        with self.store._connect() as conn:
            # Still a real message ID in the same session, but not the saved
            # cutoff. Without binding the watermark this would duplicate tail.
            conn.execute("UPDATE context_checkpoints SET through_message_id = through_message_id - 1")
        restored = self.store.load_session_context(session_id)
        self.assertIn("checksum mismatch", restored["warning"])
        self.assertIsNone(restored["checkpoint"])

    def test_checkpoint_resume_and_fork_preserve_both_wire_protocols(self):
        class WireClient(LLMClient):
            def __init__(self, api):
                super().__init__(base_url="https://example.invalid/v1", api_key="offline", model="offline", api_type=api)
                self.api = api
                self.bodies = []

            async def _post_json(self, url, body):
                self.bodies.append(body)
                if self.api == "chat":
                    return chat_response("resumed")
                return {"status": "completed", "output": [{
                    "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "resumed"}],
                }]}

        for api in ("chat", "responses"):
            with self.subTest(api=api):
                calls = assistant_calls("done", "pending")
                calls["provider_items"] = [{"type": "reasoning", "encrypted_content": "opaque-state"}] + [{
                    "type": "function_call", "call_id": call["id"], **call["function"],
                } for call in calls["tool_calls"]]
                session_id = self.seed([
                    {"role": "user", "content": "old requirement"}, {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "current request"}, calls,
                    tool_result("done"), {"role": "user", "content": "after interruption"},
                ])
                self.config.override("agent", "compact_keep_recent", value=3)
                asyncio.run(self.agent.compact_context())
                resumed = Agent(self.config)
                resumed.resume_session(session_id)
                resumed.fork_session("wire branch")
                client = WireClient(api)
                resumed.llm = client
                self.assertEqual(asyncio.run(resumed.run("continue branch")), "resumed")
                body = client.bodies[0]
                if api == "chat":
                    items = body["messages"]
                    index = next(i for i, item in enumerate(items) if item.get("tool_calls"))
                    self.assertEqual([item["tool_call_id"] for item in items[index + 1:index + 3]], ["done", "pending"])
                    self.assertNotIn("provider_items", json.dumps(items))
                    unknown = json.loads(items[index + 2]["content"])
                else:
                    items = body["input"]
                    outputs = [item for item in items if item.get("type") == "function_call_output"]
                    self.assertEqual([item["call_id"] for item in outputs], ["done", "pending"])
                    self.assertEqual(sum(item.get("type") == "function_call" for item in items), 2)
                    self.assertIn("opaque-state", json.dumps(items))
                    unknown = json.loads(outputs[1]["output"])
                self.assertTrue(unknown["metadata"]["outcome_unknown"])

    def test_no_storage_compaction_remains_in_memory(self):
        self.config.override("storage", "enabled", value=False)
        agent = Agent(self.config)
        agent._messages = [{"role": "user", "content": str(index)} for index in range(8)]
        result = asyncio.run(agent.compact_context())
        self.assertIsNone(result["checkpoint_id"])
        self.assertLess(len(agent.get_messages()), 8)

    def test_multiple_compaction_keeps_recent_tool_group_complete(self):
        self.seed([
            {"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"},
            assistant_calls("one", "two"), tool_result("one"), tool_result("two"),
        ])
        asyncio.run(self.agent.compact_context())
        self.assertEqual([item["role"] for item in self.agent.get_messages()[-3:]], ["assistant", "tool", "tool"])
        repaired, inserted, report = repair_tool_transactions(self.agent.get_messages())
        self.assertEqual(inserted, [])
        self.assertFalse(any(report.values()))


class ToolTransactionRecoveryTests(unittest.TestCase):
    def test_recovery_inserts_at_each_transaction_boundary_and_is_idempotent(self):
        original = [
            assistant_calls("one", "two"), tool_result("one"),
            {"role": "user", "content": "later request"},
            assistant_calls("three"), {"role": "assistant", "content": "later answer"},
            tool_result("orphan"), tool_result("one", "duplicate"),
        ]
        before = copy.deepcopy(original)
        repaired, inserted, report = repair_tool_transactions(original)
        self.assertEqual(original, before)
        self.assertEqual([item["tool_call_id"] for item in inserted], ["two", "three"])
        self.assertEqual(repaired[2]["tool_call_id"], "two")
        self.assertEqual(repaired[5]["tool_call_id"], "three")
        self.assertEqual(report["dropped_results"], 2)
        again, new_results, new_report = repair_tool_transactions(repaired)
        self.assertEqual(again, repaired)
        self.assertEqual(new_results, [])
        self.assertFalse(any(new_report.values()))

    def test_provider_only_call_is_normalized_for_chat_without_losing_responses_state(self):
        provider_items = [
            {"type": "reasoning", "encrypted_content": "opaque-provider-state"},
            {"type": "function_call", "call_id": "one", "name": "read_file", "arguments": "{}"},
        ]
        repaired, inserted, report = repair_tool_transactions([
            {"role": "assistant", "content": "", "provider_items": provider_items},
            {"role": "user", "content": "next"},
        ])
        self.assertEqual(report["normalized_calls"], 1)
        self.assertEqual(repaired[0]["provider_items"], provider_items)
        self.assertEqual(inserted[0]["tool_call_id"], "one")
        chat = LLMClient._messages_to_chat_input(repaired)
        self.assertEqual(chat[0]["tool_calls"][0]["id"], "one")
        self.assertNotIn("provider_items", chat[0])
        responses = LLMClient._messages_to_responses_input(repaired)
        self.assertEqual(sum(item.get("type") == "function_call" for item in responses), 1)
        self.assertEqual(responses[2]["type"], "function_call_output")

    def test_checkpoint_serialization_does_not_capture_host_context(self):
        encoded = encode_context([{"role": "user", "content": "hello", "context": object(), "approved": True}])
        self.assertEqual(json.loads(encoded), [{"role": "user", "content": "hello"}])

    def test_duplicate_call_ids_are_reported_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "duplicate call ID"):
            repair_tool_transactions([assistant_calls("same", "same")])
