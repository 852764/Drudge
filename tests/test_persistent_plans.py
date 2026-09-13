from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent import Agent
from agent.cli_renderer import CliRenderer
from agent.llm import LLMClient
from agent.storage import ConversationStore, ContextConflictError
from config import ConfigManager
from main import _handle_command
from tests.fakes import FakeLLM, chat_response, function_call
from tools.plan import normalize_plan
from tools.provider import PlanToolProvider


PLAN = [
    {"step": "Implement", "status": "in_progress", "acceptance": "Offline regression passes"},
    {"step": "Review", "status": "pending"},
]


class PersistentPlanTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = ConfigManager()
        for section, key, value in (
            ("storage", "enabled", True), ("storage", "path", str(self.root / "session.db")),
            ("security", "workspace_root", str(self.root)), ("display", "show_tool_calls", False),
            ("agent", "refusal_review_enabled", False), ("agent", "context_summary_mode", "deterministic"),
        ):
            self.config.override(section, key, value=value)
        self.config.override("toolsets", value=["file"])
        self.agent = Agent(self.config)
        self.agent.llm = FakeLLM([chat_response("ready")])
        asyncio.run(self.agent.run("start"))
        self.session = self.agent.session_id
        self.store = self.agent.store

    def test_survives_turn_compaction_restart_with_evidence(self):
        self.agent.update_plan(PLAN, "initial")
        self.agent.llm = FakeLLM([chat_response("continued")])
        asyncio.run(self.agent.run("continue"))
        self.assertEqual(self.agent.get_plan(), PLAN)
        self.assertIn("SESSION PLAN", self.agent.llm.requests[0]["messages"][0]["content"])
        completed = [{**item, "status": "completed", "evidence": "unittest: exit 0"} for item in PLAN]
        result = self.agent.update_plan(completed, "verified")
        asyncio.run(self.agent.compact_context())
        resumed = Agent(self.config)
        resumed.resume_session(self.session)
        self.assertEqual(resumed.get_plan(), completed)
        self.assertEqual(resumed.get_plan_state()["revision"], result["revision"])
        self.assertIn("unittest: exit 0", resumed.get_messages()[0]["content"])
        returned = resumed.get_plan()
        returned[0]["step"] = "changed"
        self.assertEqual(resumed.get_plan(), completed)

    def test_explicit_clear_is_durable_and_new_session_does_not_inherit(self):
        self.agent.update_plan(PLAN)
        self.agent.update_plan([], "scope changed")
        resumed = Agent(self.config)
        resumed.resume_session(self.session)
        self.assertEqual(resumed.get_plan(), [])
        self.assertEqual(resumed.get_plan_state()["revision"], 2)
        self.agent.new_session()
        self.assertEqual(self.agent.get_plan_state()["revision"], 0)
        self.assertFalse(self.agent.get_plan_state()["persistent"])

    def test_other_session_and_workspace_do_not_inherit(self):
        self.agent.update_plan(PLAN)
        other = self.store.create_session("other", "offline", cwd=str(self.root))
        self.assertEqual(self.store.load_plan(other, workspace=str(self.root))["plan"], [])
        self.config.override("security", "workspace_root", value=str(self.root / "other"))
        resumed = Agent(self.config)
        with self.assertRaisesRegex(RuntimeError, "different workspace"):
            resumed.resume_session(self.session)
        self.assertIsNone(resumed.session_id)

    def test_stale_consumer_cannot_overwrite_or_start_model(self):
        resumed = Agent(self.config)
        resumed.resume_session(self.session)
        self.agent.update_plan(PLAN)
        with self.assertRaises(ContextConflictError):
            resumed.update_plan([])
        self.assertEqual(resumed.get_plan(), [])
        resumed.llm = FakeLLM([chat_response("should not run")])
        with self.assertRaisesRegex(RuntimeError, "plan changed"):
            asyncio.run(resumed.run("continue"))
        self.assertEqual(resumed.llm.requests, [])
        self.assertEqual(self.store.load_plan(self.session, workspace=str(self.root))["plan"], PLAN)

    def test_stale_message_head_prevents_plan_commit(self):
        self.store.append_message(self.session, "user", "external change")
        with self.assertRaises(ContextConflictError):
            self.agent.update_plan(PLAN)
        self.assertEqual(self.agent.get_plan_state()["revision"], 0)

    def test_write_and_audit_failure_preserve_memory_and_database(self):
        self.agent.update_plan(PLAN)
        before = self.agent.get_plan_state()
        with patch.object(self.store, "save_plan", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaises(sqlite3.Error):
                self.agent.update_plan([])
        self.assertEqual(self.agent.get_plan_state(), before)
        self.agent._current_run_id = "missing-run"
        with self.assertRaises(ValueError):
            self.agent.update_plan([])
        self.assertEqual(self.agent.get_plan_state(), before)
        self.assertEqual(self.store.load_plan(self.session, workspace=str(self.root))["revision"], 1)

    def test_fork_clones_plan_without_linking_future_updates_or_authorization(self):
        self.agent.update_plan(PLAN)
        self.agent._session_approvals.add(("terminal", "echo test"))
        child = self.agent.fork_session("branch")
        self.assertNotEqual(child["id"], self.session)
        self.assertEqual(self.agent.get_plan(), PLAN)
        self.assertEqual(self.agent._session_approvals, set())
        self.agent.update_plan([])
        self.assertEqual(self.store.load_plan(self.session, workspace=str(self.root))["plan"], PLAN)

    def test_stale_fork_is_atomic(self):
        other = Agent(self.config)
        other.resume_session(self.session)
        self.agent.update_plan(PLAN)
        before = self.store.list_sessions()
        with self.assertRaises(ContextConflictError):
            other.fork_session("stale")
        self.assertEqual(self.store.list_sessions(), before)
        self.assertEqual(other.session_id, self.session)

    def test_corrupt_and_future_plans_fail_before_switching_session(self):
        self.agent.update_plan(PLAN)
        with self.store._connect() as conn:
            conn.execute("UPDATE session_plans SET sha256 = 'corrupt'")
        resumed = Agent(self.config)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            resumed.resume_session(self.session)
        self.assertIsNone(resumed.session_id)
        with self.store._connect() as conn:
            conn.execute("UPDATE session_plans SET schema_version = 99")
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            resumed.resume_session(self.session)

    def test_migration_is_additive_idempotent_and_preserves_history(self):
        messages = self.store.get_messages(self.session, limit=None)
        with self.store._connect() as conn:
            conn.execute("DROP TABLE session_plans")
        first = ConversationStore(str(self.store.path))
        self.assertEqual(first.load_plan(self.session, workspace=str(self.root))["revision"], 0)
        self.agent.update_plan(PLAN)
        second = ConversationStore(str(self.store.path))
        self.assertEqual(second.get_messages(self.session, limit=None), messages)
        self.assertEqual(second.load_plan(self.session, workspace=str(self.root))["plan"], PLAN)

    def test_no_storage_keeps_plan_across_turns(self):
        self.config.override("storage", "enabled", value=False)
        agent = Agent(self.config)
        agent.update_plan(PLAN)
        agent.llm = FakeLLM([chat_response("one"), chat_response("two")])
        asyncio.run(agent.run("one"))
        asyncio.run(agent.run("two"))
        self.assertEqual(agent.get_plan(), PLAN)
        self.assertFalse(agent.get_plan_state()["persistent"])

    def test_both_wire_protocols_persist_acceptance_and_audit(self):
        case = self

        class WireClient(LLMClient):
            def __init__(self, api):
                super().__init__(base_url="https://example.invalid", api_key="offline", model="offline", api_type=api)
                self.api = api
                self.count = 0

            async def _post_json(self, url, body):
                self.count += 1
                if self.count == 1:
                    arguments = json.dumps({"plan": PLAN})
                    if self.api == "chat":
                        return chat_response(tool_calls=[function_call("plan-call", "update_plan", arguments)], finish_reason="tool_calls")
                    return {"status": "completed", "output": [{"type": "function_call", "call_id": "plan-call", "name": "update_plan", "arguments": arguments}]}
                if self.api == "chat":
                    results = [m["content"] for m in body["messages"] if m["role"] == "tool"]
                else:
                    results = [m["output"] for m in body["input"] if m.get("type") == "function_call_output"]
                case.assertTrue(json.loads(results[-1])["ok"])
                if self.api == "chat":
                    return chat_response("recorded")
                return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "recorded"}]}]}

        for api in ("chat", "responses"):
            with self.subTest(api=api):
                agent = Agent(self.config)
                agent.llm = WireClient(api)
                self.assertEqual(asyncio.run(agent.run("plan work")), "recorded")
                resumed = Agent(self.config)
                resumed.resume_session(agent.session_id)
                self.assertEqual(resumed.get_plan(), PLAN)
                self.assertIn("plan_updated", [event["kind"] for event in agent.get_trace()["events"]])


class PlanValidationTests(unittest.TestCase):
    def test_invalid_fields_types_sizes_and_missing_evidence(self):
        invalid = [
            None, {}, [1], [{"step": 123, "status": "in_progress"}],
            [{"step": "x", "status": "pending"}], [{"step": "x", "status": "in_progress", "context": {}}],
            [{"step": "x" * 501, "status": "in_progress"}],
            [{"step": "x", "status": "in_progress"}] * 33,
            [{"step": "x", "status": "completed", "acceptance": "tests pass"}],
            [{"step": "x", "status": "completed", "evidence": "x" * 4001}],
            [{"step": "x", "status": "completed", "evidence": "x" * 4000}] * 10,
        ]
        for plan in invalid:
            with self.subTest(plan=str(plan)[:100]), self.assertRaises(ValueError):
                normalize_plan(plan)
        for explanation in (True, [], "x" * 2001):
            with self.assertRaises(ValueError):
                normalize_plan([], explanation)

    def test_provider_rejects_host_overrides_even_when_called_directly(self):
        callback = Mock()
        provider = PlanToolProvider(callback)
        for field in ("context", "approved", "workspace", "session_id", "revision", "run_id"):
            result = json.loads(asyncio.run(provider.call("update_plan", {"plan": PLAN, field: "override"}, None)))
            self.assertFalse(result["ok"])
        callback.assert_not_called()

    def test_provider_reports_storage_failures_as_tool_failures(self):
        provider = PlanToolProvider(Mock(side_effect=sqlite3.OperationalError("disk full")))
        result = json.loads(asyncio.run(provider.call("update_plan", {"plan": PLAN}, None)))
        self.assertFalse(result["ok"])
        self.assertIn("disk full", result["error"])

    def test_cli_read_only_control_escaping_empty_and_usage(self):
        agent = Mock()
        agent.get_plan_state.return_value = {
            "revision": 2, "persistent": True, "explanation": "why",
            "plan": [{"step": "test\x1b[2J", "status": "completed", "acceptance": "ok", "evidence": "exit 0"}],
        }
        def render(command, target=agent):
            stream = io.StringIO()
            self.assertFalse(asyncio.run(_handle_command(command, ConfigManager(), target, renderer=CliRenderer(stream=stream, pretty=False))))
            return stream.getvalue()
        output = render("/plan")
        self.assertIn("Evidence (self-reported)", output)
        self.assertIn("\\x1b", output)
        self.assertNotIn("\x1b", output)
        self.assertIn("/plan", render("/help"))
        agent.get_plan_state.reset_mock()
        self.assertIn("Usage", render("/plan clear"))
        agent.get_plan_state.assert_not_called()
        agent.update_plan.assert_not_called()
        self.assertIn("unavailable", render("/plan", None))
        agent.get_plan_state.return_value["plan"] = []
        self.assertIn("No current plan", render("/plan"))


if __name__ == "__main__":
    unittest.main()
