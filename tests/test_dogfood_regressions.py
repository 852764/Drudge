from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent import Agent, RunStatus
from agent.llm import LLMClient, create_client
from config import ConfigManager
from main import main, run_query
from tests.fakes import FakeLLM, chat_response, function_call


class DogfoodRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = ConfigManager()
        for section, key, value in (
            ("storage", "path", str(self.root / "sessions.db")),
            ("security", "workspace_root", str(self.root)),
            ("security", "approval_mode", "auto"),
            ("agent", "refusal_review_enabled", False),
            ("agent", "repo_map_enabled", False),
            ("display", "show_tool_calls", False),
            ("display", "show_cost", False),
            ("model", "api_key", "offline"),
        ):
            self.config.override(section, key, value=value)
        self.config.override("toolsets", value=["file"])

    def test_configured_timeout_reaches_both_api_clients(self):
        for api in ("chat", "responses"):
            config = {**self.config.get_model_config(), "api": api, "timeout": 7.5}
            self.assertEqual(create_client(config).timeout, 7.5)

    def test_invalid_timeout_rejected_before_any_request(self):
        for timeout in (True, 0, -1, float("nan"), float("inf"), "7", None):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                create_client({**self.config.get_model_config(), "timeout": timeout})

    def test_no_tools_removes_metadata_mcp_and_local_tools(self):
        self.config.override("agent", "tools_enabled", value=False)
        self.config.override("mcp_servers", value={"fixture": {"command": "never-spawn"}})
        agent = Agent(self.config)
        agent.llm = FakeLLM([chat_response("done")])
        with patch("asyncio.create_subprocess_exec", side_effect=RuntimeError("MCP must not start")) as spawn:
            asyncio.run(agent.run("answer only"))
        spawn.assert_not_called()
        self.assertEqual(agent.tool_provider.schemas(), [])
        self.assertEqual(agent.tool_provider.tool_names(), [])
        self.assertEqual(agent.tool_context.enabled_toolsets, frozenset())
        self.assertIsNone(agent.llm.requests[0]["tools"])
        self.assertNotIn("call update_plan", agent.get_messages()[0]["content"])

    def test_no_tools_does_not_accept_hallucinated_plan_calls(self):
        self.config.override("agent", "tools_enabled", value=False)
        agent = Agent(self.config)
        agent.llm = FakeLLM([
            chat_response(finish_reason="tool_calls", tool_calls=[function_call("call-1", "update_plan", '{"plan":[]}')]),
            chat_response("no tools"),
        ])
        asyncio.run(agent.run("answer only"))
        self.assertEqual(agent.get_plan_state()["revision"], 0)
        result = json.loads(agent.llm.requests[1]["messages"][-1]["content"])
        self.assertFalse(result["ok"])

    def test_workspace_prompt_and_saved_session_use_tool_workspace(self):
        agent = Agent(self.config)
        agent.llm = FakeLLM([chat_response("done")])
        asyncio.run(agent.run("inspect workspace"))
        system = agent.get_messages()[0]["content"]
        self.assertIn(f"Current working directory: {self.root}", system)
        self.assertEqual(agent.store.get_session(agent.session_id)["cwd"], str(self.root))

    def test_turn_budget_is_current_and_does_not_modify_durable_messages(self):
        self.config.override("agent", "max_turns", value=3)
        agent = Agent(self.config)
        agent.llm = FakeLLM([
            chat_response(finish_reason="tool_calls", tool_calls=[function_call("call-1", "read_file", '{"path":"missing.txt"}')]),
            chat_response("done"),
        ])
        asyncio.run(agent.run("inspect"))
        first, second = [request["messages"][0]["content"] for request in agent.llm.requests]
        self.assertIn("3 of 3 turns remaining", first)
        self.assertIn("2 of 3 turns remaining", second)
        self.assertNotIn("turns remaining", agent.get_messages()[0]["content"])
        self.assertNotIn("turns remaining", agent.store.get_messages(agent.session_id)[0]["content"])

    def test_incomplete_chat_text_is_not_success(self):
        for reason in ("length", "content_filter"):
            with self.subTest(reason=reason):
                agent = Agent(self.config)
                agent.llm = FakeLLM([chat_response("partial answer", finish_reason=reason)])
                result = asyncio.run(agent.run("answer"))
                self.assertEqual(agent.run_state.status, RunStatus.FAILED)
                self.assertIn("partial answer", result)
                self.assertIn(reason, agent.run_state.error)

    def test_incomplete_response_calls_never_execute_in_either_api(self):
        arguments = '{"path":"do-not-write.txt","content":"bad"}'
        call = function_call("call-1", "write_file", arguments)
        response_item = {"type": "function_call", "call_id": "call-1", "name": "write_file", "arguments": arguments}
        for api in ("chat", "responses"):
            with self.subTest(api=api):
                if api == "chat":
                    response = chat_response("partial", finish_reason="length", tool_calls=[call])
                else:
                    response = LLMClient._responses_to_chat_response({"status": "incomplete", "output": [response_item]}, "offline")
                agent = Agent(self.config)
                agent.llm = FakeLLM([response])
                asyncio.run(agent.run("write"))
                self.assertEqual(agent.run_state.status, RunStatus.FAILED)
                self.assertFalse((self.root / "do-not-write.txt").exists())
                self.assertFalse(any(message.get("tool_calls") for message in agent.get_messages()))

    def test_incomplete_stream_event_cannot_default_to_completed(self):
        payload = {"type": "response.incomplete", "response": {"output": []}}
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content="data: " + json.dumps(payload) + "\n\n",
        ))
        client = LLMClient("https://example.invalid", "offline", "offline", api_type="responses", transport=transport)
        response = asyncio.run(client.chat([{"role": "user", "content": "hi"}], stream_callback=lambda text: None))
        self.assertEqual(client.extract_finish_reason(response), "length")

    def query(self, responses, *, no_tools=False, cancel=False):
        agent = None
        def create(config, approval_callback=None):
            nonlocal agent
            agent = Agent(config, approval_callback=approval_callback)
            agent.llm = FakeLLM(responses)
            if cancel:
                agent.llm.chat = AsyncMock(side_effect=asyncio.CancelledError)
            return agent
        with patch("main.get_config", return_value=self.config), patch("main.Agent", side_effect=create), patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            code = asyncio.run(run_query("test", no_tools=no_tools))
        return code, agent

    def test_query_exit_codes_match_completed_failed_exhausted_and_cancelled(self):
        for responses, expected, cancel in (
            ([chat_response("done")], 0, False),
            ([RuntimeError("fixture offline error")], 1, False),
            ([], 130, True),
        ):
            with self.subTest(expected=expected):
                code, _ = self.query(responses, cancel=cancel)
                self.assertEqual(code, expected)
        self.config.override("agent", "max_turns", value=1)
        code, _ = self.query([chat_response(finish_reason="tool_calls", tool_calls=[function_call("call-1", "read_file", '{"path":"missing.txt"}')])])
        self.assertEqual(code, 2)

    def test_cli_no_tools_reaches_agent_host_configuration(self):
        code, agent = self.query([chat_response("done")], no_tools=True)
        self.assertEqual(code, 0)
        self.assertEqual(agent.tool_provider.tool_names(), [])

    def test_entrypoint_propagates_single_query_exit_status(self):
        with patch("sys.argv", ["drudge", "-q", "fixture"]), patch("main.run_query", new=AsyncMock(return_value=2)):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
