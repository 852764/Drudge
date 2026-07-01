from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent, RunStatus
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call


def test_config(workspace: str, *, max_turns: int = 5) -> ConfigManager:
    config = ConfigManager()
    config.override("storage", "enabled", value=False)
    config.override("display", "show_tool_calls", value=False)
    config.override("agent", "refusal_review_enabled", value=False)
    config.override("agent", "max_turns", value=max_turns)
    config.override("security", "workspace_root", value=workspace)
    config.override("toolsets", value=["file"])
    return config


class AgentLoopTests(unittest.TestCase):
    def test_tool_loop_returns_only_terminal_answer(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "sample.txt").write_text("hello", encoding="utf-8")
            fake = FakeLLM([
                chat_response(
                    "I will inspect it.",
                    finish_reason="tool_calls",
                    tool_calls=[function_call("call-1", "read_file", '{"path":"sample.txt"}')],
                ),
                chat_response("done"),
            ])
            agent = Agent(test_config(workspace))
            agent.llm = fake

            result = asyncio.run(agent.run("read sample.txt"))

            self.assertEqual(result, "done")
            self.assertEqual(agent.get_run_state().status, RunStatus.COMPLETED)
            statuses = [event.status for event in agent.get_run_state().events]
            self.assertIn(RunStatus.EXECUTING_TOOLS, statuses)
            second_messages = fake.requests[1]["messages"]
            self.assertEqual(second_messages[-1]["role"], "tool")
            self.assertIn("hello", second_messages[-1]["content"])

    def test_max_turns_is_explicit_state(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "sample.txt").write_text("hello", encoding="utf-8")
            call = lambda number: chat_response(
                finish_reason="tool_calls",
                tool_calls=[function_call(f"call-{number}", "read_file", '{"path":"sample.txt"}')],
            )
            agent = Agent(test_config(workspace, max_turns=2))
            agent.llm = FakeLLM([call(1), call(2)])

            result = asyncio.run(agent.run("keep reading"))

            self.assertIn("maximum turns", result)
            self.assertEqual(agent.get_run_state().status, RunStatus.MAX_TURNS)

    def test_model_failure_is_explicit_state(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            agent.llm = FakeLLM([RuntimeError("provider down")])

            result = asyncio.run(agent.run("hello"))

            self.assertIn("provider down", result)
            self.assertEqual(agent.get_run_state().status, RunStatus.FAILED)

    def test_empty_model_response_is_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            agent.llm = FakeLLM([chat_response("")])

            result = asyncio.run(agent.run("hello"))

            self.assertIn("neither text nor tool calls", result)
            self.assertEqual(agent.get_run_state().status, RunStatus.FAILED)

    def test_malformed_tool_arguments_are_returned_to_model(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call("call-1", "read_file", "not-json")],
                ),
                chat_response("recovered"),
            ])
            agent = Agent(test_config(workspace))
            agent.llm = fake

            result = asyncio.run(agent.run("read"))

            self.assertEqual(result, "recovered")
            payload = json.loads(fake.requests[1]["messages"][-1]["content"])
            self.assertIn("not valid JSON", payload["error"])

    def test_local_status_reports_context_and_session(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            agent.llm = FakeLLM([chat_response("done")])
            asyncio.run(agent.run("hello"))

            status = agent.get_status()

            self.assertEqual(status["run_status"], "completed")
            self.assertEqual(status["message_count"], 3)
            self.assertGreater(status["estimated_context_tokens"], 0)
            self.assertEqual(status["workspace"], workspace)


if __name__ == "__main__":
    unittest.main()
