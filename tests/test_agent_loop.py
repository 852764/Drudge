from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent, RunStatus
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call


class StreamingFakeLLM(FakeLLM):
    def __init__(self, responses, chunks):
        super().__init__(responses)
        self.chunks = list(chunks)

    async def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        stream_callback=None,
        cancel_event=None,
    ):
        response = await super().chat(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            stream_callback=stream_callback,
            cancel_event=cancel_event,
        )
        chunks = self.chunks.pop(0) if self.chunks else []
        if stream_callback:
            for chunk in chunks:
                emitted = stream_callback(chunk)
                if inspect.isawaitable(emitted):
                    await emitted
        return response


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
    def test_streaming_think_tags_are_never_printed(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            agent.llm = StreamingFakeLLM(
                [chat_response("<think>private</think>visible")],
                [["<thi", "nk>private</th", "ink>visible"]],
            )
            streamed = []

            result = asyncio.run(agent.run("question", stream_callback=streamed.append))

            self.assertEqual(result, "visible")
            self.assertEqual("".join(streamed), "visible")
            self.assertNotIn("think", "".join(streamed).lower())

    def test_degenerate_reasoning_stream_is_stopped_and_retried(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = test_config(workspace)
            config.override("agent", "reasoning_tag_max_chars", value=5)
            agent = Agent(config)
            agent.llm = StreamingFakeLLM(
                [
                    chat_response("<think>123456789"),
                    chat_response("recovered answer"),
                ],
                [
                    ["<think>", "123456"],
                    ["recovered ", "answer"],
                ],
            )
            streamed = []

            result = asyncio.run(agent.run("question", stream_callback=streamed.append))

            self.assertEqual(result, "recovered answer")
            self.assertEqual("".join(streamed), "recovered answer")
            self.assertEqual(len(agent.llm.requests), 2)

    def test_reasoning_only_response_recovers_with_final_answer(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response("<think>reasoning until truncated"),
                chat_response("final answer"),
            ])
            agent = Agent(test_config(workspace))
            agent.llm = fake
            streamed = []

            result = asyncio.run(agent.run("answer me", stream_callback=streamed.append))

            self.assertEqual(result, "final answer")
            self.assertEqual(streamed, ["final answer"])
            self.assertEqual(agent.get_token_usage()["total_tokens"], 20)
            self.assertIn("Do not emit <think> tags", fake.requests[1]["messages"][0]["content"])
            self.assertNotIn("<think>", str(agent.get_messages()))

    def test_think_block_is_removed_from_visible_answer(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            agent.llm = FakeLLM([
                chat_response("<think>private chain</think>The answer is 42."),
            ])

            result = asyncio.run(agent.run("question"))

            self.assertEqual(result, "The answer is 42.")
            self.assertEqual(agent.get_messages()[-1]["content"], "The answer is 42.")

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

    def test_manual_compaction_uses_llm_summary(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = test_config(workspace)
            config.override("agent", "compact_keep_recent", value=3)
            fake = FakeLLM([
                chat_response("<think>hidden</think>## Decisions\n- Keep SQLite."),
            ])
            agent = Agent(config)
            agent.llm = fake
            agent._messages = [{"role": "system", "content": "sys"}] + [
                {"role": "user", "content": f"message {index}"}
                for index in range(10)
            ]

            result = asyncio.run(agent.compact_context())

            self.assertEqual(result["mode"], "llm")
            self.assertEqual(result["summarized_messages"], 7)
            self.assertEqual(result["summary_tokens"], 10)
            self.assertIn("Keep SQLite", agent.get_messages()[1]["content"])
            self.assertNotIn("<think>", agent.get_messages()[1]["content"])
            self.assertIn("message 0", fake.requests[0]["messages"][1]["content"])
            self.assertIsNone(fake.requests[0]["tools"])
            self.assertEqual(agent.get_token_usage()["total_tokens"], 10)

    def test_manual_compaction_falls_back_when_summary_model_fails(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = test_config(workspace)
            config.override("agent", "compact_keep_recent", value=2)
            agent = Agent(config)
            agent.llm = FakeLLM([RuntimeError("summary provider unavailable")])
            agent._messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "important requirement"},
                {"role": "assistant", "content": "accepted"},
                {"role": "user", "content": "recent one"},
                {"role": "assistant", "content": "recent two"},
            ]

            result = asyncio.run(agent.compact_context())

            self.assertEqual(result["mode"], "fallback")
            self.assertIn("summary provider unavailable", result["fallback_reason"])
            self.assertIn("important requirement", agent.get_messages()[1]["content"])
            self.assertEqual(result["summary_tokens"], 0)

    def test_manual_compaction_skips_llm_when_nothing_is_old(self):
        with tempfile.TemporaryDirectory() as workspace:
            agent = Agent(test_config(workspace))
            fake = FakeLLM([])
            agent.llm = fake
            agent._messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "only recent"},
            ]

            result = asyncio.run(agent.compact_context())

            self.assertEqual(result["mode"], "not_needed")
            self.assertEqual(fake.requests, [])

    def test_run_automatically_summarizes_before_main_model_call(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = test_config(workspace)
            config.override("model", "context_length", value=1)
            config.override("agent", "compact_keep_recent", value=3)
            fake = FakeLLM([
                chat_response("## Working state\n- Previous work is retained."),
                chat_response("final answer"),
            ])
            agent = Agent(config)
            agent.llm = fake
            agent._messages = [{"role": "system", "content": "old system"}] + [
                {"role": "user", "content": f"old message {index}"}
                for index in range(10)
            ]

            result = asyncio.run(agent.run("current question"))

            self.assertEqual(result, "final answer")
            self.assertEqual(len(fake.requests), 2)
            self.assertIn("durable working memory", fake.requests[0]["messages"][0]["content"])
            self.assertIn(
                "Previous conversation summary",
                fake.requests[1]["messages"][1]["content"],
            )
            self.assertEqual(agent.get_status()["last_compaction"]["mode"], "llm")

    def test_configured_utility_model_handles_summary_only(self):
        with tempfile.TemporaryDirectory() as workspace:
            config = test_config(workspace)
            config.override("model", "context_length", value=1)
            config.override("agent", "compact_keep_recent", value=3)
            config.override("utility_model", value={"name": "cheap-model"})
            primary = FakeLLM([chat_response("primary answer")])
            utility = FakeLLM([
                chat_response("## Working state\n- Summarized by the cheap model."),
            ])
            utility.model = "cheap-model"
            agent = Agent(config)
            agent.llm = primary
            agent.utility_llm = utility
            agent._messages = [{"role": "system", "content": "old system"}] + [
                {"role": "user", "content": f"old message {index}"}
                for index in range(10)
            ]

            result = asyncio.run(agent.run("current question"))

            self.assertEqual(result, "primary answer")
            self.assertEqual(len(utility.requests), 1)
            self.assertEqual(len(primary.requests), 1)
            self.assertIn("durable working memory", utility.requests[0]["messages"][0]["content"])
            self.assertIn("Previous conversation summary", primary.requests[0]["messages"][1]["content"])
            self.assertEqual(agent.get_token_usage()["total_tokens"], 20)
            self.assertEqual(agent.get_token_usage()["utility_tokens"], 10)
            status = agent.get_status()
            self.assertTrue(status["utility_model_configured"])
            self.assertEqual(status["utility_model"], "cheap-model")
            self.assertEqual(status["last_compaction"]["summary_model"], "cheap-model")


if __name__ == "__main__":
    unittest.main()
