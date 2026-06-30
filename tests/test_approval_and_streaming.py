from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from agent import Agent, RunStatus
from agent.llm import LLMClient
from config import ConfigManager
from tests.fakes import FakeLLM, chat_response, function_call
from tools import ApprovalDecision, RiskLevel, ToolContext, registry


def config_for(workspace: str, toolsets: list[str]) -> ConfigManager:
    config = ConfigManager()
    config.override("storage", "enabled", value=False)
    config.override("display", "show_tool_calls", value=False)
    config.override("agent", "refusal_review_enabled", value=False)
    config.override("security", "workspace_root", value=workspace)
    config.override("security", "approval_mode", value="on_request")
    config.override("toolsets", value=toolsets)
    return config


def sse(events: list[dict]) -> httpx.Response:
    content = "\n\n".join(f"data: {json.dumps(event)}" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


class ApprovalAndStreamingTests(unittest.TestCase):
    def test_registry_requires_host_approval_for_write(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(
                Path(workspace).resolve(),
                frozenset({"file"}),
                approval_mode="on_request",
            )
            blocked = asyncio.run(registry.dispatch_async(
                "write_file",
                {"path": "sample.txt", "content": "blocked"},
                context=context,
            ))
            payload = json.loads(blocked)
            self.assertTrue(payload["blocked"])
            self.assertTrue(payload["metadata"]["approval_required"])
            self.assertFalse(Path(workspace, "sample.txt").exists())

            allowed = asyncio.run(registry.dispatch_async(
                "write_file",
                {"path": "sample.txt", "content": "allowed"},
                context=context,
                approved=True,
            ))
            self.assertTrue(json.loads(allowed)["ok"])
            self.assertEqual(Path(workspace, "sample.txt").read_text(encoding="utf-8"), "allowed")

    def test_critical_command_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(
                Path(workspace).resolve(),
                frozenset({"terminal"}),
                approval_mode="on_request",
            )
            result = asyncio.run(registry.dispatch_async(
                "terminal",
                {"command": "shutdown /s"},
                context=context,
                approved=True,
            ))
            payload = json.loads(result)
            self.assertTrue(payload["blocked"])
            self.assertEqual(payload["metadata"]["risk"], "critical")

    def test_agent_approval_callback_controls_mutation(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call(
                        "call-1",
                        "write_file",
                        '{"path":"approved.txt","content":"yes"}',
                    )],
                ),
                chat_response("done"),
            ])
            requests = []

            async def approve(request):
                requests.append(request)
                return ApprovalDecision.ALLOW_ONCE

            agent = Agent(config_for(workspace, ["file"]), approval_callback=approve)
            agent.llm = fake
            result = asyncio.run(agent.run("write it"))

            self.assertEqual(result, "done")
            self.assertEqual(requests[0].risk.level, RiskLevel.MEDIUM)
            self.assertTrue(Path(workspace, "approved.txt").exists())
            self.assertEqual(agent.get_run_state().status, RunStatus.COMPLETED)

    def test_agent_denial_is_returned_to_model(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeLLM([
                chat_response(
                    finish_reason="tool_calls",
                    tool_calls=[function_call(
                        "call-1",
                        "write_file",
                        '{"path":"denied.txt","content":"no"}',
                    )],
                ),
                chat_response("not written"),
            ])

            agent = Agent(
                config_for(workspace, ["file"]),
                approval_callback=lambda request: ApprovalDecision.DENY,
            )
            agent.llm = fake
            result = asyncio.run(agent.run("write it"))

            self.assertEqual(result, "not written")
            self.assertFalse(Path(workspace, "denied.txt").exists())
            tool_payload = json.loads(fake.requests[1]["messages"][-1]["content"])
            self.assertTrue(tool_payload["blocked"])
            self.assertIn("approval denied", tool_payload["error"].lower())

    def test_chat_completions_streams_text_and_tool_arguments(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertTrue(body["stream"])
            return sse([
                {"id": "r1", "choices": [{"delta": {"content": "hel"}}]},
                {"id": "r1", "choices": [{"delta": {"content": "lo"}}]},
                {"id": "r1", "choices": [{"delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": '{"path":'},
                }]}}]},
                {"id": "r1", "choices": [{"delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '"a.txt"}'},
                }]}, "finish_reason": "tool_calls"}]},
            ])

        client = LLMClient(
            "https://example.invalid/v1",
            "test",
            "fake",
            api_type="chat",
            transport=httpx.MockTransport(handler),
        )
        deltas = []
        response = asyncio.run(client.chat(
            [{"role": "user", "content": "hi"}],
            stream_callback=deltas.append,
        ))

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(client.extract_text(response), "hello")
        call = client.extract_tool_calls(response)[0]
        self.assertEqual(call["function"]["arguments"], '{"path":"a.txt"}')

    def test_responses_api_streams_semantic_events(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return sse([
                {"type": "response.output_text.delta", "delta": "A"},
                {"type": "response.output_text.delta", "delta": "B"},
                {"type": "response.completed", "response": {
                    "id": "resp-1", "status": "completed", "usage": {"total_tokens": 2}
                }},
            ])

        client = LLMClient(
            "https://example.invalid/v1",
            "test",
            "fake",
            api_type="responses",
            transport=httpx.MockTransport(handler),
        )
        deltas = []
        response = asyncio.run(client.chat(
            [{"role": "user", "content": "hi"}],
            stream_callback=deltas.append,
        ))

        self.assertEqual(deltas, ["A", "B"])
        self.assertEqual(client.extract_text(response), "AB")
        self.assertEqual(response["usage"]["total_tokens"], 2)

    def test_terminal_subprocess_is_cancelled(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(Path(workspace).resolve(), frozenset({"terminal"}))
            if sys.platform == "win32":
                command = "for /L %i in (1,1,2147483647) do @ver > nul"
            else:
                command = f'"{sys.executable}" -c "import time; time.sleep(30)"'

            async def exercise() -> None:
                task = asyncio.create_task(registry.dispatch_async(
                    "terminal",
                    {"command": command, "timeout": 60},
                    context=context,
                ))
                await asyncio.sleep(0.2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            started = time.monotonic()
            asyncio.run(exercise())
            self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()
