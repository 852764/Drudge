from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent import Agent, RunStatus
from agent.codex_client import CodexOAuthClient
from config import ConfigManager
from tools import registry


def sse_response(events: list[dict]) -> httpx.Response:
    body = "\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n"
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=body.encode(),
    )


def credentials(force_refresh: bool = False) -> dict:
    return {
        "access_token": "test-token",
        "account_id": "acct-test",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }


class CodexClientTests(unittest.TestCase):
    def test_streaming_text_is_normalized(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
            return sse_response([
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-1",
                        "status": "completed",
                        "output": None,
                        "usage": {"total_tokens": 7},
                    },
                },
            ])

        client = CodexOAuthClient(
            model="gpt-5.5",
            credential_resolver=credentials,
            transport=httpx.MockTransport(handler),
        )
        deltas = []
        response = asyncio.run(client.chat([
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "hello"},
        ], stream_callback=deltas.append))

        self.assertEqual(client.extract_text(response), "hello")
        self.assertEqual(deltas, ["hello"])
        self.assertEqual(client.extract_finish_reason(response), "stop")
        self.assertEqual(captured["body"]["instructions"], "system instructions")
        self.assertTrue(captured["body"]["stream"])
        self.assertFalse(captured["body"]["store"])
        self.assertEqual(captured["headers"]["chatgpt-account-id"], "acct-test")

    def test_agent_owns_tool_loop_with_codex_provider(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "sample.txt").write_text("hello from tool", encoding="utf-8")
            request_bodies = []

            def handler(request: httpx.Request) -> httpx.Response:
                request_bodies.append(json.loads(request.content))
                if len(request_bodies) == 1:
                    return sse_response([
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "id": "fc-1",
                                "call_id": "call-1",
                                "type": "function_call",
                                "name": "read_file",
                                "arguments": '{"path":"sample.txt"}',
                                "status": "completed",
                            },
                        },
                        {
                            "type": "response.completed",
                            "response": {"id": "resp-1", "status": "completed"},
                        },
                    ])
                return sse_response([
                    {"type": "response.output_text.delta", "delta": "tool result received"},
                    {
                        "type": "response.completed",
                        "response": {"id": "resp-2", "status": "completed"},
                    },
                ])

            config = ConfigManager()
            config.override("storage", "enabled", value=False)
            config.override("display", "show_tool_calls", value=False)
            config.override("agent", "refusal_review_enabled", value=False)
            config.override("security", "workspace_root", value=workspace)
            config.override("toolsets", value=["file"])
            config.enable_codex_oauth()
            agent = Agent(config)
            agent.llm = CodexOAuthClient(
                model="gpt-5.5",
                credential_resolver=credentials,
                transport=httpx.MockTransport(handler),
            )

            result = asyncio.run(agent.run("read sample.txt"))

            self.assertEqual(result, "tool result received")
            self.assertEqual(agent.get_run_state().status, RunStatus.COMPLETED)
            second_input = request_bodies[1]["input"]
            outputs = [item for item in second_input if item.get("type") == "function_call_output"]
            self.assertEqual(outputs[0]["call_id"], "call-1")
            self.assertIn("hello from tool", outputs[0]["output"])

    def test_codex_tool_schema_is_flat(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return sse_response([
                {"type": "response.output_text.delta", "delta": "done"},
                {"type": "response.completed", "response": {"status": "completed"}},
            ])

        client = CodexOAuthClient(
            model="gpt-5.5",
            credential_resolver=credentials,
            transport=httpx.MockTransport(handler),
        )
        asyncio.run(client.chat(
            [{"role": "user", "content": "test"}],
            tools=registry.get_schemas(["file"]),
        ))

        self.assertEqual(captured["body"]["tools"][0]["type"], "function")
        self.assertNotIn("function", captured["body"]["tools"][0])


if __name__ == "__main__":
    unittest.main()
