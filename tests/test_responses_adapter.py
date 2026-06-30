from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent import Agent, RunStatus
from agent.llm import LLMClient
from config import ConfigManager
from tools import registry


class CapturingResponsesClient(LLMClient):
    def __init__(self, response: dict):
        super().__init__(
            base_url="https://example.invalid/v1",
            api_key="test",
            model="fake",
            api_type="responses",
            max_retries=1,
        )
        self.response = response
        self.last_body = None

    async def _post_json(self, url: str, body: dict) -> dict:
        self.last_body = body
        return self.response


class QueueResponsesClient(CapturingResponsesClient):
    def __init__(self, responses: list[dict]):
        super().__init__(responses[0])
        self.responses = list(responses)
        self.bodies = []

    async def _post_json(self, url: str, body: dict) -> dict:
        self.bodies.append(body)
        return self.responses.pop(0)


class ResponsesAdapterTests(unittest.TestCase):
    def test_responses_tools_are_flat_and_tool_calls_are_normalized(self):
        client = CapturingResponsesClient({
            "id": "resp-1",
            "model": "fake",
            "status": "completed",
            "output": [{
                "id": "fc-1",
                "call_id": "call-1",
                "type": "function_call",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            }],
            "usage": {"total_tokens": 5},
        })
        tools = registry.get_schemas(["file"])

        normalized = asyncio.run(client.chat(
            [{"role": "user", "content": "read README"}],
            tools=tools,
        ))

        sent_tool = client.last_body["tools"][0]
        self.assertEqual(sent_tool["type"], "function")
        self.assertIn("name", sent_tool)
        self.assertNotIn("function", sent_tool)
        call = normalized["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["id"], "call-1")
        self.assertEqual(normalized["choices"][0]["finish_reason"], "tool_calls")

    def test_function_output_is_encoded_for_next_responses_request(self):
        provider_item = {
            "id": "fc-1",
            "call_id": "call-1",
            "type": "function_call",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
        converted = LLMClient._messages_to_responses_input([
            {
                "role": "assistant",
                "content": "",
                "provider_items": [provider_item],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "file body"},
        ])

        self.assertEqual(converted[0], provider_item)
        self.assertEqual(converted[1], {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "file body",
        })

    def test_completed_response_maps_to_stop(self):
        normalized = LLMClient._responses_to_chat_response({
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "finished"}],
            }],
        }, "fake")

        self.assertEqual(normalized["choices"][0]["finish_reason"], "stop")
        self.assertEqual(normalized["choices"][0]["message"]["content"], "finished")

    def test_agent_completes_full_responses_tool_loop(self):
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "sample.txt").write_text("hello", encoding="utf-8")
            client = QueueResponsesClient([
                {
                    "id": "resp-1",
                    "model": "fake",
                    "status": "completed",
                    "output": [{
                        "id": "fc-1",
                        "call_id": "call-1",
                        "type": "function_call",
                        "name": "read_file",
                        "arguments": '{"path":"sample.txt"}',
                    }],
                },
                {
                    "id": "resp-2",
                    "model": "fake",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "finished"}],
                    }],
                },
            ])
            config = ConfigManager()
            config.override("storage", "enabled", value=False)
            config.override("display", "show_tool_calls", value=False)
            config.override("agent", "refusal_review_enabled", value=False)
            config.override("security", "workspace_root", value=workspace)
            config.override("toolsets", value=["file"])
            agent = Agent(config)
            agent.llm = client

            result = asyncio.run(agent.run("read sample.txt"))

            self.assertEqual(result, "finished")
            self.assertEqual(agent.get_run_state().status, RunStatus.COMPLETED)
            second_input = client.bodies[1]["input"]
            self.assertTrue(any(item.get("type") == "function_call" for item in second_input))
            outputs = [item for item in second_input if item.get("type") == "function_call_output"]
            self.assertEqual(outputs[0]["call_id"], "call-1")
            self.assertIn("hello", outputs[0]["output"])


if __name__ == "__main__":
    unittest.main()
