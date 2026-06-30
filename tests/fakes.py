"""Offline model fakes used by Agent integration tests."""

from __future__ import annotations

from typing import Any

from agent.llm import LLMClient


class FakeLLM(LLMClient):
    def __init__(self, responses: list[dict | Exception]):
        super().__init__(base_url="https://example.invalid/v1", api_key="test", model="fake")
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, tool_choice=None):
        self.requests.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("FakeLLM response queue is empty")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chat_response(
    content: str = "",
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict] | None = None,
) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or [],
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"total_tokens": 10},
    }


def function_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
