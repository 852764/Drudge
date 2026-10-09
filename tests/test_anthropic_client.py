from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent import Agent
from agent.anthropic_client import AnthropicClient
from agent.context_manager import build_context_summary_messages
from agent.conversation import encode_context
from agent.llm import LLMClient
from agent.model_errors import ContextWindowExceeded
from tests.test_sessions_instructions_skills import configured


TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read fixture", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
CALL = {"id": "tool-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"sample.txt"}'}}


def message(content=None, stop="end_turn"):
    return {"id": "msg-fixture", "model": "fixture-model", "type": "message", "role": "assistant", "content": content if content is not None else [{"type": "text", "text": "done"}], "stop_reason": stop, "usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1}}


def stream_events(blocks=None, stop="end_turn"):
    initial = message([], stop=None)
    initial["usage"]["output_tokens"] = 0
    events = [{"type": "message_start", "message": initial}, {"type": "ping"}]
    for index, block in enumerate(blocks or [{"type": "text", "text": "done"}]):
        if block["type"] == "tool_use":
            events.append({"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}})
            encoded = json.dumps(block["input"])
            for part in (encoded[:4], encoded[4:]):
                events.append({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": part}})
        elif block["type"] == "thinking":
            events.extend([
                {"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}},
                {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": block["signature"]}},
            ])
        else:
            events.extend([
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
            ])
        events.append({"type": "content_block_stop", "index": index})
    events.extend([{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 4}}, {"type": "message_stop"}])
    return events


def sse(events):
    return "".join("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n" for event in events)


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        yield self.content.encode()
        raise httpx.ReadError("fixture connection reset")


class AnthropicTests(unittest.TestCase):
    def client(self, handler, **kwargs):
        return AnthropicClient("https://fixture.invalid/v1/messages", "fixture-key", "fixture-model", api_type="anthropic", temperature=None, transport=httpx.MockTransport(handler), **kwargs)

    def test_wrong_delta_type_does_not_expose_hidden_content(self):
        for delta in ({"type": "text_delta", "text": "hidden-fixture"}, {"type": "input_json_delta", "partial_json": "{}"}, {"type": "unexpected"}):
            with self.subTest(delta=delta):
                events = [
                    {"type": "message_start", "message": message([], stop=None)},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                ]
                client = self.client(lambda request: httpx.Response(200, text=sse(events)))
                deltas = []
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    asyncio.run(client.chat([{"role": "user", "content": "x"}], stream_callback=deltas.append))
                self.assertEqual(deltas, [])

    def test_invalid_and_truncated_native_tool_responses_are_rejected(self):
        invalid = [message(["bad"]), message([{"type": "text", "text": 123}]), message([], "tool_use"),
                   message([{"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {}}], "max_tokens")]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                asyncio.run(self.client(lambda request: httpx.Response(200, json=data)).chat([{"role": "user", "content": "x"}]))

    def test_native_headers_system_tools_results_and_token_usage(self):
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=message())
        client = self.client(handler)
        messages = [{"role": "system", "content": "system fixture"}, {"role": "developer", "content": "extra fixture"}, {"role": "user", "content": "read"}, {"role": "assistant", "content": "", "tool_calls": [CALL, {**CALL, "id": "tool-2"}]}, {"role": "tool", "tool_call_id": "tool-1", "content": "first result"}, {"role": "tool", "tool_call_id": "tool-2", "content": "second result"}]
        original = deepcopy(messages)
        result = asyncio.run(client.chat(messages, tools=[TOOL], tool_choice="required"))
        request = seen[0]
        self.assertEqual(request.url.path, "/v1/messages")
        self.assertEqual(request.headers["x-api-key"], "fixture-key")
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", request.headers)
        body = json.loads(request.content)
        self.assertEqual(body["system"], [{"type": "text", "text": "system fixture"}, {"type": "text", "text": "extra fixture"}])
        self.assertNotIn("temperature", body)
        self.assertEqual(body["tools"][0]["input_schema"], TOOL["function"]["parameters"])
        self.assertEqual(body["tool_choice"], {"type": "any"})
        self.assertEqual([item["tool_use_id"] for item in body["messages"][-1]["content"]], ["tool-1", "tool-2"])
        self.assertEqual(result["usage"]["total_tokens"], 10)
        self.assertEqual(client.extract_text(result), "done")
        self.assertEqual(messages, original)

    def test_tool_choice_named_none_and_custom_auth(self):
        client = self.client(lambda request: httpx.Response(200, json=message()), default_headers={"Authorization": "Bearer fixture-custom", "Anthropic-Version": "fixture-version"})
        self.assertNotIn("x-api-key", client._request_headers())
        self.assertEqual(client._request_headers()["Anthropic-Version"], "fixture-version")
        for choice, expected in (("auto", {"type": "auto"}), ({"type": "function", "function": {"name": "read_file"}}, {"type": "tool", "name": "read_file"})):
            body = client._messages_body([{"role": "user", "content": "x"}], [TOOL], choice, "fixture-model")
            self.assertEqual(body["tool_choice"], expected)
        body = client._messages_body([{"role": "user", "content": "x"}], [TOOL], "none", "fixture-model")
        self.assertNotIn("tools", body)

    def test_streamed_text_fragmented_tools_and_hidden_state(self):
        blocks = [{"type": "thinking", "thinking": "opaque-fixture", "signature": "fixture-signature"}, {"type": "text", "text": "reading"}, {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "sample.txt"}}]
        client = self.client(lambda request: httpx.Response(200, text=sse(stream_events(blocks, "tool_use"))))
        deltas = []
        result = asyncio.run(client.chat([{"role": "user", "content": "read"}], tools=[TOOL], stream_callback=deltas.append))
        self.assertEqual(deltas, ["reading"])
        self.assertEqual(json.loads(client.extract_tool_calls(result)[0]["function"]["arguments"]), {"path": "sample.txt"})
        self.assertEqual(result["provider_state"]["content"], blocks)
        self.assertEqual(result["usage"]["total_tokens"], 10)

    def test_native_state_does_not_cross_model_endpoint_or_protocol(self):
        blocks = [{"type": "thinking", "thinking": "opaque-fixture", "signature": "fixture-signature"}, {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "sample.txt"}}]
        client = self.client(lambda request: httpx.Response(200, json=message(blocks, "tool_use")))
        result = asyncio.run(client.chat([{"role": "user", "content": "read"}]))
        history = [{"role": "user", "content": "read"}, {"role": "assistant", "content": "", "tool_calls": [CALL], "provider_state": result["provider_state"]}, {"role": "tool", "tool_call_id": "tool-1", "content": "result"}]
        body = client._messages_body(history, None, None, "fixture-model")
        self.assertEqual(body["messages"][1]["content"], blocks)
        self.assertNotIn("opaque-fixture", json.dumps(client._messages_body(history, None, None, "other")))
        client.base_url = "https://other.invalid/v1"
        self.assertNotIn("opaque-fixture", json.dumps(client._messages_body(history, None, None, "fixture-model")))
        self.assertNotIn("opaque-fixture", json.dumps(LLMClient._messages_to_chat_input(history)))
        self.assertNotIn("opaque-fixture", json.dumps(LLMClient._messages_to_responses_input(history)))
        self.assertNotIn("opaque-fixture", json.dumps(build_context_summary_messages(history)))

    def test_retry_after_and_context_overflow(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                requests = []
                def handler(request):
                    requests.append(request)
                    if len(requests) == 1:
                        return httpx.Response(429, headers={"Retry-After": "2"})
                    return httpx.Response(200, text=sse(stream_events())) if streaming else httpx.Response(200, json=message())
                client = self.client(handler)
                with patch("agent.anthropic_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
                    result = asyncio.run(client.chat([{"role": "user", "content": "x"}], stream_callback=(lambda text: None) if streaming else None))
                self.assertEqual(client.extract_text(result), "done")
                sleep.assert_awaited_once_with(2.0)
        client = self.client(lambda request: httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "prompt is too long"}}))
        with self.assertRaises(ContextWindowExceeded):
            asyncio.run(client.chat([{"role": "user", "content": "x"}]))

    def test_partial_stream_never_retries_or_tries_alias(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, stream=BrokenStream(sse(stream_events()[:4])))
        client = self.client(handler, model_aliases={"fixture-model": "alias"})
        deltas = []
        with self.assertRaisesRegex(RuntimeError, "partial output"):
            asyncio.run(client.chat([{"role": "user", "content": "x"}], stream_callback=deltas.append))
        self.assertEqual(deltas, ["done"])
        self.assertEqual(len(requests), 1)

    def test_malformed_incomplete_and_error_streams_do_not_return_tools(self):
        for events in (stream_events()[:-1], [{"type": "error", "error": {"message": "fixture failure"}}], [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "bad order"}}]):
            with self.subTest(events=events):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(200, text=sse(events))
                with self.assertRaises(RuntimeError):
                    asyncio.run(self.client(handler).chat([{"role": "user", "content": "x"}], stream_callback=lambda text: None))
                self.assertEqual(len(calls), 1)

    def test_pre_cancelled_request_never_reaches_transport(self):
        async def exercise():
            cancel = asyncio.Event()
            cancel.set()
            calls = []
            client = self.client(lambda request: calls.append(request))
            with self.assertRaises(asyncio.CancelledError):
                await client.chat([{"role": "user", "content": "x"}], cancel_event=cancel)
            self.assertEqual(calls, [])
        asyncio.run(exercise())

    def test_multiline_sse_data_and_native_model_listing(self):
        async def parse():
            response = httpx.Response(200, text=': comment\ndata: {"type":\ndata: "ping"}\n\n')
            return [item async for item in AnthropicClient._sse_events(response)]
        self.assertEqual(asyncio.run(parse()), [{"type": "ping"}])
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [{"id": "fixture-model"}], "has_more": False})
        self.assertEqual(asyncio.run(self.client(handler).list_models()), ["fixture-model"])
        self.assertEqual(requests[0].url.path, "/v1/models")
        self.assertIn("x-api-key", requests[0].headers)

    def test_invalid_input_and_context_state_fail_closed(self):
        client = self.client(lambda request: httpx.Response(200, json=message()))
        for messages in ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "fixture"}}]}], [{"role": "system", "content": "only system"}], [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{**CALL, "function": {"name": "read_file", "arguments": "[1]"}}]}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                client._messages_body(messages, None, None, "fixture-model")
        with self.assertRaisesRegex(ValueError, "provider_state"):
            encode_context([{"role": "assistant", "content": "x", "provider_state": "bad"}])


class NativeAgentIntegrationTests(unittest.TestCase):
    def test_real_tool_loop_resume_fork_and_host_permissions(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            (root / "sample.txt").write_text("native fixture content", encoding="utf-8")
            config = configured(workspace, str(root / "sessions.db"))
            config.override("agent", "repo_map_enabled", value=False)
            config.override("security", "approval_mode", value="on_request")
            config.override("model", value={"provider": "anthropic", "name": "fixture-model"})
            requests = []
            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if len(requests) == 1:
                    return httpx.Response(200, json=message([
                        {"type": "thinking", "thinking": "opaque-fixture", "signature": "fixture-signature"},
                        {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "sample.txt"}},
                    ], "tool_use"))
                self.assertIn("native fixture content", json.dumps(body))
                self.assertIn("fixture-signature", json.dumps(body))
                return httpx.Response(200, json=message())
            agent = Agent(config)
            agent.llm = AnthropicClient("https://fixture.invalid/v1", "fixture-key", "fixture-model", api_type="anthropic", transport=httpx.MockTransport(handler))
            self.assertEqual(asyncio.run(agent.run("Read sample.txt")), "done")
            self.assertEqual(len(requests), 2)
            state = next(item["provider_state"] for item in agent.get_messages() if item.get("provider_state"))
            resumed = Agent(config)
            resumed.resume_session(agent.session_id)
            self.assertEqual(next(item["provider_state"] for item in resumed.get_messages() if item.get("provider_state")), state)
            resumed.fork_session("native branch")
            self.assertEqual(next(item["provider_state"] for item in resumed.get_messages() if item.get("provider_state")), state)
            self.assertEqual(resumed.tool_context.approval_mode, "on_request")
            self.assertFalse(resumed.tool_context.allow_outside_workspace)

    def test_native_write_tool_still_requires_host_approval(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            config = configured(workspace, str(root / "sessions.db"))
            config.override("security", "approval_mode", value="on_request")
            calls = []
            def handler(request):
                calls.append(request)
                if len(calls) == 1:
                    return httpx.Response(200, json=message([{"type": "tool_use", "id": "tool-write", "name": "write_file", "input": {"path": "blocked.txt", "content": "x"}}], "tool_use"))
                self.assertIn("blocked", request.content.decode())
                return httpx.Response(200, json=message())
            agent = Agent(config)
            agent.llm = AnthropicClient("https://fixture.invalid/v1", "fixture-key", "fixture-model", transport=httpx.MockTransport(handler))
            asyncio.run(agent.run("write a fixture"))
            self.assertFalse((root / "blocked.txt").exists())
