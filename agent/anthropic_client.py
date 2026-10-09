"""Native Anthropic Messages transport behind Drudge's canonical chat interface.

Only client-side function tools and text input are exposed. Provider-owned
history blocks are retained for same-model tool continuations, never displayed
as assistant text or forwarded to another model/endpoint.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json

import httpx

from .llm import LLMClient
from .model_errors import ProviderHTTPError, context_window_error


class AnthropicClient(LLMClient):
    def _request_headers(self):
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key and not any(str(key).lower() in ("x-api-key", "authorization") for key in self.default_headers):
            headers["x-api-key"] = self.api_key
        for key, value in self.default_headers.items():
            headers = {name: item for name, item in headers.items() if name.lower() != str(key).lower()}
            headers[str(key)] = str(value)
        return headers

    @staticmethod
    def _text_blocks(content):
        if content is None or content == "":
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list) and all(
            isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text")
            and isinstance(item.get("text"), str) for item in content
        ):
            return [{"type": "text", "text": item["text"]} for item in content if item["text"]]
        raise ValueError("Anthropic adapter currently accepts text input only")

    @staticmethod
    def _tool_block(call):
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError as exc:
            raise ValueError("Invalid JSON in stored tool arguments") from exc
        if not isinstance(arguments, dict) or not call.get("id") or not function.get("name"):
            raise ValueError("A tool call requires an ID, name and object arguments")
        return {"type": "tool_use", "id": call["id"], "name": function["name"], "input": arguments}

    def _messages_body(self, messages, tools, tool_choice, model):
        system, converted = [], []
        for message in messages:
            role = message.get("role")
            if role in ("system", "developer"):
                system.extend(self._text_blocks(message.get("content")))
                continue
            if role == "tool":
                if not message.get("tool_call_id"):
                    raise ValueError("Tool result has no tool_call_id")
                blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                           "content": message.get("content") or ""}]
                role = "user"
            elif role in ("user", "assistant"):
                blocks = self._text_blocks(message.get("content"))
                if role == "assistant":
                    calls = [self._tool_block(call) for call in message.get("tool_calls") or []]
                    blocks.extend(calls)
                    state = message.get("provider_state")
                    if self._state_matches(state, "anthropic", model):
                        original = state.get("content")
                        if isinstance(original, list) and all(isinstance(item, dict) for item in original):
                            if [item for item in original if item.get("type") == "tool_use"] == calls:
                                blocks = deepcopy(original)
            else:
                raise ValueError(f"Unsupported conversation role for Anthropic: {role}")
            if blocks:
                if converted and converted[-1]["role"] == role:
                    converted[-1]["content"].extend(blocks)
                else:
                    converted.append({"role": role, "content": blocks})
        if not converted or converted[0]["role"] != "user":
            raise ValueError("Anthropic conversation must start with a user message")
        body = {"model": model, "messages": converted, "max_tokens": self.max_tokens}
        if system:
            body["system"] = system
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools:
            definitions = []
            for tool in tools:
                if tool.get("type") != "function":
                    raise ValueError("Anthropic adapter accepts function tools only")
                function = tool["function"]
                definitions.append({"name": function["name"], "description": function.get("description", ""),
                                    "input_schema": deepcopy(function.get("parameters", {"type": "object", "properties": {}}))})
            if tool_choice != "none":
                body["tools"] = definitions
                if tool_choice in (None, "auto", "required"):
                    body["tool_choice"] = {"type": "any" if tool_choice == "required" else "auto"}
                elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                    body["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}
                else:
                    raise ValueError("Unsupported Anthropic tool_choice")
        return body

    def _convert_response(self, data, model):
        if not isinstance(data, dict) or not isinstance(data.get("content"), list) or not data.get("stop_reason"):
            raise RuntimeError("Invalid or incomplete Anthropic message")
        text, calls = [], []
        blocks = data["content"]
        for block in blocks:
            if not isinstance(block, dict):
                raise RuntimeError("Invalid Anthropic content block")
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise RuntimeError("Invalid Anthropic text block")
                text.append(block["text"])
            elif kind == "tool_use":
                if not block.get("id") or not block.get("name") or not isinstance(block.get("input"), dict):
                    raise RuntimeError("Invalid Anthropic tool_use block")
                calls.append({"id": block["id"], "type": "function", "function": {
                    "name": block["name"], "arguments": json.dumps(block["input"], ensure_ascii=False),
                }})
            elif kind not in ("thinking", "redacted_thinking"):
                raise RuntimeError(f"Unsupported Anthropic content block: {kind}")
        if calls and data["stop_reason"] != "tool_use":
            raise RuntimeError("Incomplete Anthropic tool turn; tools were not returned for execution")
        if data["stop_reason"] == "tool_use" and not calls:
            raise RuntimeError("Anthropic tool turn has no tool_use blocks")
        usage = data.get("usage") or {}
        prompt_tokens = sum(int(usage.get(key) or 0) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        output_tokens = int(usage.get("output_tokens") or 0)
        finish = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(data["stop_reason"], data["stop_reason"])
        return {
            "id": data.get("id", ""), "model": data.get("model", model),
            "choices": [{"message": {"role": "assistant", "content": "".join(text), "tool_calls": calls}, "finish_reason": finish}],
            "usage": {**usage, "prompt_tokens": prompt_tokens, "completion_tokens": output_tokens, "total_tokens": prompt_tokens + output_tokens},
            "provider_state": {"api": "anthropic", "base_url": self.base_url, "model": model, "content": deepcopy(blocks)},
        }

    async def chat(self, messages, tools=None, tool_choice=None, stream_callback=None, cancel_event=None):
        candidates = self._candidate_models()
        for model_index, model in enumerate(candidates):
            body = self._messages_body(messages, tools, tool_choice, model)
            for attempt in range(self.max_retries):
                self._raise_if_cancelled(cancel_event)
                progress = [False]
                try:
                    data = (await self._stream_message(body, stream_callback, cancel_event, progress)
                            if stream_callback else await self._post_json(f"{self.base_url}/messages", body))
                    return self._convert_response(data, model)
                except httpx.HTTPStatusError as exc:
                    if progress[0]:
                        raise RuntimeError("Anthropic stream interrupted after partial output; request was not replayed") from exc
                    overflow = context_window_error(exc.response)
                    if overflow:
                        raise overflow from exc
                    if self._should_retry_status(exc.response.status_code, attempt):
                        await asyncio.sleep(self._retry_delay(exc.response, attempt))
                        continue
                    if exc.response.status_code == 404 and model_index + 1 < len(candidates):
                        break
                    raise ProviderHTTPError(self._format_http_error(exc, model, "messages"), status_code=exc.response.status_code) from exc
                except httpx.TransportError as exc:
                    if progress[0]:
                        raise RuntimeError("Anthropic stream interrupted after partial output; request was not replayed") from exc
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 60))
                        continue
                    raise RuntimeError(f"Anthropic transport failed: {type(exc).__name__}") from exc
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise RuntimeError("Invalid JSON from Anthropic Messages") from exc
        raise RuntimeError("Anthropic request failed")

    async def _stream_message(self, body, callback, cancel_event, progress):
        data, blocks, arguments = {}, {}, {}
        open_blocks = set()
        saw_start = False
        headers = self._request_headers()
        headers["Accept"] = "text/event-stream"
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            async with client.stream("POST", f"{self.base_url}/messages", headers=headers,
                                     params=self.query_params or None, json={**body, "stream": True}) as response:
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()
                async for event in self._sse_events(response):
                    self._raise_if_cancelled(cancel_event)
                    if not isinstance(event, dict):
                        raise RuntimeError("Invalid Anthropic stream event")
                    kind = event.get("type")
                    if kind == "error":
                        # Includes overload/auth errors delivered with HTTP 200.
                        error = event.get("error") or {}
                        raise RuntimeError("Anthropic stream error: " + self._redact_error(str(error.get("message") or error.get("type") or "unknown"))[:500])
                    if kind == "message_start":
                        if saw_start or not isinstance(event.get("message"), dict):
                            raise RuntimeError("Duplicate Anthropic message_start")
                        data = deepcopy(event.get("message") or {})
                        saw_start = True
                    elif kind == "content_block_start":
                        index = event["index"]
                        if not saw_start or isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in blocks:
                            raise RuntimeError("Invalid Anthropic block order")
                        block = deepcopy(event["content_block"])
                        if not isinstance(block, dict) or block.get("type") not in ("text", "tool_use", "thinking", "redacted_thinking"):
                            raise RuntimeError("Invalid Anthropic stream content block")
                        if block.get("type") == "text" and not isinstance(block.get("text"), str):
                            raise RuntimeError("Invalid Anthropic stream text block")
                        blocks[index] = block
                        open_blocks.add(index)
                        progress[0] = True
                        if block.get("type") == "text" and block.get("text"):
                            await self._emit_delta(callback, block["text"])
                    elif kind == "content_block_delta":
                        index = event["index"]
                        if index not in open_blocks:
                            raise RuntimeError("Anthropic delta has no open content block")
                        delta = event["delta"]
                        if not isinstance(delta, dict):
                            raise RuntimeError("Invalid Anthropic content delta")
                        delta_type = delta.get("type")
                        block = blocks[index]
                        expected = {
                            "text_delta": ("text", "text"),
                            "input_json_delta": ("tool_use", "partial_json"),
                            "thinking_delta": ("thinking", "thinking"),
                            "signature_delta": ("thinking", "signature"),
                        }.get(delta_type)
                        if not expected or block.get("type") != expected[0] or not isinstance(delta.get(expected[1]), str):
                            raise RuntimeError("Anthropic delta does not match its content block")
                        progress[0] = True
                        if delta_type == "text_delta":
                            block["text"] = block.get("text", "") + delta.get("text", "")
                            await self._emit_delta(callback, delta.get("text", ""))
                        elif delta_type == "input_json_delta":
                            arguments[index] = arguments.get(index, "") + delta.get("partial_json", "")
                        elif delta_type in ("thinking_delta", "signature_delta"):
                            key = "thinking" if delta_type == "thinking_delta" else "signature"
                            block[key] = block.get(key, "") + delta.get(key, "")
                    elif kind == "content_block_stop":
                        index = event["index"]
                        if index not in open_blocks:
                            raise RuntimeError("Anthropic block stop has no open block")
                        if index in arguments:
                            blocks[index]["input"] = json.loads(arguments[index])
                        open_blocks.remove(index)
                    elif kind == "message_delta":
                        if not saw_start or open_blocks or not isinstance(event.get("delta"), dict):
                            raise RuntimeError("Invalid Anthropic message delta order")
                        data.update(event.get("delta") or {})
                        data.setdefault("usage", {}).update(event.get("usage") or {})
                    elif kind == "message_stop":
                        if not saw_start or open_blocks:
                            raise RuntimeError("Incomplete Anthropic stream content")
                        data["content"] = [blocks[index] for index in sorted(blocks)]
                        return data
        raise RuntimeError("Anthropic stream ended without message_stop; request was not replayed")

    @staticmethod
    async def _sse_events(response):
        parts = []
        async for line in response.aiter_lines():
            if line == "":
                if parts:
                    yield json.loads("\n".join(parts))
                    parts = []
            elif line.startswith("data:"):
                parts.append(line[5:].lstrip(" "))
        if parts:
            yield json.loads("\n".join(parts))
