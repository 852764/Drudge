"""OpenAI-compatible LLM client abstractions."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable

import httpx


class LLMClient:
    """Small async client for Chat Completions with optional Responses fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model_aliases: dict[str, str] | None = None,
        api_type: str = "auto",
        default_headers: dict[str, str] | None = None,
        query_params: dict[str, Any] | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
        disable_response_storage: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_aliases = model_aliases or {}
        self.api_type = api_type
        self.default_headers = dict(default_headers or {})
        self.query_params = dict(query_params or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.disable_response_storage = disable_response_storage
        self._transport = transport

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> dict:
        api_order = [self.api_type] if self.api_type != "auto" else ["chat", "responses"]
        last_error = None
        for api_type in api_order:
            try:
                if api_type == "chat":
                    if stream_callback:
                        return await self._chat_completions_stream(
                            messages, tools, tool_choice, stream_callback, cancel_event
                        )
                    return await self._chat_completions(messages, tools, tool_choice)
                if api_type == "responses":
                    if stream_callback:
                        return await self._responses_stream(
                            messages, tools, tool_choice, stream_callback, cancel_event
                        )
                    return await self._responses(messages, tools, tool_choice)
                raise RuntimeError(f"Unsupported model.api: {api_type}")
            except RuntimeError as error:
                last_error = str(error)
                if self.api_type == "auto" and api_type == "chat" and "HTTP 404" in last_error:
                    continue
                raise
        raise RuntimeError(last_error or "LLM request failed")

    @staticmethod
    async def _emit_delta(callback: Callable[[str], Any] | None, delta: str) -> None:
        if not callback or not delta:
            return
        result = callback(delta)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError

    async def _chat_completions(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        url = f"{self.base_url}/chat/completions"
        body = {
            "messages": self._messages_to_chat_input(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        last_error = None
        for model in self._candidate_models():
            body["model"] = model
            for attempt in range(self.max_retries):
                try:
                    data = await self._post_json(url, body)
                    return {
                        "id": data.get("id", ""),
                        "model": data.get("model", model),
                        "choices": data.get("choices", []),
                        "usage": data.get("usage", {}),
                    }
                except httpx.HTTPStatusError as error:
                    last_error = self._format_http_error(error, model, endpoint="chat/completions")
                    if error.response.status_code in (429, 503):
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    if error.response.status_code == 404:
                        break
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as error:
                    last_error = str(error)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 10))
                        continue
                    break
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    async def _responses(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        url = f"{self.base_url}/responses"
        body = {
            "input": self._messages_to_responses_input(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        self._apply_responses_options(body)
        if tools:
            body["tools"] = self._tools_to_responses_tools(tools)
            body["tool_choice"] = tool_choice or "auto"

        last_error = None
        for model in self._candidate_models():
            body["model"] = model
            for attempt in range(self.max_retries):
                try:
                    data = await self._post_json(url, body)
                    return self._responses_to_chat_response(data, model)
                except httpx.HTTPStatusError as error:
                    last_error = self._format_http_error(error, model, endpoint="responses")
                    if error.response.status_code in (429, 503):
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    if error.response.status_code == 404:
                        break
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as error:
                    last_error = str(error)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 10))
                        continue
                    break
        raise RuntimeError(f"Responses request failed after {self.max_retries} attempts: {last_error}")

    async def _chat_completions_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None,
        stream_callback: Callable[[str], Any],
        cancel_event: asyncio.Event | None,
    ) -> dict:
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "messages": self._messages_to_chat_input(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        last_error = None
        for model in self._candidate_models():
            body["model"] = model
            for attempt in range(self.max_retries):
                text_parts: list[str] = []
                tool_parts: dict[int, dict[str, Any]] = {}
                usage: dict[str, Any] = {}
                finish_reason = "stop"
                saw_terminal = False
                response_id = ""
                response_model = model
                try:
                    headers = self._request_headers()
                    headers["Accept"] = "text/event-stream"
                    async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                        async with client.stream(
                            "POST",
                            url,
                            headers=headers,
                            params=self.query_params or None,
                            json=body,
                        ) as response:
                            if response.status_code >= 400:
                                await response.aread()
                                response.raise_for_status()
                            async for line in response.aiter_lines():
                                self._raise_if_cancelled(cancel_event)
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                if not raw:
                                    continue
                                if raw == "[DONE]":
                                    saw_terminal = True
                                    break
                                event = json.loads(raw)
                                response_id = str(event.get("id") or response_id)
                                response_model = str(event.get("model") or response_model)
                                if isinstance(event.get("usage"), dict):
                                    usage = event["usage"]
                                choices = event.get("choices") or []
                                if not choices:
                                    continue
                                choice = choices[0]
                                if choice.get("finish_reason"):
                                    finish_reason = str(choice["finish_reason"])
                                    saw_terminal = True
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    text_parts.append(content)
                                    await self._emit_delta(stream_callback, content)
                                for part in delta.get("tool_calls") or []:
                                    index = int(part.get("index", 0))
                                    current = tool_parts.setdefault(index, {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                    if part.get("id"):
                                        current["id"] = part["id"]
                                    function = part.get("function") or {}
                                    current["function"]["name"] += str(function.get("name") or "")
                                    current["function"]["arguments"] += str(function.get("arguments") or "")
                    if not saw_terminal:
                        raise RuntimeError("Chat stream ended without a terminal event")
                    return {
                        "id": response_id,
                        "model": response_model,
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "".join(text_parts),
                                "tool_calls": [tool_parts[index] for index in sorted(tool_parts)],
                            },
                            "finish_reason": finish_reason,
                        }],
                        "usage": usage,
                    }
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid SSE payload from chat/completions: {error}") from error
                except httpx.HTTPStatusError as error:
                    last_error = self._format_http_error(error, model, endpoint="chat/completions")
                    if error.response.status_code in (429, 503) and attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as error:
                    last_error = str(error)
                    if text_parts or tool_parts:
                        raise RuntimeError(
                            f"Chat stream interrupted after partial output: {error}"
                        ) from error
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 10))
                        continue
                    break
        raise RuntimeError(f"Streaming LLM request failed: {last_error}")

    async def _responses_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None,
        stream_callback: Callable[[str], Any],
        cancel_event: asyncio.Event | None,
    ) -> dict:
        url = f"{self.base_url}/responses"
        body: dict[str, Any] = {
            "input": self._messages_to_responses_input(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "stream": True,
        }
        self._apply_responses_options(body)
        if tools:
            body["tools"] = self._tools_to_responses_tools(tools)
            body["tool_choice"] = tool_choice or "auto"

        last_error = None
        for model in self._candidate_models():
            body["model"] = model
            for attempt in range(self.max_retries):
                output_items: list[dict] = []
                text_parts: list[str] = []
                terminal: dict[str, Any] = {}
                saw_terminal = False
                try:
                    headers = self._request_headers()
                    headers["Accept"] = "text/event-stream"
                    async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                        async with client.stream(
                            "POST",
                            url,
                            headers=headers,
                            params=self.query_params or None,
                            json=body,
                        ) as response:
                            if response.status_code >= 400:
                                await response.aread()
                                response.raise_for_status()
                            async for line in response.aiter_lines():
                                self._raise_if_cancelled(cancel_event)
                                line = line.strip()
                                if not line.startswith("data:"):
                                    continue
                                raw = line[5:].strip()
                                if not raw or raw == "[DONE]":
                                    continue
                                event = json.loads(raw)
                                event_type = str(event.get("type") or "")
                                if event_type == "response.output_text.delta":
                                    delta = str(event.get("delta") or "")
                                    if delta:
                                        text_parts.append(delta)
                                        await self._emit_delta(stream_callback, delta)
                                elif event_type == "response.output_item.done":
                                    item = event.get("item")
                                    if isinstance(item, dict):
                                        output_items.append(item)
                                elif event_type == "error":
                                    raise RuntimeError(str(event.get("message") or "Responses stream error"))
                                elif event_type in (
                                    "response.completed",
                                    "response.incomplete",
                                    "response.failed",
                                ):
                                    saw_terminal = True
                                    terminal = event.get("response") or {}
                                    if event_type == "response.failed":
                                        raise RuntimeError(f"Responses stream failed: {terminal.get('error') or event}")
                                    break
                    if not saw_terminal:
                        raise RuntimeError("Responses stream ended without a terminal event")
                    data = dict(terminal)
                    data.setdefault("model", model)
                    data.setdefault("status", "completed")
                    data["output"] = data.get("output") or output_items
                    data["output_text"] = "".join(text_parts)
                    return self._responses_to_chat_response(data, model)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid SSE payload from responses: {error}") from error
                except httpx.HTTPStatusError as error:
                    last_error = self._format_http_error(error, model, endpoint="responses")
                    if error.response.status_code in (429, 503) and attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as error:
                    last_error = str(error)
                    if text_parts or output_items:
                        raise RuntimeError(
                            f"Responses stream interrupted after partial output: {error}"
                        ) from error
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 10))
                        continue
                    break
        raise RuntimeError(f"Streaming Responses request failed: {last_error}")

    async def _post_json(self, url: str, body: dict) -> dict:
        headers = self._request_headers()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            response = await client.post(
                url,
                headers=headers,
                params=self.query_params or None,
                json=body,
            )
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        headers = self._request_headers()
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            response = await client.get(
                url,
                headers=headers,
                params=self.query_params or None,
            )
            response.raise_for_status()
            data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        return [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update({str(key): str(value) for key, value in self.default_headers.items()})
        return headers

    def _candidate_models(self) -> list[str]:
        models = [self.model]
        alias = self.model_aliases.get(self.model)
        if alias and alias not in models:
            models.append(alias)
        return models

    def _apply_responses_options(self, body: dict[str, Any]) -> None:
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.disable_response_storage:
            body["store"] = False

    @staticmethod
    def _messages_to_chat_input(messages: list[dict]) -> list[dict]:
        allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
        return [
            {key: value for key, value in message.items() if key in allowed}
            for message in messages
        ]

    @staticmethod
    def _messages_to_responses_input(messages: list[dict]) -> list[dict]:
        converted = []
        for message in messages:
            role = message.get("role", "user")
            provider_items = message.get("provider_items")
            if role == "assistant" and provider_items:
                converted.extend(provider_items)
                continue
            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": message.get("content") or "",
                })
                continue

            content = message.get("content") or ""
            if content:
                converted.append({"role": role, "content": content})
            if role == "assistant":
                for tool_call in message.get("tool_calls", []) or []:
                    function = tool_call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    })
        return converted

    @staticmethod
    def _tools_to_responses_tools(tools: list[dict]) -> list[dict]:
        converted = []
        for tool in tools:
            if tool.get("type") != "function":
                converted.append(tool)
                continue
            function = tool.get("function", {})
            converted.append({
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                "strict": bool(function.get("strict", False)),
            })
        return converted

    @staticmethod
    def _responses_to_chat_response(data: dict, model: str) -> dict:
        output_items = data.get("output", []) or []
        parts = []
        tool_calls = []
        for item in output_items:
            if item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                })
                continue
            for content in item.get("content", []) or []:
                if content.get("type") in ("output_text", "text") and content.get("text"):
                    parts.append(content["text"])
        text = data.get("output_text") or "".join(parts)
        status = data.get("status", "completed")
        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "completed":
            finish_reason = "stop"
        elif status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = status
        return {
            "id": data.get("id", ""),
            "model": data.get("model", model),
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text or "",
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }],
            "usage": data.get("usage", {}),
            "provider_items": output_items,
        }

    def _format_http_error(
        self,
        error: httpx.HTTPStatusError,
        model: str | None = None,
        endpoint: str = "chat/completions",
    ) -> str:
        status_code = error.response.status_code
        response_text = error.response.text[:500]
        active_model = model or self.model
        message = f"HTTP {status_code} on /{endpoint}: {response_text}"
        if status_code == 404 and active_model:
            message += (
                f"\nModel '{active_model}' is listed or configured, but this endpoint rejected it. "
                "Try model.api: responses, model.api: chat, or check provider routing for this model."
            )
            alias = self.model_aliases.get(active_model)
            if alias:
                message += f" Configured alias fallback: '{alias}'."
        return message

    @staticmethod
    def extract_text(response: dict) -> str | None:
        choices = response.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        return msg.get("content")

    @staticmethod
    def extract_tool_calls(response: dict) -> list[dict]:
        choices = response.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        return msg.get("tool_calls", [])

    @staticmethod
    def extract_finish_reason(response: dict) -> str:
        choices = response.get("choices", [])
        if not choices:
            return "stop"
        return choices[0].get("finish_reason", "stop")

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    total += len(str(func)) // 4
        return total


def create_client(config: dict) -> LLMClient:
    if (
        config.get("provider") == "openai-codex"
        or config.get("api") == "codex_responses"
    ):
        from .codex_client import CodexOAuthClient

        return CodexOAuthClient(
            model=config["name"],
            timeout=config.get("timeout", 300),
            max_retries=config.get("max_retries", 3),
        )
    return LLMClient(
        base_url=config["base_url"],
        api_key=config.get("api_key", ""),
        model=config["name"],
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
        model_aliases=config.get("aliases", {}),
        api_type=config.get("api", "auto"),
        default_headers=config.get("headers", {}),
        query_params=config.get("query_params", {}),
        max_retries=config.get("max_retries", 3),
        reasoning_effort=config.get("reasoning_effort"),
        disable_response_storage=bool(config.get("disable_response_storage", False)),
    )
