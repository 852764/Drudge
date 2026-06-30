"""OpenAI-compatible LLM client abstractions."""

from __future__ import annotations

import asyncio
from typing import Any

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

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        api_order = [self.api_type] if self.api_type != "auto" else ["chat", "responses"]
        last_error = None
        for api_type in api_order:
            try:
                if api_type == "chat":
                    return await self._chat_completions(messages, tools, tool_choice)
                if api_type == "responses":
                    return await self._responses(messages, tools, tool_choice)
                raise RuntimeError(f"Unsupported model.api: {api_type}")
            except RuntimeError as error:
                last_error = str(error)
                if self.api_type == "auto" and api_type == "chat" and "HTTP 404" in last_error:
                    continue
                raise
        raise RuntimeError(last_error or "LLM request failed")

    async def _chat_completions(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        url = f"{self.base_url}/chat/completions"
        body = {
            "messages": messages,
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

    async def _post_json(self, url: str, body: dict) -> dict:
        headers = self._request_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
    )
