"""Experimental direct client for the ChatGPT Codex Responses backend."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx

from .codex_auth import CODEX_BASE_URL, resolve_runtime_credentials
from .llm import LLMClient
from utils import format_exception


class CodexProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CodexTransportError(CodexProviderError):
    def __init__(self, message: str, *, partial_output: bool = False):
        super().__init__(message)
        self.partial_output = partial_output


class CodexOAuthClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        timeout: int = 300,
        max_retries: int = 3,
        credential_resolver: Callable[[bool], dict[str, Any]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(
            base_url=CODEX_BASE_URL,
            api_key="",
            model=model,
            api_type="codex_responses",
            timeout=timeout,
            max_retries=max(1, int(max_retries)),
        )
        self._credential_resolver = credential_resolver or resolve_runtime_credentials
        self._transport = transport

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> dict:
        credentials = await asyncio.to_thread(self._credential_resolver, False)
        refreshed = False
        failed_attempts = 0
        retryable_statuses = {429, 500, 502, 503, 504}
        while True:
            try:
                return await self._stream_response(
                    messages,
                    tools,
                    tool_choice,
                    credentials,
                    stream_callback,
                    cancel_event,
                )
            except CodexProviderError as exc:
                if exc.status_code == 401 and not refreshed:
                    credentials = await asyncio.to_thread(self._credential_resolver, True)
                    refreshed = True
                    continue
                retryable = (
                    isinstance(exc, CodexTransportError)
                    and not exc.partial_output
                ) or exc.status_code in retryable_statuses
                if not retryable:
                    raise
                failed_attempts += 1
                if failed_attempts >= self.max_retries:
                    raise
                await asyncio.sleep(min(2 ** (failed_attempts - 1), 10))

    @staticmethod
    def _instructions_and_input(messages: list[dict]) -> tuple[str, list[dict]]:
        instruction_parts = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") in ("system", "developer") and message.get("content")
        ]
        conversational = [
            message
            for message in messages
            if message.get("role") not in ("system", "developer")
        ]
        instructions = "\n\n".join(instruction_parts).strip()
        if not instructions:
            instructions = "You are a helpful coding agent."
        return instructions, LLMClient._messages_to_responses_input(conversational)

    @staticmethod
    def _headers(credentials: dict[str, Any]) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {credentials['access_token']}",
            "ChatGPT-Account-ID": str(credentials["account_id"]),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "originator": "codex_cli_rs",
            "User-Agent": "codex_cli_rs/0.0.0 (Drudge experimental)",
        }

    async def _stream_response(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | None,
        credentials: dict[str, Any],
        stream_callback: Callable[[str], Any] | None,
        cancel_event: asyncio.Event | None,
    ) -> dict:
        instructions, response_input = self._instructions_and_input(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": response_input,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            body["tools"] = self._tools_to_responses_tools(tools)
            body["tool_choice"] = tool_choice or "auto"

        base_url = str(credentials.get("base_url") or CODEX_BASE_URL).rstrip("/")
        timeout = httpx.Timeout(float(self.timeout))
        progress = {"has_output": False}
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/responses",
                    headers=self._headers(credentials),
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", errors="replace")[:1000]
                        raise CodexProviderError(
                            f"Codex backend HTTP {response.status_code}: {raw}",
                            status_code=response.status_code,
                        )
                    return await self._consume_sse(
                        response,
                        stream_callback,
                        cancel_event,
                        progress,
                    )
        except httpx.TransportError as exc:
            raise CodexTransportError(
                f"Codex transport failed: {format_exception(exc)}",
                partial_output=bool(progress["has_output"]),
            ) from exc

    async def _consume_sse(
        self,
        response: httpx.Response,
        stream_callback: Callable[[str], Any] | None = None,
        cancel_event: asyncio.Event | None = None,
        progress: dict[str, bool] | None = None,
    ) -> dict:
        output_items: list[dict] = []
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        response_id = ""
        status = "completed"
        saw_terminal = False

        async for line in response.aiter_lines():
            self._raise_if_cancelled(cancel_event)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event_type = str(event.get("type") or "")
            if event_type == "error":
                raise CodexProviderError(str(event.get("message") or "Codex stream error"))
            if event_type == "response.output_text.delta" and event.get("delta"):
                delta = str(event["delta"])
                text_parts.append(delta)
                if progress is not None:
                    progress["has_output"] = True
                await self._emit_delta(stream_callback, delta)
            elif event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict):
                    output_items.append(item)
                    if progress is not None:
                        progress["has_output"] = True
            elif event_type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                saw_terminal = True
                terminal = event.get("response") or {}
                if isinstance(terminal, dict):
                    response_id = str(terminal.get("id") or "")
                    status = str(terminal.get("status") or status)
                    if isinstance(terminal.get("usage"), dict):
                        usage = terminal["usage"]
                    if event_type == "response.failed" and terminal.get("error"):
                        raise CodexProviderError(f"Codex response failed: {terminal['error']}")
                    if event_type == "response.incomplete":
                        details = terminal.get("incomplete_details") or "no details"
                        raise CodexProviderError(f"Codex response incomplete: {details}")
                break

        if not saw_terminal:
            raise CodexTransportError(
                "Codex stream ended without a terminal response",
                partial_output=bool(output_items or text_parts),
            )
        if not output_items and text_parts:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "".join(text_parts)}],
            })
        data = {
            "id": response_id,
            "model": self.model,
            "status": status,
            "output": output_items,
            "output_text": "".join(text_parts),
            "usage": usage,
        }
        return self._responses_to_chat_response(data, self.model)
