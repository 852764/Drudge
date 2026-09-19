"""Typed provider failures eligible for host-controlled context recovery."""

from __future__ import annotations

import httpx


class ContextWindowExceeded(RuntimeError):
    """The provider rejected input before generating any response output."""

    def __init__(self, *, status_code: int):
        super().__init__(f"Provider HTTP {status_code}: context_length_exceeded")
        self.status_code = status_code


def context_window_error(response: httpx.Response) -> ContextWindowExceeded | None:
    """Classify structured HTTP rejection, never arbitrary exception text.

    Bare 413 can mean a proxy upload limit, and 429 'too many tokens' can mean
    rate limiting. Neither is evidence of a model context-window overflow.
    Stream error events are deliberately excluded: output may already exist.
    """
    if response.status_code not in (400, 413):
        return None
    try:
        payload = response.json()
    except (ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error", payload)
    if not isinstance(error, dict):
        return None
    code = str(error.get("code") or "").lower()
    message = str(error.get("message") or "").lower()
    if code in {"context_length_exceeded", "context_window_exceeded"} or any(
        marker in message for marker in (
            "context_length_exceeded", "context window exceeded",
            "maximum context length", "context limit exceeded",
            "prompt is too long", "input is too long",
        )
    ):
        return ContextWindowExceeded(status_code=response.status_code)
    return None
