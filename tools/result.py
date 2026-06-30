"""Standard tool result envelope used between tools and the agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
        }
        if self.blocked:
            payload["blocked"] = True
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def success(cls, content: str = "", **metadata: Any) -> "ToolResult":
        return cls(ok=True, content=content, metadata=metadata)

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        blocked: bool = False,
        content: str = "",
        **metadata: Any,
    ) -> "ToolResult":
        return cls(ok=False, content=content, error=error, metadata=metadata, blocked=blocked)


def normalize_tool_payload(value: Any) -> dict[str, Any]:
    """Return a stable ok/content/error/metadata envelope while preserving legacy keys."""
    if isinstance(value, ToolResult):
        return value.to_dict()

    original: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ToolResult.success(value).to_dict()
        original = parsed

    if isinstance(original, dict):
        if {"ok", "content", "error", "metadata"}.issubset(original):
            payload = dict(original)
            payload.setdefault("metadata", {})
            return payload

        error = original.get("error")
        blocked = bool(original.get("blocked", False))
        content_value = original.get("content")
        if content_value is None:
            content = "" if error else json.dumps(original, ensure_ascii=False)
        elif isinstance(content_value, str):
            content = content_value
        else:
            content = json.dumps(content_value, ensure_ascii=False)

        explicit_metadata = original.get("metadata")
        metadata = dict(explicit_metadata) if isinstance(explicit_metadata, dict) else {}
        metadata.update({
            key: item
            for key, item in original.items()
            if key not in {"ok", "content", "error", "metadata", "blocked"}
        })
        payload = ToolResult(
            ok=error is None and not blocked,
            content=content,
            error=str(error) if error is not None else None,
            metadata=metadata,
            blocked=blocked,
        ).to_dict()
        payload.update(original)
        payload["ok"] = error is None and not blocked
        payload["content"] = content
        payload["error"] = str(error) if error is not None else None
        payload["metadata"] = metadata
        if blocked:
            payload["blocked"] = True
        return payload

    return ToolResult.success(str(original)).to_dict()


def normalize_tool_result(value: Any) -> str:
    return json.dumps(normalize_tool_payload(value), ensure_ascii=False)
