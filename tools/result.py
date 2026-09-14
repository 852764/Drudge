"""Standard tool result envelope used between tools and the agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


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
            payload["ok"] = bool(payload["ok"]) and payload["error"] is None and not payload.get("blocked", False)
            if not isinstance(payload["metadata"], dict):
                payload["metadata"] = {}
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
            ok=original.get("ok") is not False and error is None and not blocked,
            content=content,
            error=str(error) if error is not None else None,
            metadata=metadata,
            blocked=blocked,
        ).to_dict()
        payload.update(original)
        payload["ok"] = original.get("ok") is not False and error is None and not blocked
        payload["content"] = content
        payload["error"] = str(error) if error is not None else None
        payload["metadata"] = metadata
        if blocked:
            payload["blocked"] = True
        return payload

    return ToolResult.success(str(original)).to_dict()


def normalize_tool_result(value: Any) -> str:
    return json.dumps(normalize_tool_payload(value), ensure_ascii=False)


def limit_tool_result(
    value: Any, *, max_chars: int = 10_000,
    save_output: Callable[[str], dict[str, Any]] | None = None,
) -> str:
    """Return bounded, parseable JSON; preserve status and reference full output.

    Saving an output is bookkeeping, not tool execution. Its failure must not
    turn a successful mutation into a failed operation that invites a retry.
    """
    if max_chars < 1024:
        raise ValueError("Tool result budget must be at least 1024 characters")
    payload = normalize_tool_payload(value)
    full = json.dumps(payload, ensure_ascii=False)
    if len(full) <= max_chars:
        return full
    metadata = {
        "truncated": True,
        "original_chars": len(full),
        "output_ref": None,
    }
    if save_output is not None:
        try:
            receipt = save_output(full)
            if not isinstance(receipt, dict) or not isinstance(receipt.get("id"), str):
                raise ValueError("Output store returned an invalid receipt")
            metadata["output_ref"] = {
                key: item[:128] if isinstance(item, str) else item
                for key, item in receipt.items()
                if key in {"id", "kind", "size_bytes", "char_count", "source_chars", "sha256", "complete"}
                and isinstance(item, (str, int, bool))
            }
        except Exception as exc:
            metadata["output_warning"] = f"Output persistence failed: {type(exc).__name__}: {str(exc)[:160]}"
    else:
        metadata["output_warning"] = "Full output was not persisted; only a bounded preview is available."

    # Keep operational facts, not giant duplicated result fields. Other fields
    # remain available in the referenced original envelope.
    for key in (
        "exit_code", "timed_out", "cancelled", "interrupted", "outcome_unknown",
        "conflict", "checkpoint_created", "changed", "approval_required",
        "sha256", "before_sha256", "expected_sha256", "actual_sha256",
        "total_lines", "offset", "shown_lines", "replacements",
        "complete", "files_scanned", "bytes_read", "entries_seen",
    ):
        item = payload.get(key, payload["metadata"].get(key))
        if item is not None and isinstance(item, (str, int, float, bool)):
            metadata[key] = item[:128] if isinstance(item, str) else item
    reasons = payload.get("incomplete_reasons", payload["metadata"].get("incomplete_reasons"))
    if isinstance(reasons, list):
        metadata["incomplete_reasons"] = [reason[:80] for reason in reasons[:8] if isinstance(reason, str)]
    content = payload.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    error = payload.get("error")
    result = {
        "ok": bool(payload["ok"]), "content": "", "error": str(error)[:256] if error is not None else None,
        "metadata": metadata,
    }
    if payload.get("blocked"):
        result["blocked"] = True
    # Prioritize the receipt and outcome when using an unusually small budget.
    for key in reversed(list(metadata)):
        if len(json.dumps(result, ensure_ascii=False)) <= max_chars:
            break
        if key not in {"truncated", "original_chars", "output_ref", "exit_code", "conflict", "checkpoint_created", "complete"}:
            del metadata[key]
    if len(json.dumps(result, ensure_ascii=False)) > max_chars:
        result["error"] = "Tool reported an error; inspect full output if available." if error is not None else None
        ref = metadata.get("output_ref")
        minimal = {key: metadata[key] for key in ("truncated", "original_chars")}
        if ref:
            minimal["output_ref"] = {"id": ref["id"], "complete": ref.get("complete", False)}
        for key in ("conflict", "checkpoint_created", "timed_out", "interrupted", "outcome_unknown", "complete"):
            if isinstance(metadata.get(key), bool):
                minimal[key] = metadata[key]
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int) and -(2**63) <= exit_code < 2**63:
            minimal["exit_code"] = exit_code
        result["metadata"] = minimal
        # Receipt identifiers are opaque printable handles, not log content.
        if len(json.dumps(result, ensure_ascii=False)) > max_chars:
            minimal["output_ref"] = None
            minimal["output_warning"] = "Output receipt exceeded the result budget."
    # Escaped control characters can cost six JSON characters each: size the
    # preview by serialization, never by slicing a serialized JSON document.
    low, high = 0, len(content)
    while low < high:
        count = (low + high + 1) // 2
        head = count // 2
        tail = count - head
        result["content"] = content[:head] + "\n... [output preview truncated] ...\n" + (content[-tail:] if tail else "")
        if len(json.dumps(result, ensure_ascii=False)) <= max_chars:
            low = count
        else:
            high = count - 1
    head = low // 2
    tail = low - head
    result["content"] = content[:head] + "\n... [output preview truncated] ...\n" + (content[-tail:] if tail else "") if low else ""
    return json.dumps(result, ensure_ascii=False)
