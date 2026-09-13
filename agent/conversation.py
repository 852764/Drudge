"""Canonical conversation serialization and non-executing tool recovery."""

from __future__ import annotations

import copy
import json
from typing import Any


MESSAGE_FIELDS = ("role", "content", "tool_call_id", "tool_calls", "provider_items")


def message_from_row(row: dict[str, Any]) -> dict[str, Any]:
    message = {"role": row.get("role", "user"), "content": row.get("content")}
    if row.get("tool_call_id"):
        message["tool_call_id"] = row["tool_call_id"]
    metadata = row.get("metadata") or {}
    for key in ("tool_calls", "provider_items"):
        if metadata.get(key):
            message[key] = metadata[key]
    return message


def encode_context(messages: list[dict[str, Any]]) -> str:
    """Persist wire conversation fields, never execution/authorization objects."""
    if not isinstance(messages, list):
        raise ValueError("Context messages must be a list")
    projected = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid context message role")
        for field in ("tool_calls", "provider_items"):
            if field in message and (
                not isinstance(message[field], list)
                or any(not isinstance(item, dict) for item in message[field])
            ):
                raise ValueError(f"Context {field} must be a list of objects")
        for call in message.get("tool_calls", []):
            if not isinstance(call.get("function"), dict):
                raise ValueError("Context tool call function must be an object")
        projected.append({key: message[key] for key in MESSAGE_FIELDS if key in message})
    return json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def repair_tool_transactions(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Complete each call group before the next non-tool message.

    Missing results describe an UNKNOWN outcome, not a failed execution that is
    safe to retry. Duplicate/orphan outputs stay in the raw audit log but do not
    enter the next model request. This helper never executes tools.
    """
    rebuilt: list[dict[str, Any]] = []
    inserted: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    report = {"inserted_results": 0, "dropped_results": 0, "normalized_calls": 0}

    def flush() -> None:
        for call_id, name in pending.items():
            result = {
                "role": "tool", "tool_call_id": call_id,
                "content": json.dumps({
                    "ok": False, "content": "",
                    "error": f"Tool call interrupted before a result was recorded: {name}. "
                    "Execution outcome is unknown; inspect current state before any retry.",
                    "metadata": {"interrupted": True, "outcome_unknown": True},
                }, ensure_ascii=False),
            }
            rebuilt.append(result)
            inserted.append(result)
        pending.clear()

    for original in messages:
        message = copy.deepcopy(original)
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id in pending:
                rebuilt.append(message)
                del pending[call_id]
            else:
                report["dropped_results"] += 1
            continue
        flush()
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            if not calls:
                calls = [{
                    "id": item.get("call_id") or item.get("id"), "type": "function",
                    "function": {"name": item.get("name", "unknown"), "arguments": item.get("arguments", "{}")},
                } for item in message.get("provider_items", []) if item.get("type") == "function_call"]
                if calls:
                    message["tool_calls"] = calls
                    report["normalized_calls"] += len(calls)
            for call in calls:
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    raise ValueError("Tool transaction has an empty or duplicate call ID")
                pending[call_id] = str((call.get("function") or {}).get("name") or "unknown")
        rebuilt.append(message)
    flush()
    report["inserted_results"] = len(inserted)
    return rebuilt, inserted, report
