"""Context helpers: repo map generation and deterministic conversation compaction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".drudge",
    ".drudge-live",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}


def build_repo_map(root: str | Path, *, max_files: int = 80, max_depth: int = 3) -> str:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return f"Repository map unavailable; workspace does not exist: {base}"

    lines = [f"Repository map for: {base}"]
    count = 0
    for current, dirs, files in os.walk(base):
        current_path = Path(current)
        rel = current_path.relative_to(base)
        depth = 0 if rel == Path(".") else len(rel.parts)
        dirs[:] = sorted(d for d in dirs if d not in DEFAULT_EXCLUDE_DIRS and not d.startswith("."))
        if depth >= max_depth:
            dirs[:] = []
        if rel != Path("."):
            lines.append(f"{'  ' * depth}{current_path.name}/")
        for name in sorted(files):
            if name.endswith((".pyc", ".db", ".db-shm", ".db-wal")):
                continue
            count += 1
            if count > max_files:
                lines.append(f"... truncated after {max_files} files")
                return "\n".join(lines)
            lines.append(f"{'  ' * (depth + 1)}{name}")
    return "\n".join(lines)


def compact_messages(messages: list[dict[str, Any]], *, keep_recent: int = 8) -> list[dict[str, Any]]:
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    later_messages = [msg for msg in messages if msg.get("role") != "system"]
    if len(later_messages) <= keep_recent:
        return list(messages)

    old_messages = later_messages[:-keep_recent]
    recent_messages = later_messages[-keep_recent:]
    summary = summarize_messages(old_messages)
    summary_msg = {
        "role": "user",
        "content": "[Previous conversation summary]\n"
        f"{summary}\n"
        "[End summary]\n"
        "Use this summary as context, but prefer current user instructions and recent tool results.",
    }
    return system_messages[:1] + [summary_msg] + recent_messages


def summarize_messages(messages: list[dict[str, Any]], *, max_items: int = 20) -> str:
    items: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content") or "").replace("\n", " ").strip()
        if role == "user" and content:
            items.append(f"User: {_clip(content, 240)}")
        elif role == "assistant" and content:
            tool_count = len(msg.get("tool_calls") or [])
            suffix = f"; requested {tool_count} tool call(s)" if tool_count else ""
            items.append(f"Assistant: {_clip(content, 240)}{suffix}")
        elif role == "tool":
            if "error" in content.lower():
                items.append(f"Tool error: {_clip(content, 200)}")
            else:
                items.append(f"Tool result: {_clip(content, 200)}")
        if len(items) >= max_items:
            break
    return "\n".join(items) or "No prior messages retained."


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
