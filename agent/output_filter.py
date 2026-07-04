"""Filter model-specific visible reasoning tags from streamed/final text."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


class DegenerateReasoningError(RuntimeError):
    """Raised when a stream is stuck producing only tagged reasoning."""


@dataclass(frozen=True, slots=True)
class FilteredText:
    text: str
    saw_tag: bool
    unclosed_tag: bool
    reasoning_chars: int


class ReasoningTagFilter:
    def __init__(self, *, max_reasoning_chars: int = 12_000, max_depth: int = 8) -> None:
        self.max_reasoning_chars = max(1, int(max_reasoning_chars))
        self.max_depth = max(1, int(max_depth))
        self.buffer = ""
        self.depth = 0
        self.saw_tag = False
        self.reasoning_chars = 0
        self.visible_parts: list[str] = []

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.buffer += chunk
        emitted: list[str] = []
        while self.buffer:
            lower = self.buffer.lower()
            if self.depth == 0:
                open_index = lower.find(OPEN_TAG)
                close_index = lower.find(CLOSE_TAG)
                indexes = [index for index in (open_index, close_index) if index >= 0]
                if indexes:
                    index = min(indexes)
                    self._emit(self.buffer[:index], emitted)
                    self.saw_tag = True
                    if index == close_index:
                        self.buffer = self.buffer[index + len(CLOSE_TAG):]
                    else:
                        self.buffer = self.buffer[index + len(OPEN_TAG):]
                        self.depth = 1
                    continue
                keep = _partial_tag_suffix(lower, (OPEN_TAG, CLOSE_TAG))
                visible = self.buffer[:-keep] if keep else self.buffer
                self._emit(visible, emitted)
                self.buffer = self.buffer[-keep:] if keep else ""
                break

            open_index = lower.find(OPEN_TAG)
            close_index = lower.find(CLOSE_TAG)
            indexes = [index for index in (open_index, close_index) if index >= 0]
            if not indexes:
                keep = _partial_tag_suffix(lower, (OPEN_TAG, CLOSE_TAG))
                hidden = self.buffer[:-keep] if keep else self.buffer
                self._hide(hidden)
                self.buffer = self.buffer[-keep:] if keep else ""
                break
            index = min(indexes)
            self._hide(self.buffer[:index])
            if index == close_index:
                self.buffer = self.buffer[index + len(CLOSE_TAG):]
                self.depth = max(0, self.depth - 1)
            else:
                self.buffer = self.buffer[index + len(OPEN_TAG):]
                self.depth += 1
                self.saw_tag = True
                if self.depth > self.max_depth:
                    raise DegenerateReasoningError(
                        f"Model emitted more than {self.max_depth} nested <think> tags"
                    )
        return "".join(emitted)

    def finish(self) -> str:
        if self.depth > 0:
            self._hide(self.buffer)
            self.buffer = ""
            return ""
        emitted: list[str] = []
        self._emit(self.buffer, emitted)
        self.buffer = ""
        return "".join(emitted)

    def _emit(self, text: str, emitted: list[str]) -> None:
        if text:
            emitted.append(text)
            self.visible_parts.append(text)

    def _hide(self, text: str) -> None:
        self.reasoning_chars += len(text)
        if self.reasoning_chars > self.max_reasoning_chars and not "".join(self.visible_parts).strip():
            raise DegenerateReasoningError(
                f"Model emitted over {self.max_reasoning_chars} characters of reasoning without an answer"
            )


def filter_reasoning_text(text: str, *, max_reasoning_chars: int = 12_000) -> FilteredText:
    parser = ReasoningTagFilter(max_reasoning_chars=max(max_reasoning_chars, len(text) + 1))
    try:
        visible = parser.feed(text) + parser.finish()
    except DegenerateReasoningError:
        visible = "".join(parser.visible_parts)
    return FilteredText(
        text=visible,
        saw_tag=parser.saw_tag,
        unclosed_tag=parser.depth > 0,
        reasoning_chars=parser.reasoning_chars,
    )


def sanitize_provider_items(items: list[dict[str, Any]], *, max_reasoning_chars: int = 12_000) -> list[dict[str, Any]]:
    """Strip tagged reasoning from visible message items, preserving encrypted reasoning items."""
    sanitized = copy.deepcopy(items)
    for item in sanitized:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in ("output_text", "text") and isinstance(content.get("text"), str):
                content["text"] = filter_reasoning_text(
                    content["text"],
                    max_reasoning_chars=max_reasoning_chars,
                ).text
    return sanitized


class MarkdownStreamFormatter:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._buffer = ""
        self._lines: list[str] = []
        self._in_code_fence = False
        self._pending_blank_after_heading = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        normalized = chunk.replace("\r\n", "\n").replace("\r", "\n")
        if not self.enabled:
            return normalized
        self._buffer += normalized
        emitted: list[str] = []
        while True:
            index = self._buffer.find("\n")
            if index < 0:
                break
            line = self._buffer[:index]
            self._buffer = self._buffer[index + 1:]
            emitted.append(self._push_line(line, terminated=True))
        return "".join(emitted)

    def finish(self) -> str:
        if not self.enabled:
            tail = self._buffer
            self._buffer = ""
            return tail
        emitted = ""
        if self._buffer:
            emitted += self._push_line(self._buffer, terminated=False)
            self._buffer = ""
        return emitted

    def _push_line(self, line: str, *, terminated: bool) -> str:
        raw_line = line if self._in_code_fence else line.rstrip(" \t")
        stripped = raw_line.strip()

        if not stripped:
            self._pending_blank_after_heading = False
            return self._append_blank_line() if terminated else ""

        is_fence = _is_fence_line(raw_line)
        is_heading = _is_heading_line(raw_line) and not self._in_code_fence
        is_list = _is_list_line(raw_line) and not self._in_code_fence
        previous = self._last_line()
        needs_blank_before = False

        if previous is not None and previous != "" and not self._in_code_fence:
            if is_heading or is_fence:
                needs_blank_before = True
            elif is_list and not _is_list_line(previous):
                needs_blank_before = True
            elif _is_fence_line(previous):
                needs_blank_before = True

        if self._pending_blank_after_heading and previous is not None and previous != "":
            needs_blank_before = True
        self._pending_blank_after_heading = False

        prefix = self._append_blank_line() if needs_blank_before else ""
        self._lines.append(raw_line)
        if is_heading and terminated:
            self._pending_blank_after_heading = True
        if is_fence:
            self._in_code_fence = not self._in_code_fence
        return prefix + raw_line + ("\n" if terminated else "")

    def _append_blank_line(self) -> str:
        if self._last_line() == "":
            return ""
        self._lines.append("")
        return "\n"

    def _last_line(self) -> str | None:
        if not self._lines:
            return None
        return self._lines[-1]


def normalize_markdown_text(text: str) -> str:
    formatter = MarkdownStreamFormatter(enabled=True)
    normalized = formatter.feed(text)
    normalized += formatter.finish()
    return normalized.rstrip()


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+\S|\d+\.\s+\S)")


def _is_heading_line(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _is_list_line(line: str) -> bool:
    return bool(_LIST_RE.match(line))


def _is_fence_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _partial_tag_suffix(value: str, tags: tuple[str, ...]) -> int:
    maximum = min(max(len(tag) for tag in tags) - 1, len(value))
    for size in range(maximum, 0, -1):
        suffix = value[-size:]
        if any(tag.startswith(suffix) for tag in tags):
            return size
    return 0
