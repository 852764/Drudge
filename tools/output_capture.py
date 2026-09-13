"""Bounded text previews and temporary disk spooling for subprocess streams."""

from __future__ import annotations

import tempfile
from typing import Any, Callable


MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_PAGE_CHARS = 1000
OUTPUT_CHUNK_CHARS = 4096
TERMINAL_PREVIEW_CHARS = 3000


def utf8_prefix(text: str, max_bytes: int) -> str:
    size = max(0, max_bytes)
    return text[:size].encode("utf-8")[:size].decode("utf-8", errors="ignore")


class OutputCapture:
    """Keep O(preview + chunk) memory while spooling at most max_bytes to disk."""

    def __init__(self, *, persist: bool, max_bytes: int = MAX_OUTPUT_BYTES, preview_chars: int = TERMINAL_PREVIEW_CHARS):
        if max_bytes < 0 or preview_chars < 2:
            raise ValueError("Invalid output capture limits")
        self.max_bytes = max_bytes
        self.preview_chars = preview_chars
        self.source_chars = 0
        self.stored_bytes = 0
        self.storage_truncated = False
        self.prefix = ""
        self.tail = ""
        self.warning: str | None = None
        self.spool = None
        if persist:
            try:
                self.spool = tempfile.TemporaryFile(mode="w+b")
            except OSError as exc:
                self.warning = f"Output spool unavailable; only a bounded preview is available: {exc}"

    def write(self, text: str) -> None:
        self.source_chars += len(text)
        self.prefix += text[:max(0, self.preview_chars - len(self.prefix))]
        self.tail = (self.tail + text)[-(self.preview_chars // 2):]
        if self.spool is None or self.storage_truncated:
            return
        data = text.encode("utf-8")
        remaining = self.max_bytes - self.stored_bytes
        if len(data) > remaining:
            data = data[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
            self.storage_truncated = True
        try:
            self.spool.write(data)
            self.stored_bytes += len(data)
        except OSError as exc:
            self.warning = f"Output spooling failed: {exc}"
            self.storage_truncated = True

    @property
    def truncated(self) -> bool:
        return self.source_chars > self.preview_chars

    def preview(self) -> str:
        if not self.truncated:
            return self.prefix
        return self.prefix[:self.preview_chars // 2] + "\n... [stream output omitted] ...\n" + self.tail

    def finish(self, save_output: Callable[..., dict[str, Any]] | None, *, tool_name: str, status: str) -> dict[str, Any] | None:
        if not self.source_chars or (not self.truncated and status == "completed"):
            return None
        if self.spool is None or save_output is None:
            self.warning = "Full stream output was not persisted; only a bounded preview is available."
            return None
        try:
            self.spool.seek(0)
            text = self.spool.read(self.max_bytes).decode("utf-8")
            return save_output(
                text, tool_name=tool_name, source_chars=self.source_chars,
                complete=not self.storage_truncated and status == "completed",
                kind="text", status=status,
            )
        except Exception as exc:
            self.warning = f"Output persistence failed: {type(exc).__name__}: {exc}"
            return None

    def close(self) -> None:
        if self.spool is not None:
            try:
                self.spool.close()
            except OSError as exc:
                self.warning = f"Output spool cleanup failed: {exc}"
