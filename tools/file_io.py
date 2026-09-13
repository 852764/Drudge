"""Byte-exact snapshots and optimistic, atomic single-file writes.

The last-moment comparison detects stale edits; it is not an OS-level lock
against arbitrary external writers. Callers must authorize paths first.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import stat
import tempfile
from pathlib import Path


class FileConflictError(RuntimeError):
    def __init__(self, path: Path, expected: str | None, actual: str | None):
        self.expected_sha256 = expected
        self.actual_sha256 = actual
        super().__init__(
            f"File changed since the snapshot: {path}. "
            "Read the current file and review the changes before retrying."
        )


def sha256(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def text_diff(path: Path, before: str | None, after: str | None, *, limit: int = 24) -> str:
    lines = list(difflib.unified_diff(
        (before or "").splitlines(), (after or "").splitlines(),
        fromfile=str(path), tofile=str(path), lineterm="",
    ))
    if len(lines) > limit:
        lines = lines[:limit] + ["... (truncated)"]
    return "\n".join(lines)


def read_bytes(path: Path) -> bytes | None:
    """Distinguish a missing file from an empty file; never mask I/O errors."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def check_expected_hash(path: Path, data: bytes | None, expected: str | None) -> None:
    if expected is None:
        return
    if expected == "missing":
        wanted = None
    elif len(expected) == 64 and all(char in "0123456789abcdefABCDEF" for char in expected):
        wanted = expected.lower()
    else:
        raise ValueError("expected_sha256 must be a 64-digit SHA-256 hash or 'missing'")
    actual = sha256(data)
    if actual != wanted:
        raise FileConflictError(path, wanted, actual)


def check_unchanged(path: Path, expected: bytes | None) -> None:
    # A parent replaced by a symlink must not redirect the final write/read.
    if path.resolve() != path:
        raise PermissionError(f"File path changed during the operation: {path}")
    actual = read_bytes(path)
    if actual != expected:
        raise FileConflictError(path, sha256(expected), sha256(actual))


def atomic_write(path: Path, data: bytes, *, expected: bytes | None) -> None:
    """Stage in the destination directory, fsync, compare, then replace.

    Preserve existing permission bits. A failed stage/replace leaves the original
    file untouched and cleans up the temporary file.
    """
    check_unchanged(path, expected)
    mode = stat.S_IMODE(path.stat().st_mode) if expected is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".drudge-edit-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        check_unchanged(path, expected)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            # Windows may inherit a read-only bit from the destination.
            os.chmod(temporary, stat.S_IREAD | stat.S_IWRITE)
            temporary.unlink()


def delete_unchanged(path: Path, *, expected: bytes | None) -> None:
    check_unchanged(path, expected)
    if expected is not None:
        path.unlink()
