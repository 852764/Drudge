"""Deterministic, pruned workspace discovery shared by search and repo maps.

Only workspace-local .gitignore files are used; the Git index, global excludes
and .git/info/exclude are deliberately not consulted. No external programs run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import lru_cache
from itertools import islice
import os
from pathlib import Path
import stat
from typing import Callable, Iterator

from pathspec import GitIgnoreSpec

from .context import ToolContext


DEFAULT_EXCLUDE_DIRS = frozenset({
    ".git", ".drudge", ".drudge-live", ".codex", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules", "dist", "build",
})
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".db", ".db-shm", ".db-wal", ".sqlite", ".sqlite3")
MAX_IGNORE_BYTES = 64 * 1024
MAX_IGNORE_TOTAL_BYTES = 1024 * 1024


@dataclass
class DiscoveryStats:
    entries_seen: int = 0
    ignore_files: int = 0
    ignore_bytes: int = 0
    skipped: Counter = field(default_factory=Counter)
    incomplete_reasons: set[str] = field(default_factory=set)


def matches_glob(path: Path, pattern: str | None) -> bool:
    """Slashless globs match basenames; slash globs are search-root relative."""
    if not pattern:
        return True
    pattern = pattern.replace("\\", "/")
    if "/" not in pattern:
        return fnmatchcase(path.name, pattern)
    names, patterns = path.parts, tuple(pattern.split("/"))

    @lru_cache(maxsize=None)
    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(names)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(names) and match(i + 1, j))
        return i < len(names) and fnmatchcase(names[i], patterns[j]) and match(i + 1, j + 1)

    return match(0, 0)


def is_link(path: Path) -> bool:
    info = path.lstat()
    # Junctions on Windows are reparse points too (including on Python 3.10).
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


class RepositoryWalker:
    def __init__(
        self, root: Path, *, anchor: Path | None = None,
        authorize: Callable[[str], Path] | None = None, include_hidden: bool = False,
        max_entries: int = 50_000, max_depth: int = 64,
    ):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 0 <= max_depth <= 64:
            raise ValueError("max_depth must be between 0 and 64")
        if not isinstance(include_hidden, bool):
            raise ValueError("include_hidden must be a boolean")
        self.root = root.resolve()
        self.anchor = anchor.resolve() if anchor else self.root
        if not self.root.is_relative_to(self.anchor):
            self.anchor = self.root
        self.authorize = authorize or ToolContext(self.anchor, frozenset({"file"})).resolve_path
        self.include_hidden = include_hidden
        self.max_entries = max_entries
        self.max_depth = max_depth
        self.stats = DiscoveryStats()

    def _authorized(self, path: Path) -> Path:
        resolved = self.authorize(str(path))
        # Discovery does not follow aliases, even when their destination happens
        # to be readable. Explicit file tools remain available for intentional reads.
        if resolved != path or not resolved.is_relative_to(self.anchor):
            raise PermissionError("Discovery path changed or resolves outside its scope")
        return resolved

    def _load_ignore(self, directory: Path, frames: tuple) -> tuple | None:
        path = directory / ".gitignore"
        try:
            info = path.lstat()
        except FileNotFoundError:
            return frames
        except OSError:
            self.stats.incomplete_reasons.add("ignore_unreadable")
            return None
        try:
            self._authorized(path)
            if is_link(path) or not stat.S_ISREG(info.st_mode):
                raise ValueError("Ignore files must be ordinary files")
            if info.st_size > MAX_IGNORE_BYTES or self.stats.ignore_bytes + info.st_size > MAX_IGNORE_TOTAL_BYTES:
                self.stats.incomplete_reasons.add("ignore_limit")
                return None
            with path.open("rb") as stream:
                raw = stream.read(MAX_IGNORE_BYTES + 1)
            self.stats.ignore_bytes += len(raw)
            if len(raw) > MAX_IGNORE_BYTES or self.stats.ignore_bytes > MAX_IGNORE_TOTAL_BYTES:
                self.stats.incomplete_reasons.add("ignore_limit")
                return None
            rules = raw.decode("utf-8-sig").splitlines()
            if len(rules) > 2048 or any(len(rule) > 4096 for rule in rules):
                self.stats.incomplete_reasons.add("ignore_limit")
                return None
            spec = GitIgnoreSpec.from_lines(rules)
        except Exception:
            # Do not scan a directory whose ignore policy could not be loaded.
            self.stats.incomplete_reasons.add("ignore_unreadable")
            return None
        self.stats.ignore_files += 1
        return (*frames, (directory, spec))

    def _excluded(self, path: Path, *, directory: bool, frames: tuple) -> bool:
        name = path.name.lower()
        if (directory and name in DEFAULT_EXCLUDE_DIRS) or (not directory and (
            name.endswith(EXCLUDE_SUFFIXES) or name == ".env" or
            (name.startswith(".env.") and name != ".env.example")
        )):
            self.stats.skipped["excluded"] += 1
            return True
        if not self.include_hidden and path.name.startswith("."):
            self.stats.skipped["hidden"] += 1
            return True
        ignored = False
        for base, spec in frames:
            name_in_frame = path.relative_to(base).as_posix() + ("/" if directory else "")
            decision = spec.check_file(name_in_frame).include
            if decision is not None:
                ignored = decision
        if ignored:
            self.stats.skipped["ignored"] += 1
        return ignored

    def files(self, file_glob: str | None = None) -> Iterator[Path]:
        frames: tuple = ()
        parent = self.anchor
        # Scoped searches inherit ancestor rules inside the configured workspace.
        for part in self.root.relative_to(self.anchor).parts:
            frames = self._load_ignore(parent, frames)
            if frames is None:
                return
            parent = parent / part
            if self._excluded(parent, directory=True, frames=frames):
                return
        yield from self._walk(self.root, frames, 0, file_glob)

    def _walk(self, directory: Path, frames: tuple, depth: int, file_glob: str | None) -> Iterator[Path]:
        if self.stats.entries_seen >= self.max_entries:
            self.stats.incomplete_reasons.add("entry_limit")
            return
        try:
            self._authorized(directory)
            frames = self._load_ignore(directory, frames)
            if frames is None:
                return
            remaining = self.max_entries - self.stats.entries_seen
            with os.scandir(directory) as iterator:
                entries = list(islice(iterator, remaining + 1))
        except (OSError, ValueError, RuntimeError):
            self.stats.incomplete_reasons.add("directory_unreadable")
            return
        if len(entries) > remaining:
            self.stats.incomplete_reasons.add("entry_limit")
            self.stats.entries_seen += remaining
            # Do not choose an arbitrary subset based on filesystem enumeration
            # order. Ask the caller to narrow a directory that exceeds the cap.
            return
        self.stats.entries_seen += len(entries)
        ordered = []
        for entry in entries:
            try:
                ordered.append((entry.is_dir(follow_symlinks=False), entry))
            except OSError:
                self.stats.skipped["unreadable"] += 1
                self.stats.incomplete_reasons.add("path_unreadable")
        # Root files/configuration precede directory contents in bounded maps.
        for is_dir, entry in sorted(ordered, key=lambda item: (item[0], item[1].name.casefold(), item[1].name)):
            path = directory / entry.name
            try:
                if is_link(path):
                    self.stats.skipped["links"] += 1
                    continue
                if self._excluded(path, directory=is_dir, frames=frames):
                    continue
                self._authorized(path)
                if is_dir:
                    if depth >= self.max_depth:
                        self.stats.incomplete_reasons.add("depth_limit")
                    else:
                        yield from self._walk(path, frames, depth + 1, file_glob)
                elif entry.is_file(follow_symlinks=False):
                    if matches_glob(path.relative_to(self.root), file_glob):
                        yield path
                    else:
                        self.stats.skipped["glob"] += 1
                else:
                    self.stats.skipped["special"] += 1
            except (OSError, ValueError, RuntimeError):
                self.stats.skipped["unreadable"] += 1
                self.stats.incomplete_reasons.add("path_unreadable")
