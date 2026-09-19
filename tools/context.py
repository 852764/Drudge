"""Immutable execution context injected by the host, never by the model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ApprovalMode(str, Enum):
    AUTO = "auto"
    ON_REQUEST = "on_request"
    NEVER = "never"


def is_sensitive_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return path.name.lower() == "auth.json" and bool({".drudge", ".codex"} & parts)


def display_path(path: str | Path) -> Path:
    """Return an absolute user-facing path without resolving aliases/symlinks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def path_identity(path: str | Path) -> str:
    """Return the canonical identity used for security boundary comparisons."""
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def is_within_path(path: str | Path, root: str | Path) -> bool:
    """Check containment after resolving aliases without prefix collisions."""
    candidate = Path(path_identity(path))
    base = Path(path_identity(root))
    try:
        candidate.relative_to(base)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    enabled_toolsets: frozenset[str]
    allow_outside_workspace: bool = False
    allow_terminal: bool = True
    allow_network: bool = True
    approval_mode: str = ApprovalMode.ON_REQUEST.value
    session_id: str | None = None
    run_id: str | None = None
    record_file_change: Callable[[dict[str, Any]], None] | None = None
    save_tool_output: Callable[..., dict[str, Any]] | None = None
    read_tool_output: Callable[..., dict[str, Any]] | None = None
    max_output_bytes: int = 8 * 1024 * 1024
    search_max_files: int = 10_000
    search_max_file_bytes: int = 2 * 1024 * 1024
    search_max_total_bytes: int = 32 * 1024 * 1024
    search_max_entries: int = 50_000

    def __post_init__(self) -> None:
        if self.approval_mode not in tuple(mode.value for mode in ApprovalMode):
            raise ValueError("approval_mode must be auto, on_request, or never")
        for name in ("search_max_files", "search_max_file_bytes", "search_max_total_bytes", "search_max_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_config(
        cls,
        security: dict[str, Any],
        toolsets: list[str],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        record_file_change: Callable[[dict[str, Any]], None] | None = None,
        save_tool_output: Callable[..., dict[str, Any]] | None = None,
        read_tool_output: Callable[..., dict[str, Any]] | None = None,
    ) -> "ToolContext":
        workspace = display_path(security.get("workspace_root") or os.getcwd())
        return cls(
            workspace=workspace,
            enabled_toolsets=frozenset(toolsets),
            allow_outside_workspace=bool(security.get("allow_outside_workspace", False)),
            allow_terminal=bool(security.get("allow_terminal", True)),
            allow_network=bool(security.get("allow_network", True)),
            approval_mode=str(security.get("approval_mode", ApprovalMode.ON_REQUEST.value)),
            session_id=session_id,
            run_id=run_id,
            record_file_change=record_file_change,
            save_tool_output=save_tool_output,
            read_tool_output=read_tool_output,
            **{name: security[name] for name in (
                "search_max_files", "search_max_file_bytes", "search_max_total_bytes", "search_max_entries",
            ) if name in security},
        )

    def allows_toolset(self, toolset: str) -> bool:
        return toolset in self.enabled_toolsets

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        if is_sensitive_path(candidate):
            raise PermissionError("Access to credential files is blocked")
        resolved = candidate.resolve()
        if is_sensitive_path(resolved):
            raise PermissionError("Access to credential files is blocked")
        if not self.allow_outside_workspace:
            if not is_within_path(resolved, self.workspace):
                raise PermissionError(f"Path outside workspace is blocked: {resolved}")
        return resolved

    def display_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return display_path(candidate)

    def mutation_allowed(self, action: str) -> tuple[bool, str | None]:
        if self.approval_mode == ApprovalMode.NEVER.value:
            return False, f"Mutation is blocked by approval_mode=never: {action}"
        return True, None

    def network_allowed(self, action: str) -> tuple[bool, str | None]:
        if not self.allow_network:
            return False, "Network tools are disabled by config"
        if self.approval_mode == ApprovalMode.NEVER.value:
            return False, f"Network access is blocked by approval_mode=never: {action}"
        return True, None

    def terminal_allowed(self, command: str) -> tuple[bool, str | None]:
        if not self.allow_terminal:
            return False, "Terminal tool is disabled by config"
        if self.approval_mode == ApprovalMode.NEVER.value:
            return False, "Terminal commands are blocked by approval_mode=never"
        lowered = command.lower()
        dangerous_markers = [
            "rm -rf /",
            "format ",
            "mkfs.",
            "dd if=",
            "shutdown",
            "restart-computer",
            "remove-item -recurse",
            "del /s",
            "rmdir /s",
            ":(){ :|:& };:",
        ]
        for marker in dangerous_markers:
            if marker in lowered:
                return False, f"Dangerous command marker blocked: {marker}"
        return True, None
