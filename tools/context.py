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


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    enabled_toolsets: frozenset[str]
    allow_outside_workspace: bool = False
    allow_terminal: bool = True
    allow_network: bool = True
    approval_mode: str = ApprovalMode.AUTO.value
    session_id: str | None = None
    run_id: str | None = None
    record_file_change: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def from_config(
        cls,
        security: dict[str, Any],
        toolsets: list[str],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        record_file_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> "ToolContext":
        workspace = Path(security.get("workspace_root") or os.getcwd()).expanduser().resolve()
        return cls(
            workspace=workspace,
            enabled_toolsets=frozenset(toolsets),
            allow_outside_workspace=bool(security.get("allow_outside_workspace", False)),
            allow_terminal=bool(security.get("allow_terminal", True)),
            allow_network=bool(security.get("allow_network", True)),
            approval_mode=str(security.get("approval_mode", ApprovalMode.AUTO.value)),
            session_id=session_id,
            run_id=run_id,
            record_file_change=record_file_change,
        )

    def allows_toolset(self, toolset: str) -> bool:
        return toolset in self.enabled_toolsets

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not self.allow_outside_workspace:
            if resolved != self.workspace and self.workspace not in resolved.parents:
                raise PermissionError(f"Path outside workspace is blocked: {resolved}")
        return resolved

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
