"""Immutable execution context injected by the host, never by the model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    enabled_toolsets: frozenset[str]
    allow_outside_workspace: bool = False
    allow_terminal: bool = True
    allow_network: bool = True

    @classmethod
    def from_config(
        cls,
        security: dict[str, Any],
        toolsets: list[str],
    ) -> "ToolContext":
        workspace = Path(security.get("workspace_root") or os.getcwd()).expanduser().resolve()
        return cls(
            workspace=workspace,
            enabled_toolsets=frozenset(toolsets),
            allow_outside_workspace=bool(security.get("allow_outside_workspace", False)),
            allow_terminal=bool(security.get("allow_terminal", True)),
            allow_network=bool(security.get("allow_network", True)),
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
