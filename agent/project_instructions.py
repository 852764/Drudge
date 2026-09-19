"""Hierarchical AGENTS.md discovery for the active workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.context import display_path, is_sensitive_path, is_within_path


@dataclass(frozen=True, slots=True)
class ProjectInstruction:
    path: Path
    scope: Path
    content: str


def load_project_instructions(
    workspace: str | Path,
    *,
    cwd: str | Path | None = None,
    filename: str = "AGENTS.md",
    max_chars: int = 64_000,
) -> list[ProjectInstruction]:
    """Load root-to-leaf instruction files without escaping the workspace."""
    root = display_path(workspace)
    secure_root = root.resolve()
    active = display_path(cwd) if cwd else root
    secure_active = active.resolve()
    if not is_within_path(secure_active, secure_root):
        active = root
        secure_active = secure_root
    if secure_active.is_file():
        active = active.parent
        secure_active = secure_active.parent

    directories = [root]
    if active != root:
        relative = secure_active.relative_to(secure_root)
        current = root
        for part in relative.parts:
            current = current / part
            directories.append(current)

    loaded: list[ProjectInstruction] = []
    remaining = max(0, int(max_chars))
    for directory in directories:
        candidate = directory / filename
        if is_sensitive_path(candidate):
            continue
        if not candidate.is_file() or remaining <= 0:
            continue
        resolved = candidate.resolve()
        if is_sensitive_path(resolved):
            continue
        if not is_within_path(resolved, secure_root):
            continue
        try:
            with resolved.open(encoding="utf-8") as stream:
                content = stream.read(remaining)
        except (OSError, UnicodeDecodeError):
            continue
        content = content[:remaining].strip()
        if not content:
            continue
        loaded.append(ProjectInstruction(candidate, directory, content))
        remaining -= len(content)
    return loaded


def render_project_instructions(items: list[ProjectInstruction], workspace: str | Path) -> str:
    root = display_path(workspace)
    sections = []
    for item in items:
        try:
            label = item.path.relative_to(root).as_posix()
            scope = item.scope.relative_to(root).as_posix() or "."
        except ValueError:
            label = str(item.path)
            scope = str(item.scope)
        sections.append(f"[{label} | scope: {scope}]\n{item.content}")
    return "\n\n".join(sections)
