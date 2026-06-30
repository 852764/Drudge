"""Hierarchical AGENTS.md discovery for the active workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    root = Path(workspace).expanduser().resolve()
    active = Path(cwd).expanduser().resolve() if cwd else root
    if active != root and root not in active.parents:
        active = root
    if active.is_file():
        active = active.parent

    directories = [root]
    if active != root:
        relative = active.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            directories.append(current)

    loaded: list[ProjectInstruction] = []
    remaining = max(0, int(max_chars))
    for directory in directories:
        candidate = directory / filename
        if not candidate.is_file() or remaining <= 0:
            continue
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        content = content[:remaining].strip()
        if not content:
            continue
        loaded.append(ProjectInstruction(resolved, directory, content))
        remaining -= len(content)
    return loaded


def render_project_instructions(items: list[ProjectInstruction], workspace: str | Path) -> str:
    root = Path(workspace).expanduser().resolve()
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
