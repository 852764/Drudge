"""Local Drudge skill discovery and explicit activation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_SKILL_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path
    instructions: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return (
            f"SKILL: {self.name}\n"
            f"Description: {self.description}\n"
            f"Directory: {self.path.parent}\n"
            f"Instructions:\n{self.instructions}"
        )


class SkillManager:
    def __init__(
        self,
        workspace: str | Path,
        *,
        drudge_home: str | Path | None = None,
        max_chars: int = 32_000,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        home_value = drudge_home or os.getenv("DRUDGE_HOME") or "~/.drudge"
        self.drudge_home = Path(home_value).expanduser().resolve()
        self.max_chars = max(1, int(max_chars))

    @property
    def roots(self) -> list[Path]:
        roots = [self.drudge_home / "skills", self.workspace / ".drudge" / "skills"]
        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    def discover(self) -> dict[str, Skill]:
        discovered: dict[str, Skill] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            resolved_root = root.resolve()
            for skill_file in sorted(root.glob("*/SKILL.md")):
                resolved = skill_file.resolve()
                if resolved_root not in resolved.parents:
                    continue
                try:
                    skill = self._load_file(resolved)
                except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError):
                    continue
                discovered[skill.name] = skill
        return discovered

    def get(self, name: str) -> Skill:
        skill = self.discover().get(name)
        if skill is None:
            raise KeyError(f"Skill not found: {name}")
        return skill

    def _load_file(self, path: Path) -> Skill:
        raw = path.read_text(encoding="utf-8")[: self.max_chars]
        metadata: dict[str, Any] = {}
        instructions = raw.strip()
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) == 3:
                parsed = yaml.safe_load(parts[1]) or {}
                if not isinstance(parsed, dict):
                    raise ValueError("Skill front matter must be a mapping")
                metadata = parsed
                instructions = parts[2].strip()

        name = str(metadata.get("name") or path.parent.name).strip()
        if not _SKILL_NAME.fullmatch(name):
            raise ValueError(f"Invalid skill name: {name}")
        description = str(metadata.get("description") or "").strip()
        if not description:
            description = next(
                (line.strip("# ") for line in instructions.splitlines() if line.strip()),
                name,
            )[:240]
        if not instructions:
            raise ValueError(f"Skill has no instructions: {name}")
        return Skill(name, description, path, instructions, metadata)
