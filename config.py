"""Configuration loading for Drudge."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": os.getenv("DRUDGE_MODEL", "gpt-4o-mini"),
        "base_url": os.getenv("DRUDGE_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", os.getenv("DRUDGE_API_KEY", "")),
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": 128000,
    },
    "agent": {
        "max_turns": 50,
        "compression_threshold": 0.80,
    },
    "display": {
        "show_cost": True,
        "show_tool_calls": True,
    },
    "toolsets": ["terminal", "file", "web"],
}


class ConfigManager:
    """Small dict-backed config manager used by the CLI and agent."""

    def __init__(self, config_path: str | None = None):
        self._config = deepcopy(DEFAULT_CONFIG)
        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a YAML mapping")
        self._deep_update(self._config, loaded)

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self._config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def get_model_config(self) -> dict[str, Any]:
        return self.get("model", default={})

    def get_agent_config(self) -> dict[str, Any]:
        return self.get("agent", default={})

    def get_toolsets(self) -> list[str]:
        toolsets = self.get("toolsets", default=[])
        if isinstance(toolsets, str):
            return [item.strip() for item in toolsets.split(",") if item.strip()]
        return list(toolsets or [])

    @staticmethod
    def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                ConfigManager._deep_update(target[key], value)
            else:
                target[key] = value


def get_config(config_path: str | None = None) -> ConfigManager:
    return ConfigManager(config_path)
