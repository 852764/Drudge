"""Configuration loading for Drudge."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": os.getenv("DRUDGE_MODEL", "gpt-5.5"),
        "base_url": os.getenv("DRUDGE_BASE_URL", "https://anyrouter.top/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", os.getenv("DRUDGE_API_KEY", "")),
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": 128000,
        "max_retries": 3,
        "api": os.getenv("DRUDGE_MODEL_API", "auto"),
        "aliases": {},
    },
    # Optional low-cost model for context summaries and other background tasks.
    # Missing fields inherit from "model"; null means reuse the primary client.
    "utility_model": None,
    "agent": {
        "max_turns": 50,
        "compression_threshold": 0.80,
        "compact_keep_recent": 8,
        "context_summary_mode": "llm",
        "context_summary_fallback": True,
        "repo_map_enabled": True,
        "repo_map_max_files": 80,
        "instructions_enabled": True,
        "instructions_filename": "AGENTS.md",
        "instructions_max_chars": 64000,
        "skill_max_chars": 32000,
        "reasoning_tag_max_chars": 12000,
        "reasoning_recovery_attempts": 1,
        "refusal_review_enabled": True,
        "refusal_review_notice": "[Drudge] 检测到模型可能拒绝了请求，正在进行安全二次处理...",
    },
    "display": {
        "show_cost": True,
        "show_tool_calls": True,
        "hide_reasoning_tags": True,
        "format_markdown_output": True,
        "pretty_cli": True,
    },
    "storage": {
        "enabled": True,
        "path": os.getenv("DRUDGE_DB_PATH", "~/.drudge/drudge.db"),
    },
    "mcp_servers": {},
    "tool_selection": {
        "enabled": True,
        "min_tools": 16,
        "min_schema_tokens": 3000,
        "max_selected": 12,
        "search_limit": 5,
        "sticky_recent": 4,
        "always_include": [],
    },
    "security": {
        "workspace_root": os.getenv("DRUDGE_WORKSPACE", os.getcwd()),
        "allow_outside_workspace": False,
        "allow_terminal": True,
        "allow_network": True,
        "approval_mode": os.getenv("DRUDGE_APPROVAL_MODE", "auto"),
    },
    "toolsets": ["terminal", "file", "web"],
}


class ConfigManager:
    """Small dict-backed config manager used by the CLI and agent."""

    def __init__(
        self,
        config_path: str | None = None,
        codex_config_path: str | None = None,
    ):
        self._config = deepcopy(DEFAULT_CONFIG)
        self.codex_config_path: Path | None = None
        if codex_config_path:
            self.load_codex(codex_config_path)
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

    @staticmethod
    def default_codex_config_path() -> Path:
        codex_home = os.getenv("CODEX_HOME")
        base = Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"
        return base / "config.toml"

    def load_codex(self, config_path: str | None = None) -> None:
        """Load the safe provider subset of a Codex config.toml file."""
        path = Path(config_path).expanduser() if config_path else self.default_codex_config_path()
        if not path.exists():
            raise FileNotFoundError(f"Codex config file not found: {path}")

        with path.open("rb") as file:
            loaded = tomllib.load(file)
        if not isinstance(loaded, dict):
            raise ValueError("Codex config file must contain a TOML table")

        effective = dict(loaded)
        profile_name = loaded.get("profile")
        if profile_name:
            profiles = loaded.get("profiles", {})
            profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
            if not isinstance(profile, dict):
                raise ValueError(f"Codex profile not found: {profile_name}")
            effective.update(profile)

        provider_id = effective.get("model_provider", "openai")
        model_update: dict[str, Any] = {
            "api": "responses",
            "provider": provider_id,
        }
        if effective.get("model"):
            model_update["name"] = effective["model"]
        if effective.get("model_context_window"):
            model_update["context_length"] = effective["model_context_window"]

        if provider_id == "openai":
            model_update["base_url"] = effective.get(
                "openai_base_url",
                "https://api.openai.com/v1",
            )
            model_update["api_key"] = os.getenv(
                "OPENAI_API_KEY",
                os.getenv("DRUDGE_API_KEY", ""),
            )
            model_update["api_key_env"] = "OPENAI_API_KEY"
            model_update["allow_unauthenticated"] = False
        else:
            providers = loaded.get("model_providers", {})
            provider = providers.get(provider_id) if isinstance(providers, dict) else None
            if not isinstance(provider, dict):
                raise ValueError(f"Codex model provider not found: {provider_id}")
            if not provider.get("base_url"):
                raise ValueError(f"Codex model provider has no base_url: {provider_id}")
            if provider.get("auth"):
                raise ValueError(
                    "Codex command-backed provider auth is not supported by Drudge; "
                    "configure env_key or env_http_headers instead"
                )

            wire_api = provider.get("wire_api", "responses")
            if wire_api != "responses":
                raise ValueError(
                    f"Unsupported Codex wire_api '{wire_api}'; expected 'responses'"
                )
            model_update["base_url"] = provider["base_url"]
            model_update["api"] = wire_api

            env_key = provider.get("env_key")
            if env_key:
                model_update["api_key"] = os.getenv(str(env_key), "")
                model_update["api_key_env"] = str(env_key)
                model_update["allow_unauthenticated"] = False
            elif provider.get("requires_openai_auth"):
                model_update["api_key"] = os.getenv("OPENAI_API_KEY", "")
                model_update["api_key_env"] = "OPENAI_API_KEY"
                model_update["allow_unauthenticated"] = False
            else:
                model_update["api_key"] = ""
                model_update["allow_unauthenticated"] = True

            headers = dict(provider.get("http_headers") or {})
            for header, env_name in (provider.get("env_http_headers") or {}).items():
                value = os.getenv(str(env_name))
                if value is None:
                    raise ValueError(
                        f"Missing environment variable for Codex header {header}: {env_name}"
                    )
                headers[str(header)] = value
            if headers:
                model_update["headers"] = headers
            if provider.get("query_params"):
                model_update["query_params"] = dict(provider["query_params"])
            if provider.get("request_max_retries") is not None:
                max_retries = int(provider["request_max_retries"])
                if not 1 <= max_retries <= 100:
                    raise ValueError("Codex request_max_retries must be between 1 and 100")
                model_update["max_retries"] = max_retries

        self._deep_update(self._config["model"], model_update)
        self.codex_config_path = path.resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self._config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def get_model_config(self) -> dict[str, Any]:
        return self.get("model", default={})

    def has_utility_model(self) -> bool:
        return isinstance(self.get("utility_model", default=None), dict) and bool(
            self.get("utility_model", default={})
        )

    def get_utility_model_config(self) -> dict[str, Any]:
        """Return utility model settings overlaid on the primary model settings."""
        merged = deepcopy(self.get_model_config())
        override = self.get("utility_model", default=None)
        if isinstance(override, dict):
            self._deep_update(merged, override)
        api_key_env = merged.get("api_key_env")
        if api_key_env:
            merged["api_key"] = os.getenv(str(api_key_env), merged.get("api_key", ""))
        return merged

    def get_agent_config(self) -> dict[str, Any]:
        return self.get("agent", default={})

    def get_storage_config(self) -> dict[str, Any]:
        return self.get("storage", default={})

    def get_security_config(self) -> dict[str, Any]:
        return self.get("security", default={})

    def get_toolsets(self) -> list[str]:
        toolsets = self.get("toolsets", default=[])
        if isinstance(toolsets, str):
            return [item.strip() for item in toolsets.split(",") if item.strip()]
        return list(toolsets or [])

    def as_safe_dict(self) -> dict[str, Any]:
        """Return a copy suitable for logs and terminal output."""
        sensitive_names = {
            "api_key",
            "password",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "client_secret",
            "authorization",
            "proxy-authorization",
            "x-api-key",
        }

        def is_sensitive_name(key: Any) -> bool:
            normalized = str(key).lower().replace("-", "_")
            return (
                normalized in {name.replace("-", "_") for name in sensitive_names}
                or normalized.endswith(("_token", "_secret", "_password", "_api_key"))
            )

        def redact(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: "***REDACTED***" if is_sensitive_name(key) and item else redact(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [redact(item) for item in value]
            return deepcopy(value)

        return redact(self._config)

    def override(self, *keys: str, value: Any) -> None:
        if not keys:
            raise ValueError("override requires at least one key")
        current = self._config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
            if not isinstance(current, dict):
                raise ValueError(f"Cannot override nested key below non-mapping: {key}")
        current[keys[-1]] = value

    def enable_codex_oauth(self) -> None:
        """Select the experimental direct ChatGPT Codex OAuth provider."""
        self._deep_update(self._config["model"], {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api": "codex_responses",
            "api_key": "",
            "auth_mode": "codex_oauth",
            "allow_unauthenticated": False,
        })

    @staticmethod
    def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                ConfigManager._deep_update(target[key], value)
            else:
                target[key] = value


def get_config(
    config_path: str | None = None,
    codex_config_path: str | None = None,
) -> ConfigManager:
    return ConfigManager(config_path, codex_config_path)
