"""Provider routing defaults, separate from wire adapters and host tool policy."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any
from urllib.parse import urlsplit


# Endpoints are explicit presets, not URL/model-name guesses. Model IDs remain
# user-selected: aggregators and local servers can expose arbitrary names.
PROVIDER_PRESETS = {
    "openai": {"base_url": "https://api.openai.com/v1", "api": "responses", "api_key_env": "OPENAI_API_KEY", "temperature": None, "chat_token_limit": "max_completion_tokens"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "api": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "temperature": None},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api": "chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api": "chat", "api_key_env": "OPENROUTER_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api": "chat", "api_key_env": "GEMINI_API_KEY"},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api": "chat", "api_key_env": "DASHSCOPE_API_KEY"},
    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "api": "chat", "api_key_env": "", "allow_unauthenticated": True},
    "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "api": "chat", "api_key_env": "", "allow_unauthenticated": True},
}

API_ALIASES = {
    "chat_completions": "chat", "openai_chat": "chat",
    "openai_responses": "responses", "anthropic_messages": "anthropic", "messages": "anthropic",
}

_ROUTING_DEFAULTS = {
    "api": "chat", "api_key": "", "api_key_env": "", "base_url": "",
    "headers": {}, "env_headers": {}, "query_params": {}, "auth_mode": "api_key",
    "allow_unauthenticated": False, "temperature": 0.7, "reasoning_effort": None,
    "disable_response_storage": False, "aliases": {}, "chat_token_limit": "max_tokens",
    "stream_usage": False,
}


def merge_model_config(base: dict, override: dict, *, isolate_endpoint: bool = False) -> dict:
    result = deepcopy(base)
    changed_provider = ("provider" in override and str(override["provider"]).lower() != str(base.get("provider", "custom")).lower())
    if changed_provider:
        result.update(deepcopy(_ROUTING_DEFAULTS))
        result.update(deepcopy(PROVIDER_PRESETS.get(str(override["provider"]).lower(), {})))
    elif isolate_endpoint and "base_url" in override and _origin(override["base_url"]) != _origin(base.get("base_url", "")):
        # An auxiliary model on another origin must opt into its own credentials.
        for key in ("api_key", "api_key_env", "headers", "env_headers", "query_params", "auth_mode"):
            result[key] = deepcopy(_ROUTING_DEFAULTS[key])
    if "api_key_env" in override and override["api_key_env"] != base.get("api_key_env") and "api_key" not in override:
        result["api_key"] = ""
    # Headers/query params are replaced, not mixed across identities.
    result.update(deepcopy(override))
    result["api"] = API_ALIASES.get(result.get("api"), result.get("api", "auto"))
    return result


def _origin(value: str) -> tuple:
    parsed = urlsplit(str(value))
    return parsed.scheme.lower(), parsed.hostname, parsed.port


def resolve_model_config(raw: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(raw)
    for key, value in PROVIDER_PRESETS.get(str(result.get("provider", "")).lower(), {}).items():
        result.setdefault(key, deepcopy(value))
    env_name = result.get("api_key_env")
    if env_name and str(env_name) in os.environ:
        result["api_key"] = os.environ[str(env_name)]
    headers = {}
    for header, value in (result.get("headers") or {}).items():
        headers = {key: item for key, item in headers.items() if key.lower() != str(header).lower()}
        headers[str(header)] = value
    for header, env in (result.get("env_headers") or {}).items():
        if str(env) not in os.environ:
            raise ValueError(f"Missing environment variable for model header {header}: {env}")
        headers = {key: item for key, item in headers.items() if key.lower() != str(header).lower()}
        headers[str(header)] = os.environ[str(env)]
    result["headers"] = headers
    result["api"] = API_ALIASES.get(result.get("api"), result.get("api", "auto"))
    return result
