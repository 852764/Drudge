from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.codex_client import CodexOAuthClient
from agent.llm import create_client
from config import ConfigManager


class ConfigTests(unittest.TestCase):
    def test_safe_dict_redacts_secrets_without_mutating_config(self):
        config = ConfigManager()
        config.override("model", "api_key", value="secret-key")
        config.override("nested", value={"token": "abc", "value": 3})

        safe = config.as_safe_dict()

        self.assertEqual(safe["model"]["api_key"], "***REDACTED***")
        self.assertEqual(safe["nested"]["token"], "***REDACTED***")
        self.assertEqual(config.get("model", "api_key"), "secret-key")

    def test_loads_custom_codex_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.toml")
            path.write_text(
                '''
model = "relay-model"
model_provider = "relay"
model_context_window = 200000

[model_providers.relay]
name = "Relay"
base_url = "https://relay.example/v1"
env_key = "RELAY_API_KEY"
wire_api = "responses"
http_headers = { "X-Static" = "yes" }
env_http_headers = { "X-Feature" = "RELAY_FEATURE" }
query_params = { "api-version" = "2026-01-01" }
request_max_retries = 5
''',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {
                "RELAY_API_KEY": "relay-secret",
                "RELAY_FEATURE": "feature-value",
            }):
                config = ConfigManager(codex_config_path=str(path))

            model = config.get_model_config()
            self.assertEqual(model["name"], "relay-model")
            self.assertEqual(model["provider"], "relay")
            self.assertEqual(model["base_url"], "https://relay.example/v1")
            self.assertEqual(model["api_key"], "relay-secret")
            self.assertEqual(model["api"], "responses")
            self.assertEqual(model["context_length"], 200000)
            self.assertEqual(model["headers"]["X-Feature"], "feature-value")
            self.assertEqual(model["query_params"]["api-version"], "2026-01-01")
            self.assertEqual(model["max_retries"], 5)

            client = create_client(model)
            self.assertEqual(client.query_params["api-version"], "2026-01-01")
            self.assertEqual(client._request_headers()["X-Static"], "yes")

    def test_drudge_yaml_overrides_codex_config(self):
        with tempfile.TemporaryDirectory() as directory:
            codex_path = Path(directory, "config.toml")
            codex_path.write_text(
                'model = "codex-model"\nopenai_base_url = "https://api.openai.com/v1"\n',
                encoding="utf-8",
            )
            drudge_path = Path(directory, "drudge.yaml")
            drudge_path.write_text(
                'model:\n  name: drudge-model\n  temperature: 0.2\n',
                encoding="utf-8",
            )

            config = ConfigManager(str(drudge_path), str(codex_path))

            self.assertEqual(config.get("model", "name"), "drudge-model")
            self.assertEqual(config.get("model", "temperature"), 0.2)
            self.assertEqual(config.get("model", "api"), "responses")

    def test_codex_profile_and_builtin_openai_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.toml")
            path.write_text(
                '''
profile = "work"
model = "base-model"

[profiles.work]
model = "profile-model"
openai_base_url = "https://proxy.example/v1"
''',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-secret"}):
                config = ConfigManager(codex_config_path=str(path))

            self.assertEqual(config.get("model", "name"), "profile-model")
            self.assertEqual(config.get("model", "base_url"), "https://proxy.example/v1")
            self.assertEqual(config.get("model", "api_key"), "openai-secret")

    def test_safe_dict_redacts_provider_authorization_header(self):
        config = ConfigManager()
        config.override("model", "headers", value={"Authorization": "Bearer secret"})

        safe = config.as_safe_dict()

        self.assertEqual(safe["model"]["headers"]["Authorization"], "***REDACTED***")

    def test_safe_dict_redacts_custom_mcp_environment_secrets(self):
        config = ConfigManager()
        config.override("mcp_servers", value={
            "github": {
                "command": "server",
                "env": {
                    "GITHUB_TOKEN": "secret-token",
                    "SERVICE_API_KEY": "secret-key",
                    "MODE": "safe",
                },
            },
        })

        safe = config.as_safe_dict()["mcp_servers"]["github"]["env"]

        self.assertEqual(safe["GITHUB_TOKEN"], "***REDACTED***")
        self.assertEqual(safe["SERVICE_API_KEY"], "***REDACTED***")
        self.assertEqual(safe["MODE"], "safe")

    def test_codex_oauth_selects_experimental_client(self):
        config = ConfigManager()
        config.enable_codex_oauth()

        client = create_client(config.get_model_config())

        self.assertIsInstance(client, CodexOAuthClient)

    def test_utility_model_inherits_primary_provider_settings(self):
        config = ConfigManager()
        config.override("model", "name", value="primary-model")
        config.override("model", "base_url", value="https://provider.example/v1")
        config.override("model", "api_key", value="shared-secret")
        config.override("utility_model", value={
            "name": "cheap-model",
            "temperature": 0.1,
            "api_key_env": "UTILITY_TEST_API_KEY",
        })

        with patch.dict(os.environ, {"UTILITY_TEST_API_KEY": "utility-secret"}):
            utility = config.get_utility_model_config()

        self.assertTrue(config.has_utility_model())
        self.assertEqual(utility["name"], "cheap-model")
        self.assertEqual(utility["base_url"], "https://provider.example/v1")
        self.assertEqual(utility["api_key"], "utility-secret")
        self.assertEqual(utility["temperature"], 0.1)

    def test_utility_model_can_override_codex_oauth_with_another_provider(self):
        config = ConfigManager()
        config.enable_codex_oauth()
        config.override("utility_model", value={
            "provider": "openai-compatible",
            "name": "cheap-model",
            "base_url": "https://cheap.example/v1",
            "api_key": "cheap-secret",
            "api": "chat",
        })

        client = create_client(config.get_utility_model_config())

        self.assertNotIsInstance(client, CodexOAuthClient)
        self.assertEqual(client.model, "cheap-model")
        self.assertEqual(client.base_url, "https://cheap.example/v1")
        self.assertEqual(
            config.as_safe_dict()["utility_model"]["api_key"],
            "***REDACTED***",
        )


if __name__ == "__main__":
    unittest.main()
