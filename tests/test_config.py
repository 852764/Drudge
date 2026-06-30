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

    def test_codex_oauth_selects_experimental_client(self):
        config = ConfigManager()
        config.enable_codex_oauth()

        client = create_client(config.get_model_config())

        self.assertIsInstance(client, CodexOAuthClient)


if __name__ == "__main__":
    unittest.main()
