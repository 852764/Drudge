from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent.anthropic_client import AnthropicClient
from agent.cli_renderer import CliRenderer
from agent.llm import LLMClient, create_client
from agent.model_errors import ProviderHTTPError
from agent.provider_probe import probe_provider
from config import ConfigManager
from main import _handle_command, _validate_runtime_config
from model_config import PROVIDER_PRESETS
from tests.fakes import chat_response
from tests.test_provider_probe import FakeProbeClient


class ProviderConfigTests(unittest.TestCase):
    def test_raw_client_config_uses_preset_and_case_insensitive_env_headers(self):
        raw = {"provider": "anthropic", "name": "fixture-model", "api_key_env": "", "api_key": "fixture-key"}
        client = create_client(raw)
        self.assertIsInstance(client, AnthropicClient)
        self.assertEqual(client.base_url, "https://api.anthropic.com/v1")
        self.assertIsNone(client.temperature)
        self.assertNotIn("base_url", raw)
        with patch.dict(os.environ, {"FIXTURE_AUTH": "Bearer new-fixture"}):
            client = create_client({**raw, "api": "chat", "headers": {"Authorization": "Bearer old-fixture"}, "env_headers": {"authorization": "FIXTURE_AUTH"}})
            headers = client._request_headers()
            self.assertEqual([key for key in headers if key.lower() == "authorization"], ["authorization"])
            self.assertEqual(headers["authorization"], "Bearer new-fixture")

    def test_yaml_provider_presets_and_explicit_overrides(self):
        for provider, preset in PROVIDER_PRESETS.items():
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "config.yaml")
                path.write_text(f"model:\n  provider: {provider}\n  name: fixture-model\n", encoding="utf-8")
                config = ConfigManager(str(path))
                model = config.get_model_config()
                self.assertEqual(model["base_url"], preset["base_url"])
                self.assertEqual(model["api"], preset["api"])
                self.assertEqual(model["api_key_env"], preset["api_key_env"])
                self.assertEqual(model["name"], "fixture-model")
                self.assertEqual(isinstance(create_client(model), AnthropicClient), provider == "anthropic")
                config.override("model", "api", value="chat")
                self.assertEqual(config.get("model", "api"), "chat")

    def test_environment_selection_and_credentials_are_resolved_at_runtime(self):
        with patch.dict(os.environ, {"DRUDGE_MODEL_PROVIDER": "anthropic", "DRUDGE_MODEL": "fixture-model", "ANTHROPIC_API_KEY": "fixture-anthropic"}):
            config = ConfigManager()
            self.assertEqual(config.get("model", "base_url"), "https://api.anthropic.com/v1")
            self.assertEqual(config.get("model", "api_key"), "fixture-anthropic")
            _validate_runtime_config(config)
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture-rotated"}):
                self.assertEqual(create_client(config.get_model_config()).api_key, "fixture-rotated")

    def test_primary_env_key_and_header_resolution_without_secret_logging(self):
        config = ConfigManager()
        config.override("model", value={"api_key_env": "FIXTURE_KEY", "api_key": "fallback", "env_headers": {"Authorization": "FIXTURE_HEADER"}})
        with patch.dict(os.environ, {"FIXTURE_KEY": "fixture-value", "FIXTURE_HEADER": "Bearer fixture-header"}):
            model = config.get_model_config()
            self.assertEqual(model["api_key"], "fixture-value")
            self.assertEqual(create_client(model)._request_headers()["Authorization"], "Bearer fixture-header")
            safe = json.dumps(config.as_safe_dict())
            self.assertNotIn("fixture-value", safe)
            self.assertNotIn("Bearer fixture-header", safe)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "FIXTURE_HEADER"):
            config.get_model_config()

    def test_provider_switch_removes_stale_endpoint_headers_query_and_oauth(self):
        config = ConfigManager()
        config.enable_codex_oauth()
        config.override("model", "headers", value={"Authorization": "Bearer stale"})
        config.override("model", "query_params", value={"key": "stale"})
        config.override("model", "api_key", value="stale")
        config.override("model", "provider", value="ollama")
        model = config.get_model_config()
        self.assertEqual(model["api"], "chat")
        self.assertEqual(model["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(model["api_key"], "")
        self.assertEqual(model["headers"], {})
        self.assertEqual(model["query_params"], {})
        self.assertNotEqual(model["auth_mode"], "codex_oauth")
        _validate_runtime_config(config)

    def test_utility_provider_and_origin_switch_isolates_auth(self):
        for override in ({"provider": "ollama"}, {"base_url": "https://other.invalid/v1"}):
            with self.subTest(override=override):
                config = ConfigManager()
                config.override("model", value={"api_key": "primary", "api_key_env": "PRIMARY_FIXTURE_KEY", "headers": {"Authorization": "primary-header"}, "query_params": {"key": "primary-query"}})
                config.override("utility_model", value={"name": "utility", **override})
                with patch.dict(os.environ, {"PRIMARY_FIXTURE_KEY": "primary-from-env"}):
                    utility = config.get_utility_model_config()
                self.assertEqual(utility["api_key"], "")
                self.assertEqual(utility["api_key_env"], "")
                self.assertEqual(utility["headers"], {})
                self.assertEqual(utility["query_params"], {})
                self.assertEqual(config.get("model", "name"), "gpt-5.5")

    def test_missing_new_utility_key_does_not_fall_back_to_primary(self):
        config = ConfigManager()
        config.override("model", "api_key", value="primary-secret")
        config.override("utility_model", value={"name": "utility", "api_key_env": "MISSING_FIXTURE_KEY"})
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_utility_model_config()["api_key"], "")

    def test_custom_codex_provider_and_explicit_api_alias_stay_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "fixture.toml")
            path.write_text('model="fixture"\nmodel_provider="CodeGo"\nmodel_reasoning_effort="xhigh"\n[model_providers.CodeGo]\nbase_url="http://127.0.0.1:8318/v1"\nenv_key="FIXTURE_CPA_KEY"\nwire_api="responses"\n', encoding="utf-8")
            with patch.dict(os.environ, {"FIXTURE_CPA_KEY": "fixture-key"}):
                config = ConfigManager(codex_config_path=str(path))
            model = config.get_model_config()
            self.assertEqual(model["provider"], "CodeGo")
            self.assertEqual(model["api"], "responses")
            self.assertEqual(model["reasoning_effort"], "xhigh")
            self.assertEqual(create_client(model).api_key, "fixture-key")
            config.override("model", "api", value="anthropic_messages")
            self.assertIsInstance(create_client(config.get_model_config()), AnthropicClient)

    def test_providers_command_lists_presets_without_network_or_secrets(self):
        output = io.StringIO()
        config = ConfigManager()
        config.override("model", "api_key", value="not-for-display")
        with patch("main.create_client") as factory:
            asyncio.run(_handle_command("/providers", config, renderer=CliRenderer(stream=output, pretty=False)))
        factory.assert_not_called()
        for name in PROVIDER_PRESETS:
            self.assertIn(name, output.getvalue())
        self.assertNotIn("not-for-display", output.getvalue())

    def test_native_probe_uses_only_native_routes_and_respects_null_temperature(self):
        seen = []
        def factory(config):
            seen.append(config)
            return FakeProbeClient(config)
        report = asyncio.run(probe_provider({"name": "fixture", "api": "anthropic", "base_url": "https://fixture.invalid/v1", "temperature": None}, client_factory=factory))
        self.assertEqual(set(report.capabilities), {"anthropic.basic", "anthropic.tools", "anthropic.streaming"})
        self.assertTrue(all(item["api"] == "anthropic" and item["temperature"] is None for item in seen))


class CompatibleTransportTests(unittest.TestCase):
    def test_generation_parameters_match_in_streaming_and_json_requests(self):
        for api in ("chat", "responses"):
            for streaming in (False, True):
                with self.subTest(api=api, streaming=streaming):
                    captured = []
                    def handle(request):
                        captured.append(json.loads(request.content))
                        if api == "chat":
                            if streaming:
                                return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                            return httpx.Response(200, json=chat_response("ok"))
                        data = {"status": "completed", "output": [], "output_text": "ok"}
                        return httpx.Response(200, text="data: " + json.dumps({"type": "response.completed", "response": data}) + "\n\n") if streaming else httpx.Response(200, json=data)
                    client = LLMClient("https://fixture.invalid/v1/" + ("chat/completions" if api == "chat" else "responses"), "fixture", "model", api_type=api, temperature=None, chat_token_limit="max_completion_tokens", reasoning_effort="high", stream_usage=True, transport=httpx.MockTransport(handle))
                    asyncio.run(client.chat([{"role": "user", "content": "test"}], stream_callback=(lambda text: None) if streaming else None))
                    body = captured[0]
                    self.assertNotIn("temperature", body)
                    self.assertNotIn("max_tokens", body)
                    self.assertEqual(body["max_completion_tokens" if api == "chat" else "max_output_tokens"], 4096)
                    if api == "chat":
                        self.assertEqual(body["reasoning_effort"], "high")
                        self.assertEqual("stream_options" in body, streaming)
                    else:
                        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_actual_404_falls_back_but_auth_and_validation_do_not(self):
        for status in (400, 401, 403, 404):
            for streaming in (False, True):
                with self.subTest(status=status, streaming=streaming):
                    paths = []
                    def handle(request):
                        paths.append(request.url.path)
                        if request.url.path.endswith("chat/completions"):
                            return httpx.Response(status, json={"error": {"message": "fixture HTTP 404 hint"}})
                        data = {"status": "completed", "output": []}
                        return httpx.Response(200, text="data: " + json.dumps({"type": "response.completed", "response": data}) + "\n\n") if streaming else httpx.Response(200, json=data)
                    client = LLMClient("https://fixture.invalid/v1", "fixture", "model", api_type="auto", max_retries=1, transport=httpx.MockTransport(handle))
                    call = client.chat([{"role": "user", "content": "test"}], stream_callback=(lambda text: None) if streaming else None)
                    if status == 404:
                        asyncio.run(call)
                        self.assertEqual(paths, ["/v1/chat/completions", "/v1/responses"])
                    else:
                        with self.assertRaises(ProviderHTTPError):
                            asyncio.run(call)
                        self.assertEqual(paths, ["/v1/chat/completions"])

    def test_stream_error_mentioning_404_never_falls_back(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, text='data: {"error":{"message":"HTTP 404 internal stream error"}}\n\n')
        client = LLMClient("https://fixture.invalid/v1", "fixture", "model", transport=httpx.MockTransport(handle))
        with self.assertRaisesRegex(RuntimeError, "Chat stream error"):
            asyncio.run(client.chat([{"role": "user", "content": "x"}], stream_callback=lambda text: None))
        self.assertEqual(len(requests), 1)

    def test_reasoning_content_is_retained_but_scoped_and_not_displayed(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                def handle(request):
                    if streaming:
                        return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"opaque-fixture"}}]}\n\ndata: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n')
                    data = chat_response("answer")
                    data["choices"][0]["message"]["reasoning_content"] = "opaque-fixture"
                    return httpx.Response(200, json=data)
                client = LLMClient("https://fixture.invalid/v1", "fixture", "model", api_type="chat", transport=httpx.MockTransport(handle))
                deltas = []
                result = asyncio.run(client.chat([{"role": "user", "content": "x"}], stream_callback=deltas.append if streaming else None))
                self.assertNotIn("opaque-fixture", "".join(deltas))
                message = {"role": "assistant", "content": "answer", "provider_state": result["provider_state"]}
                self.assertEqual(client._chat_input([message], "model")[0]["reasoning_content"], "opaque-fixture")
                self.assertNotIn("reasoning_content", client._chat_input([message], "other")[0])
                other = LLMClient("https://other.invalid/v1", "fixture", "model")
                self.assertNotIn("reasoning_content", other._chat_input([message], "model")[0])
                self.assertNotIn("opaque-fixture", json.dumps(client._messages_to_responses_input([message])))

    def test_http_errors_redact_credentials_and_do_not_try_alias_on_401(self):
        for api in ("chat", "responses", "anthropic"):
            with self.subTest(api=api):
                calls = []
                def handle(request):
                    calls.append(request)
                    return httpx.Response(401, text="fixture-secret-key fixture-secret-header fixture-secret-query")
                cls = AnthropicClient if api == "anthropic" else LLMClient
                client = cls("https://fixture.invalid/v1", "fixture-secret-key", "model", api_type=api, model_aliases={"model": "alias"}, default_headers={"x-api-key": "fixture-secret-header"}, query_params={"token": "fixture-secret-query"}, transport=httpx.MockTransport(handle))
                with self.assertRaises(ProviderHTTPError) as raised:
                    asyncio.run(client.chat([{"role": "user", "content": "x"}]))
                self.assertEqual(len(calls), 1)
                self.assertNotIn("fixture-secret", str(raised.exception))

    def test_invalid_request_settings_fail_before_network(self):
        for setting in ({"temperature": float("nan")}, {"max_tokens": 0}, {"chat_token_limit": "unexpected"}, {"base_url": "https://user:password@fixture.invalid/v1"}, {"base_url": "https://fixture.invalid/v1?key=secret"}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                LLMClient(**{"base_url": "https://fixture.invalid/v1", "api_key": "fixture", "model": "model", **setting})
