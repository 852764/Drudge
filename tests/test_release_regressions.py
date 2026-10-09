"""Offline release-review regressions; no credentials or provider access needed."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent.llm import LLMClient
from agent.provider_probe import probe_provider
from config import ConfigManager
from tests.test_provider_probe import FakeProbeClient


class ReviewRegressions(unittest.TestCase):
    def test_reimport_same_codex_provider_clears_old_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "fixture.toml")
            config = ConfigManager()
            config.override("model", value={
                "provider": "custom", "api_key": "old-fixture-key", "api_key_env": "OLD_FIXTURE_KEY",
                "headers": {"Authorization": "Bearer old-fixture"},
                "env_headers": {"X-Token": "OLD_FIXTURE_HEADER"},
                "query_params": {"token": "old-fixture"}, "auth_mode": "codex_oauth",
            })
            source.write_text('model="fixture-model"\nmodel_provider="custom"\n[model_providers.custom]\nbase_url="https://new.invalid/v1"\nwire_api="responses"\n', encoding="utf-8")
            with patch.dict(os.environ, {"OLD_FIXTURE_KEY": "old-env-key", "DRUDGE_API_KEY": "unrelated-env-key"}):
                config.load_codex(str(source))
                model = config.get_model_config()
                self.assertEqual(model["api_key"], "")
                self.assertEqual(model["api_key_env"], "")
                self.assertEqual(model["headers"], {})
                self.assertEqual(model["query_params"], {})
                self.assertEqual(model["auth_mode"], "api_key")
                self.assertTrue(model["allow_unauthenticated"])

    def test_probe_normalizes_native_preset_and_alias_before_selecting_routes(self):
        for settings in ({"provider": "anthropic"}, {"api": "anthropic_messages", "base_url": "https://fixture.invalid/v1"}):
            with self.subTest(settings=settings):
                seen = []
                def factory(config):
                    seen.append(config)
                    return FakeProbeClient(config)
                raw = {"name": "fixture", "api_key_env": "", **settings}
                report = asyncio.run(probe_provider(raw, client_factory=factory))
                self.assertTrue(all(item["api"] == "anthropic" for item in seen))
                self.assertEqual(set(report.capabilities), {"anthropic.basic", "anthropic.tools", "anthropic.streaming"})
                self.assertEqual(raw, {"name": "fixture", "api_key_env": "", **settings})

    def test_probe_requires_final_text_and_real_streaming_deltas(self):
        class EmptyClient(FakeProbeClient):
            async def chat(self, messages, tools=None, stream_callback=None, **kwargs):
                return {"model": "fixture", "choices": [{"message": {"content": "  "}}]}
        class FinalOnlyClient(EmptyClient):
            async def chat(self, *args, **kwargs):
                return {"model": "fixture", "choices": [{"message": {"content": "OK"}}]}
        for factory in (EmptyClient, FinalOnlyClient):
            with self.subTest(factory=factory.__name__):
                report = asyncio.run(probe_provider({"name": "fixture", "api": "anthropic"}, client_factory=factory))
                self.assertFalse(report.capabilities["anthropic.streaming"].supported)
                self.assertFalse(report.capabilities["anthropic.streaming"].streamed)
                self.assertEqual(report.capabilities["anthropic.basic"].supported, factory is FinalOnlyClient)
                self.assertTrue(report.capabilities["anthropic.streaming"].error)

    def test_responses_partial_arguments_and_reasoning_are_not_replayed(self):
        for event_type in ("response.function_call_arguments.delta", "response.reasoning_summary_text.delta", "response.output_item.added"):
            with self.subTest(event_type=event_type):
                class PartialStream(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield ("data: " + json.dumps({"type": event_type, "delta": "fragment"}) + "\n\n").encode()
                        raise httpx.ReadError("fixture stream interrupted")
                calls = []
                def handle(request):
                    calls.append(request)
                    return httpx.Response(200, stream=PartialStream())
                client = LLMClient("https://fixture.invalid/v1", "fixture-key", "fixture", api_type="responses", max_retries=3, model_aliases={"fixture": "alias"}, transport=httpx.MockTransport(handle))
                with self.assertRaisesRegex(RuntimeError, "partial output"):
                    asyncio.run(client.chat([{"role": "user", "content": "test"}], stream_callback=lambda text: None))
                self.assertEqual(len(calls), 1)

    def test_responses_stream_errors_redact_configured_keys(self):
        for event in ({"type": "error", "message": "failure fixture-secret-key"}, {"type": "response.failed", "response": {"error": {"message": "failure fixture-secret-key"}}}):
            with self.subTest(event=event):
                client = LLMClient("https://fixture.invalid/v1", "fixture-secret-key", "fixture", api_type="responses", transport=httpx.MockTransport(lambda request: httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")))
                with self.assertRaises(RuntimeError) as raised:
                    asyncio.run(client.chat([{"role": "user", "content": "test"}], stream_callback=lambda text: None))
                self.assertNotIn("fixture-secret-key", str(raised.exception))
                self.assertIn("[REDACTED]", str(raised.exception))
