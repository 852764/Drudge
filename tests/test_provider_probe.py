import asyncio
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent.provider_probe import ProviderProbeReport, format_probe_report, probe_provider
from main import parse_args, run_doctor


class FakeProbeClient:
    def __init__(self, config):
        self.api_type = config["api"]
        self.model = config["name"]

    async def list_models(self):
        return ["other-model", self.model]

    async def chat(self, messages, tools=None, stream_callback=None, **kwargs):
        if self.api_type == "chat" and tools:
            raise RuntimeError("LLM request failed: HTTP 400 tool schema unsupported")
        if stream_callback:
            stream_callback("OK")
        tool_calls = []
        if tools:
            tool_calls = [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "probe_echo", "arguments": '{"value":"OK"}'},
            }]
        return {
            "model": self.model,
            "choices": [{
                "message": {"content": "OK", "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {},
        }

    @staticmethod
    def extract_text(response):
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def extract_tool_calls(response):
        return response["choices"][0]["message"].get("tool_calls", [])


class ProviderProbeTests(unittest.TestCase):
    def test_probe_builds_endpoint_capability_matrix(self):
        report = asyncio.run(probe_provider(
            {
                "name": "target-model",
                "base_url": "https://provider.example/v1",
                "api_key": "secret",
                "api": "auto",
            },
            client_factory=FakeProbeClient,
        ))

        self.assertTrue(report.listed)
        self.assertEqual(report.model_count, 2)
        self.assertTrue(report.capabilities["chat.basic"].supported)
        self.assertFalse(report.capabilities["chat.tools"].supported)
        self.assertEqual(report.capabilities["chat.tools"].status_code, 400)
        self.assertTrue(report.capabilities["chat.streaming"].streamed)
        self.assertTrue(report.capabilities["responses.tools"].tool_call_received)
        self.assertNotIn("secret", report.to_json())

        rendered = format_probe_report(report)
        self.assertIn("chat.tools: no", rendered)
        self.assertIn("responses.tools: yes", rendered)
        json.loads(report.to_json())

    def test_doctor_probe_cli_parsing(self):
        with patch.object(sys, "argv", [
            "main.py",
            "doctor",
            "--probe-model",
            "gpt-5.5",
            "--probe-json",
            "--no-probe-streaming",
        ]):
            args = parse_args()

        self.assertEqual(args.command, "doctor")
        self.assertEqual(args.probe_model, "gpt-5.5")
        self.assertTrue(args.probe_json)
        self.assertTrue(args.no_probe_streaming)

    def test_doctor_probe_json_is_machine_readable(self):
        report = ProviderProbeReport(
            base_url="https://provider.example/v1",
            model="target-model",
            listed=True,
            model_count=1,
        )
        output = io.StringIO()
        with patch("main._run_provider_probe", return_value=report):
            with redirect_stdout(output):
                run_doctor(probe_model="target-model", probe_json=True)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["model"], "target-model")
        self.assertTrue(payload["listed"])


if __name__ == "__main__":
    unittest.main()
