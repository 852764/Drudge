from __future__ import annotations

import asyncio
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import yaml

from agent import Agent
from agent.llm import LLMClient
from config import ConfigManager
from prompt import build_system_prompt
from scripts.configure_browser import configure, main, parse_args
from tools import ToolContext, create_tool_provider, registry
from tools.mcp_presets import BROWSER_TOOLS, expand_mcp_preset
from tools.provider import MCPServerProvider


class BrowserMCPTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config = {
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "browser_mcp_server.py")],
            "timeout": 5,
            "allowed_tools": ["echo", "navigate_page"],
        }
        self.context = ToolContext(self.root, frozenset(), approval_mode="on_request")

    def provider(self, **overrides):
        return MCPServerProvider("browser", {**self.config, **overrides}, self.root)

    def test_preset_is_explicit_isolated_headless_and_narrow_by_default(self):
        raw = {"preset": "jsreverser", "server_path": "server with spaces/build/src/index.js"}
        expanded = expand_mcp_preset(raw, self.root)
        self.assertNotIn("command", raw)
        self.assertIn("--headless=true", expanded["args"])
        self.assertIn("--isolated", expanded["args"])
        self.assertEqual(expanded["args"][1:3], ["--toolProfile", "full"])
        self.assertFalse(expanded["inherit_env"])
        self.assertTrue(expanded["requires_network"])
        self.assertEqual(expanded["risk"], "medium")
        self.assertEqual(expanded["allowed_tools"], list(BROWSER_TOOLS))
        self.assertNotIn("get_storage", expanded["allowed_tools"])
        self.assertNotIn("run_reverse_agent", expanded["allowed_tools"])
        self.assertNotIn("query_dom", expanded["allowed_tools"])
        self.assertNotIn("check_browser_health", expanded["allowed_tools"])
        self.assertIn("evaluate_script", expanded["allowed_tools"])
        self.assertIn("breakpoint", expanded["allowed_tools"])
        self.assertFalse((self.root / ".drudge").exists())
        self.assertNotIn("allowed_tools", expand_mcp_preset({**raw, "profile": "full"}, self.root))

    def test_attach_mode_never_launches_or_auto_discovers_personal_browser(self):
        for url in ("http://127.0.0.1:9222", "http://localhost:9333", "http://[::1]:9222/"):
            expanded = expand_mcp_preset({"preset": "jsreverser", "server_path": "index.js", "browser_url": url}, self.root)
            self.assertEqual(expanded["args"][-2:], ["--browserUrl", url])
            self.assertNotIn("--isolated", expanded["args"])
            self.assertNotIn("--autoConnect", expanded["args"])

    def test_invalid_presets_and_connection_options_fail_closed(self):
        base = {"preset": "jsreverser", "server_path": "index.js"}
        for fields in ({"preset": "typo"}, {"server_path": ""}, {"profile": "unknown"},
                       {"headless": "false"}, {"browser_url": "http://example.com:9222"},
                       {"browser_url": "http://user:pass@localhost:9222"},
                       {"browser_url": "http://localhost:99999"},
                       {"browser_url": "http://localhost:9222/page"},
                       {"browser_url": "http://localhost:9222", "executable_path": "chrome"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                expand_mcp_preset({**base, **fields}, self.root)

    def test_preset_respects_disabled_server_and_no_tools(self):
        servers = {"browser": {"preset": "jsreverser", "server_path": "index.js", "enabled": False}}
        provider = create_tool_provider(registry, [], servers, self.root)
        self.assertEqual(len(provider.providers), 1)
        config = ConfigManager()
        config.override("storage", "enabled", value=False)
        config.override("agent", "tools_enabled", value=False)
        config.override("mcp_servers", value={"browser": {"preset": "jsreverser"}})
        self.assertEqual(Agent(config).tool_provider.schemas(), [])

    def test_large_discovery_allowlist_and_concurrent_requests(self):
        async def exercise():
            provider = self.provider()
            await provider.start()
            try:
                self.assertEqual(set(provider.tool_names()), {"mcp__browser__echo", "mcp__browser__navigate_page"})
                self.assertEqual({s["function"]["name"] for s in provider.schemas()}, set(provider.tool_names()))
                self.assertEqual({s["name"] for s in provider.catalog()}, set(provider.tool_names()))
                bad = json.loads(await provider.call("mcp__browser__not_allowed", {}, self.context, approved=True))
                self.assertFalse(bad["ok"])
                responses = await asyncio.gather(*[
                    provider.call("mcp__browser__echo", {"text": str(i)}, self.context, approved=True)
                    for i in range(4)
                ])
                for i, raw in enumerate(responses):
                    self.assertEqual(json.loads(json.loads(raw)["content"])["text"], str(i))
            finally:
                await provider.close()
        asyncio.run(exercise())

    def test_large_stderr_is_drained_and_bounded(self):
        async def exercise():
            provider = self.provider()
            await provider.start()
            try:
                result = json.loads(await provider.call("mcp__browser__echo", {"stderr": True}, self.context, approved=True))
                self.assertTrue(result["ok"], result)
                await asyncio.sleep(0.02)
                self.assertTrue(provider._stderr_lines)
                self.assertLessEqual(len(provider._stderr_lines), 20)
                self.assertTrue(all(len(line) <= 2000 for line in provider._stderr_lines))
            finally:
                await provider.close()
        asyncio.run(exercise())

    def test_notifications_do_not_extend_request_deadline(self):
        async def exercise():
            provider = self.provider()
            await provider.start()
            try:
                provider.timeout = 0.15
                start = time.monotonic()
                result = json.loads(await provider.call("mcp__browser__echo", {"flood": True}, self.context, approved=True))
                self.assertFalse(result["ok"])
                self.assertIn("remote work may still be running", result["error"])
                self.assertLess(time.monotonic() - start, 2)
            finally:
                await provider.close()
        asyncio.run(exercise())

    def test_message_budget_and_repeated_pagination_cleanup_process(self):
        async def exercise():
            for options in ({"max_message_bytes": 4096}, {"env": {"FIXTURE_REPEAT_CURSOR": "1"}}):
                provider = self.provider(**options)
                with self.assertRaises((ValueError, RuntimeError)):
                    await provider.start()
                self.assertIsNone(provider.process)
        asyncio.run(exercise())

    def test_no_environment_secrets_are_inherited_when_disabled(self):
        async def exercise():
            provider = self.provider(inherit_env=False, env_passthrough=["SYSTEMROOT", "WINDIR"], env={"FIXTURE_VISIBLE": "fixture"})
            await provider.start()
            try:
                result = json.loads(await provider.call("mcp__browser__echo", {}, self.context, approved=True))
                content = json.loads(result["content"])
                self.assertFalse(content["has_secret"])
                self.assertEqual(content["visible"], "fixture")
            finally:
                await provider.close()
        with patch.dict(os.environ, {"FIXTURE_SECRET": "fake-secret"}):
            asyncio.run(exercise())

    def test_network_and_approval_are_checked_before_reconnect(self):
        async def exercise():
            provider = self.provider(requires_network=True)
            provider._tools = {"mcp__browser__echo": {"name": "echo"}}
            provider._ensure_connected = AsyncMock()
            for context, approved in ((replace(self.context, allow_network=False), True),
                                      (self.context, False), (replace(self.context, approval_mode="never"), True)):
                args = {"approved": True, "allow_network": True, "context": {"approval_mode": "auto"}}
                result = json.loads(await provider.call("mcp__browser__echo", args, context, approved=approved))
                self.assertTrue(result["blocked"])
            provider._ensure_connected.assert_not_awaited()
        asyncio.run(exercise())

    def test_real_stdio_tools_round_trip_through_both_model_api_adapters(self):
        class OfflineClient(LLMClient):
            def __init__(self, api):
                super().__init__(base_url="https://example.invalid", api_key="offline", model="offline", api_type=api)
                self.api = api
                self.requests = []

            async def _post_json(self, url, body):
                self.requests.append(body)
                if len(self.requests) == 1:
                    name = "mcp__browser__echo"
                    arguments = json.dumps({"text": "offline-browser-marker"})
                    if self.api == "responses":
                        return {"status": "completed", "output": [{"type": "function_call", "call_id": "call-1", "name": name, "arguments": arguments}]}
                    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}]}}]}
                if self.api == "responses":
                    return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "verified"}]}]}
                return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "verified"}}]}

        for api in ("chat", "responses"):
            with self.subTest(api=api):
                config = ConfigManager()
                for section, key, value in (
                    ("storage", "enabled", False), ("tool_selection", "enabled", False),
                    ("agent", "repo_map_enabled", False), ("agent", "instructions_enabled", False),
                    ("display", "show_tool_calls", False), ("security", "workspace_root", str(self.root)),
                    ("security", "approval_mode", "auto"),
                ):
                    config.override(section, key, value=value)
                config.override("toolsets", value=[])
                config.override("mcp_servers", value={"browser": self.config})
                agent = Agent(config)
                agent.llm = client = OfflineClient(api)
                self.assertEqual(asyncio.run(agent.run("Call the fixture once")), "verified")
                self.assertEqual(len(client.requests), 2)
                self.assertIn("offline-browser-marker", json.dumps(client.requests[1]))

    def test_cancelled_discovery_closes_process_and_empty_allowlist_stays_empty(self):
        async def exercise():
            provider = self.provider()
            provider._load_tools = AsyncMock(side_effect=asyncio.CancelledError)
            with self.assertRaises(asyncio.CancelledError):
                await provider.start()
            self.assertIsNone(provider.process)
            provider = self.provider(allowed_tools=[])
            try:
                await provider.start()
                self.assertEqual(provider.schemas(), [])
                self.assertEqual(provider.catalog(), [])
            finally:
                await provider.close()
        asyncio.run(exercise())

    def test_close_stops_descendants_holding_inherited_pipes(self):
        async def exercise():
            provider = self.provider()
            await provider.start()
            process = provider.process
            try:
                result = json.loads(await provider.call("mcp__browser__echo", {"child_ready": str(self.root / "ready")}, self.context, approved=True))
                self.assertTrue(result["ok"], result)
                self.assertTrue((self.root / "ready").exists())
            finally:
                await asyncio.wait_for(provider.close(), 4)
            self.assertIsNotNone(process.returncode)
            self.assertTrue(process.stdout.at_eof())
        asyncio.run(exercise())

    def test_host_option_validation(self):
        for options in ({"timeout": float("nan")}, {"timeout": 0}, {"max_message_bytes": True},
                        {"max_message_bytes": 10}, {"allowed_tools": "echo"},
                        {"inherit_env": "false"}, {"env_passthrough": "PATH"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.provider(**options)

    def test_browser_prompt_hints_are_not_claimed_when_tools_disabled(self):
        self.assertIn("MCP timeout", build_system_prompt(["file"]))
        self.assertNotIn("Start with list_pages", build_system_prompt([], tools_enabled=False))
        self.assertIn("untrusted data", build_system_prompt(["file"]))


class BrowserSetupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.entry = self.root / "index.js"
        self.entry.write_text("// fixture", encoding="utf-8")
        self.path = self.root / "config.local.yaml"

    def args(self, *extra):
        return parse_args(["--server-path", str(self.entry), "--config", str(self.path), *extra])

    def test_cli_options_merge_preserves_model_approval_and_other_servers(self):
        self.path.write_text(yaml.safe_dump({
            "model": {"api_key": "fake-key", "name": "unchanged"},
            "security": {"approval_mode": "never", "allow_network": False},
            "mcp_servers": {"other": {"command": "helper"}},
            "tool_selection": {"always_include": ["read_file"]},
        }), encoding="utf-8")
        configure(self.args())
        value = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.assertEqual(value["model"], {"api_key": "fake-key", "name": "unchanged"})
        self.assertEqual(value["security"], {"approval_mode": "never", "allow_network": False})
        self.assertEqual(value["mcp_servers"]["other"], {"command": "helper"})
        self.assertEqual(value["mcp_servers"]["browser"]["profile"], "browser")
        self.assertEqual(len(value["tool_selection"]["always_include"]), 3)
        with self.assertRaisesRegex(ValueError, "already configured"):
            configure(self.args())
        configure(self.args("--replace", "--enable-network", "--profile", "full", "--headed"))
        updated = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.assertTrue(updated["security"]["allow_network"])
        self.assertEqual(updated["security"]["approval_mode"], "never")
        self.assertFalse(updated["mcp_servers"]["browser"]["headless"])
        self.assertEqual(len(updated["tool_selection"]["always_include"]), 3)

    def test_bad_configuration_and_failed_probe_do_not_write(self):
        for source in ("[]", "security: false", "mcp_servers: []", "tool_selection:\n  always_include: no", "x: ["):
            self.path.write_text(source, encoding="utf-8")
            with self.subTest(source=source), self.assertRaises(ValueError):
                configure(self.args())
            self.assertEqual(self.path.read_text(encoding="utf-8"), source)
        self.path.write_text("{}", encoding="utf-8")
        with patch("scripts.configure_browser.probe_server", new=AsyncMock(side_effect=RuntimeError("offline"))):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                configure(self.args("--probe"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{}")

    def test_credential_file_rejected_before_read(self):
        args = self.args()
        args.config = str(self.root / ".drudge" / "auth.json")
        with patch.object(Path, "read_bytes") as read, self.assertRaises(ValueError):
            configure(args)
        read.assert_not_called()

    def test_main_failure_returns_nonzero_without_printing_existing_keys(self):
        self.path.write_text("model: {api_key: fake-key}", encoding="utf-8")
        with patch("sys.stderr", new_callable=io.StringIO) as output:
            code = main(["--server-path", str(self.root / "missing.js"), "--config", str(self.path)])
        self.assertEqual(code, 1)
        self.assertNotIn("fake-key", output.getvalue())


if __name__ == "__main__":
    unittest.main()
