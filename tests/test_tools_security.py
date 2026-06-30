from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tools import ToolContext, registry


class ToolSecurityTests(unittest.TestCase):
    def test_schema_rejects_additional_properties(self):
        schema = next(
            item for item in registry.get_schemas(["file"])
            if item["function"]["name"] == "read_file"
        )
        self.assertFalse(schema["function"]["parameters"]["additionalProperties"])

    def test_model_cannot_override_runtime_context(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}))
            target = Path(outside) / "escaped.txt"
            result = asyncio.run(registry.dispatch_async(
                "write_file",
                {
                    "path": str(target),
                    "content": "bad",
                    "allow_outside_workspace": True,
                },
                context=context,
            ))
            payload = json.loads(result)
            self.assertTrue(payload["blocked"])
            self.assertIn("Unknown tool arguments", payload["error"])
            self.assertFalse(target.exists())

    def test_path_traversal_is_blocked(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}))
            result = asyncio.run(registry.dispatch_async(
                "read_file",
                {"path": "../outside.txt"},
                context=context,
            ))
            payload = json.loads(result)
            self.assertTrue(payload["blocked"])
            self.assertIn("outside workspace", payload["error"].lower())

    def test_disabled_toolset_cannot_be_dispatched(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}))
            result = asyncio.run(registry.dispatch_async(
                "terminal",
                {"command": "echo should-not-run"},
                context=context,
            ))
            payload = json.loads(result)
            self.assertTrue(payload["blocked"])
            self.assertIn("disabled for this run", payload["error"])

    def test_terminal_permission_is_enforced_by_context(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(
                Path(workspace).resolve(),
                frozenset({"terminal"}),
                allow_terminal=False,
            )
            result = asyncio.run(registry.dispatch_async(
                "terminal",
                {"command": "echo should-not-run"},
                context=context,
            ))
            payload = json.loads(result)
            self.assertTrue(payload["blocked"])
            self.assertIn("disabled by config", payload["error"])

    def test_tool_result_envelope_and_apply_patch(self):
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace, "sample.txt")
            target.write_text("hello old", encoding="utf-8")
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}))

            result = asyncio.run(registry.dispatch_async(
                "apply_patch",
                {"path": "sample.txt", "old_string": "old", "new_string": "new"},
                context=context,
            ))

            payload = json.loads(result)
            self.assertTrue(payload["ok"])
            self.assertIsNone(payload["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), "hello new")

    def test_approval_never_blocks_mutation(self):
        with tempfile.TemporaryDirectory() as workspace:
            context = ToolContext(
                Path(workspace).resolve(),
                frozenset({"file"}),
                approval_mode="never",
            )
            result = asyncio.run(registry.dispatch_async(
                "write_file",
                {"path": "blocked.txt", "content": "nope"},
                context=context,
            ))

            payload = json.loads(result)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["blocked"])
            self.assertFalse(Path(workspace, "blocked.txt").exists())

    def test_sensitive_auth_file_read_is_blocked(self):
        with tempfile.TemporaryDirectory() as workspace:
            auth_dir = Path(workspace, ".drudge")
            auth_dir.mkdir()
            Path(auth_dir, "auth.json").write_text("secret", encoding="utf-8")
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}))

            result = asyncio.run(registry.dispatch_async(
                "read_file",
                {"path": ".drudge/auth.json"},
                context=context,
            ))

            payload = json.loads(result)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["blocked"])
            self.assertIn("credential", payload["error"])


if __name__ == "__main__":
    unittest.main()
