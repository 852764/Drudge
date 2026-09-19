from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tools import ToolContext, registry
from tools.context import is_within_path


class ToolSecurityTests(unittest.TestCase):
    def test_workspace_boundary_rejects_sibling_prefix_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "repo"
            sibling = base / "repo-sibling"
            workspace.mkdir()
            sibling.mkdir()
            target = sibling / "outside.txt"
            target.write_text("outside", encoding="utf-8")
            context = ToolContext.from_config({"workspace_root": str(workspace)}, ["file"])

            self.assertTrue(is_within_path(workspace, workspace))
            self.assertFalse(is_within_path(sibling, workspace))
            payload = json.loads(registry.dispatch("read_file", {"path": str(target)}, context=context))
            self.assertTrue(payload["blocked"])
            self.assertIn("outside workspace", payload["error"].lower())

    def test_default_context_blocks_unapproved_mutation(self):
        with tempfile.TemporaryDirectory() as workspace:
            for context in (
                ToolContext(Path(workspace).resolve(), frozenset({"file", "terminal"})),
                ToolContext.from_config({"workspace_root": workspace}, ["file", "terminal"]),
            ):
                for name, args in (
                    ("write_file", {"path": "blocked.txt", "content": "no"}),
                    ("terminal", {"command": "echo not-approved"}),
                ):
                    payload = json.loads(asyncio.run(registry.dispatch_async(name, args, context=context)))
                    self.assertFalse(payload["ok"])
                    self.assertTrue(payload["metadata"]["approval_required"])
            self.assertFalse(Path(workspace, "blocked.txt").exists())

    def test_invalid_approval_modes_fail_closed(self):
        for mode in ("", "automatic", "AUTO", None, True, []):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ToolContext(Path.cwd(), frozenset({"terminal"}), approval_mode=mode)
            with self.assertRaises(ValueError):
                ToolContext.from_config({"approval_mode": mode}, ["terminal"])

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
                approval_mode="auto",
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
            context = ToolContext(Path(workspace).resolve(), frozenset({"file"}), approval_mode="auto")

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
