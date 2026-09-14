from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from agent.context_manager import build_repo_map
from agent.llm import LLMClient
from config import ConfigManager
from tests.fakes import chat_response, function_call
from tools import ToolContext, registry
from tools.repository import RepositoryWalker, MAX_IGNORE_BYTES
from tools.result import limit_tool_result


class RepositorySearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.context = ToolContext(self.root, frozenset({"file"}))

    def write(self, name, content="NEEDLE"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    def search(self, *, context=None, **args):
        return json.loads(registry.dispatch("search_files", {"pattern": "NEEDLE", **args}, context=context or self.context))

    def names(self, result):
        return [Path(item["file"]).relative_to(self.root).as_posix() for item in result["matches"]]

    def test_match_after_old_200_file_cutoff_is_found(self):
        for index in range(250):
            self.write(f"f{index:03}.txt", "ordinary")
        self.write("z_last.txt")
        result = self.search()
        self.assertEqual(self.names(result), ["z_last.txt"])
        self.assertEqual(result["files_scanned"], 251)
        self.assertTrue(result["complete"])
        self.assertFalse(result["truncated"])

    def test_hidden_ancestor_does_not_hide_workspace(self):
        self.write(".parent/project/source.py")
        context = replace(self.context, workspace=self.root / ".parent/project")
        result = self.search(context=context)
        self.assertEqual(result["total"], 1)
        self.assertIn("source.py", build_repo_map(context.workspace))

    def test_root_and_nested_ignores_apply_to_scoped_searches(self):
        self.write(".gitignore", "*.tmp\n/root_only.txt\nskip/\n")
        self.write("src/.gitignore", "!keep.tmp\nlocal.txt\n")
        for name in ("root_only.txt", "src/root_only.txt", "src/keep.tmp", "src/drop.tmp", "src/local.txt", "src/source.py", "skip/x.py"):
            self.write(name)
        expected = ["src/keep.tmp", "src/root_only.txt", "src/source.py"]
        self.assertEqual(self.names(self.search()), expected)
        self.assertEqual(self.names(self.search(path="src")), expected)
        repo_map = build_repo_map(self.root)
        self.assertIn("keep.tmp", repo_map)
        self.assertNotIn("drop.tmp", repo_map)
        self.assertNotIn("local.txt", repo_map)

    def test_ignored_parent_is_pruned_before_child_rules_are_read(self):
        self.write(".gitignore", "vendor/\n")
        nested = self.write("vendor/.gitignore", "!keep.py\n")
        self.write("vendor/keep.py")
        original = Path.open
        opened = []
        def observe(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", observe):
            self.assertEqual(self.search()["total"], 0)
        self.assertNotIn(nested, opened)

    def test_hidden_option_does_not_override_private_directory_exclusions(self):
        for name in (".config.txt", ".github/workflows/check.yml", ".codex/private.txt", ".drudge/private.txt", ".env", ".env.example", "node_modules/dep.js"):
            self.write(name)
        self.assertEqual(self.search()["total"], 0)
        result = self.search(include_hidden=True)
        self.assertEqual(self.names(result), [".config.txt", ".env.example", ".github/workflows/check.yml"])
        self.assertTrue(result["complete"])

    def test_exact_match_limit_at_eof_is_complete(self):
        self.write("a.txt", "NEEDLE\nNEEDLE\n")
        exact = self.search(limit=2)
        self.assertTrue(exact["complete"])
        self.assertFalse(exact["truncated"])
        self.write("a.txt", "NEEDLE\nNEEDLE\nNEEDLE\n")
        capped = self.search(limit=2)
        self.assertEqual(capped["total"], 2)
        self.assertFalse(capped["complete"])
        self.assertEqual(capped["incomplete_reasons"], ["match_limit"])

    def test_file_byte_and_entry_caps_are_explicit(self):
        for name in ("a.txt", "b.txt", "c.txt"):
            self.write(name, "ordinary")
        limited = self.search(context=replace(self.context, search_max_files=2))
        self.assertEqual(limited["files_scanned"], 2)
        self.assertIn("file_limit", limited["incomplete_reasons"])
        limited = self.search(context=replace(self.context, search_max_total_bytes=8))
        self.assertEqual(limited["bytes_read"], 8)
        self.assertIn("byte_limit", limited["incomplete_reasons"])
        limited = self.search(context=replace(self.context, search_max_entries=2))
        self.assertEqual(limited["files_scanned"], 0)
        self.assertIn("entry_limit", limited["incomplete_reasons"])

    def test_binary_invalid_utf8_and_large_files_are_accounted(self):
        self.write("binary.txt", b"NEEDLE\0")
        self.write("invalid.txt", b"NEEDLE\xff")
        self.write("large.txt", "NEEDLE" * 20)
        self.write("text.txt")
        result = self.search(context=replace(self.context, search_max_file_bytes=20))
        self.assertEqual(self.names(result), ["text.txt"])
        self.assertEqual(result["skipped"]["binary"], 1)
        self.assertEqual(result["skipped"]["non_utf8"], 1)
        self.assertEqual(result["skipped"]["too_large"], 1)
        self.assertIn("file_size_limit", result["incomplete_reasons"])

    def test_explicit_ignored_file_is_still_subject_to_host_budget_and_authorization(self):
        self.write("build/output.log")
        self.assertEqual(self.search()["total"], 0)
        self.assertEqual(self.search(path="build/output.log", file_glob="*.py")["total"], 1)
        result = self.search(path="build/output.log", context=replace(self.context, search_max_file_bytes=3))
        self.assertIn("file_size_limit", result["incomplete_reasons"])
        for name in (".drudge/auth.json", ".codex/auth.json"):
            with patch("builtins.open") as opened:
                self.assertTrue(self.search(path=name, include_hidden=True)["blocked"])
                opened.assert_not_called()

    def test_literal_case_and_unicode_match_centered_columns(self):
        line = " " * 4 + "x" * 300 + "价格[0] = NEEDLE"
        self.write("unicode.txt", ("\ufeff" + line + "\nneedle").encode("utf-8"))
        result = self.search(pattern="价格[0]", literal=True, case_sensitive=True)
        match = result["matches"][0]
        self.assertEqual(match["line"], 1)
        self.assertEqual(match["column"], line.index("价格") + 1)
        self.assertGreater(match["content_start_column"], 1)
        self.assertIn("价格[0]", match["content"])
        self.assertEqual(self.search(pattern="NEEDLE", case_sensitive=True)["total"], 1)
        self.assertEqual(self.search(pattern="NEEDLE")["total"], 2)

    def test_globs_match_root_files_and_zero_or_multiple_directories(self):
        for name in ("a.py", "src/b.py", "src/nested/c.py", "folder.py/not_python.txt"):
            self.write(name)
        self.assertEqual(self.names(self.search(file_glob="*.py")), ["a.py", "src/b.py", "src/nested/c.py"])
        self.assertEqual(self.names(self.search(file_glob="**/*.py")), ["a.py", "src/b.py", "src/nested/c.py"])
        self.assertEqual(self.names(self.search(file_glob="src/**/*.py")), ["src/b.py", "src/nested/c.py"])

    def test_invalid_arguments_never_open_files(self):
        invalid = [{"limit": 0}, {"limit": True}, {"limit": 1001}, {"pattern": "["}, {"pattern": ""},
                   {"pattern": "x" * 4097}, {"include_hidden": 1}, {"literal": "true"},
                   {"file_glob": "../*.py"}, {"file_glob": "a/" * 65}, {"search_max_files": 999999},
                   {"context": {}}, {"approved": True}]
        with patch("builtins.open") as opened, patch.object(Path, "open") as path_opened:
            for args in invalid:
                with self.subTest(args=str(args)[:100]):
                    result = json.loads(registry.dispatch(
                        "search_files", {"pattern": "NEEDLE", **args}, context=self.context,
                    ))
                    self.assertFalse(result["ok"])
        opened.assert_not_called()
        path_opened.assert_not_called()

    def test_ignore_alias_to_credentials_fails_closed_without_reading_it(self):
        ignore = self.write(".gitignore", "ordinary")
        self.write("public.txt")
        original = Path.resolve
        def resolve(path, *args, **kwargs):
            return self.root / ".codex/auth.json" if path == ignore else original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve), patch.object(Path, "open") as opened, patch("builtins.open") as builtins_open:
            result = self.search()
        opened.assert_not_called()
        builtins_open.assert_not_called()
        self.assertEqual(result["total"], 0)
        self.assertIn("ignore_unreadable", result["incomplete_reasons"])

    def test_oversized_ignore_file_fails_closed(self):
        self.write(".gitignore", b"x" * (MAX_IGNORE_BYTES + 1))
        self.write("public.txt")
        with patch.object(Path, "open") as opened:
            result = self.search()
        opened.assert_not_called()
        self.assertEqual(result["total"], 0)
        self.assertIn("ignore_limit", result["incomplete_reasons"])

    def test_ignore_changes_are_seen_on_next_call(self):
        self.write("a.py")
        self.assertEqual(self.search()["total"], 1)
        self.write(".gitignore", "*.py\n")
        self.assertEqual(self.search()["total"], 0)

    def test_bounded_search_preserves_completeness_separately_from_receipt(self):
        self.write("large_result.txt", ("NEEDLE " + "x" * 180 + "\n") * 80)
        result = self.search(limit=60)
        bounded = json.loads(limit_tool_result(result, save_output=lambda text: {"id": "a" * 32, "complete": True}))
        self.assertFalse(bounded["metadata"]["complete"])
        self.assertIn("match_limit", bounded["metadata"]["incomplete_reasons"])
        self.assertTrue(bounded["metadata"]["output_ref"]["complete"])
        self.assertEqual(bounded["metadata"]["files_scanned"], 1)

    def test_tiny_budget_keeps_incomplete_flag_and_bounds_reason_metadata(self):
        result = {"matches": [{"content": "x" * 20000}], "complete": False,
                  "incomplete_reasons": ["\x00" * 1000] * 100}
        for budget in (1024, 10000):
            encoded = limit_tool_result(result, max_chars=budget)
            self.assertLessEqual(len(encoded), budget)
            payload = json.loads(encoded)
            self.assertFalse(payload["metadata"]["complete"])
            self.assertLessEqual(len(payload["metadata"].get("incomplete_reasons", [])), 8)

    def test_map_prioritizes_root_files_and_caps_directory_only_output(self):
        self.write("root.txt")
        self.write("a/deep.txt")
        for index in range(100):
            (self.root / f"empty{index}").mkdir()
        mapped = build_repo_map(self.root, max_files=1)
        self.assertIn("root.txt", mapped)
        self.assertNotIn("deep.txt", mapped)
        self.assertIn("truncated after 1 files", mapped)
        self.assertLess(len(mapped.splitlines()), 5)
        self.assertIn("depth_limit", build_repo_map(self.root, max_depth=0))
        self.assertIn("listing disabled", build_repo_map(self.root, max_files=0))

    def test_host_limits_are_loaded_and_validated_without_model_overrides(self):
        context = ToolContext.from_config({"workspace_root": str(self.root), "search_max_files": 4}, ["file"])
        self.assertEqual(context.search_max_files, 4)
        for value in (0, -1, True, "10"):
            with self.assertRaises(ValueError):
                ToolContext.from_config({"search_max_entries": value}, ["file"])
        for args in ({"max_entries": 0}, {"max_depth": -1}, {"include_hidden": 1}):
            with self.assertRaises(ValueError):
                RepositoryWalker(self.root, **args)

    def test_both_api_loops_receive_search_scope_and_completion(self):
        self.write("source.txt")
        case = self
        class Client(LLMClient):
            def __init__(self, api):
                super().__init__("https://example.invalid", "offline", "offline", api_type=api)
                self.api, self.count = api, 0
            async def _post_json(self, url, body):
                self.count += 1
                if self.count == 1:
                    args = json.dumps({"pattern": "NEEDLE", "literal": True, "limit": 1})
                    schemas = [item["function"] for item in body["tools"]] if self.api == "chat" else body["tools"]
                    properties = next(item["parameters"]["properties"] for item in schemas if item["name"] == "search_files")
                    case.assertIn("literal", properties)
                    case.assertNotIn("search_max_files", properties)
                    if self.api == "chat":
                        return chat_response(finish_reason="tool_calls", tool_calls=[function_call("search", "search_files", args)])
                    return {"status": "completed", "output": [{"type": "function_call", "call_id": "search", "name": "search_files", "arguments": args}]}
                results = [item["content"] for item in body["messages"] if item["role"] == "tool"] if self.api == "chat" else [item["output"] for item in body["input"] if item.get("type") == "function_call_output"]
                result = json.loads(results[-1])
                case.assertTrue(result["complete"])
                case.assertEqual(result["total"], 1)
                if self.api == "chat":
                    return chat_response("found")
                return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "found"}]}]}
        for api in ("chat", "responses"):
            config = ConfigManager()
            config.override("storage", "enabled", value=False)
            config.override("security", "workspace_root", value=str(self.root))
            config.override("toolsets", value=["file"])
            config.override("display", "show_tool_calls", value=False)
            config.override("agent", "refusal_review_enabled", value=False)
            agent = Agent(config)
            agent.llm = Client(api)
            self.assertEqual(asyncio.run(agent.run("find NEEDLE")), "found")
