from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.project_instructions import load_project_instructions
from agent.skills import SkillManager
from tools import ToolContext, registry


class ContextLoadingSafetyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def test_host_path_guard_protects_credentials_even_with_outside_access(self):
        context = ToolContext(self.root, frozenset({"file"}), allow_outside_workspace=True)
        for filename in (".drudge/auth.json", ".codex/auth.json", ".CODEX/AUTH.JSON"):
            with self.subTest(filename=filename), patch.object(Path, "open") as open_file:
                for name, arguments in (
                    ("read_file", {}), ("search_files", {"pattern": "."}),
                    ("write_file", {"content": "fictional"}),
                    ("patch", {"old_string": "old", "new_string": "new"}),
                ):
                    result = json.loads(registry.dispatch(name, {"path": filename, **arguments}, context=context))
                    self.assertTrue(result["blocked"])
                open_file.assert_not_called()

    def test_instruction_filename_cannot_load_credentials(self):
        for filename in (".drudge/auth.json", ".codex/auth.json"):
            with self.subTest(filename=filename), patch.object(Path, "is_file", lambda path: path.name == "auth.json"), patch.object(Path, "open") as open_file:
                self.assertEqual(load_project_instructions(self.root, filename=filename), [])
                open_file.assert_not_called()

    def test_instruction_alias_to_credentials_is_excluded(self):
        alias = self.root / "AGENTS.md"
        alias.write_text("ordinary instructions", encoding="utf-8")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            return self.root / ".codex" / "auth.json" if path == alias else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve), patch.object(Path, "open") as open_file:
            self.assertEqual(load_project_instructions(self.root), [])
            open_file.assert_not_called()

    def test_instruction_context_stays_bounded_and_in_workspace(self):
        (self.root / "AGENTS.md").write_text("0123456789", encoding="utf-8")
        nested = self.root / "child"
        nested.mkdir()
        (nested / "AGENTS.md").write_text("nested", encoding="utf-8")
        loaded = load_project_instructions(self.root, cwd=nested, max_chars=4)
        self.assertEqual([item.content for item in loaded], ["0123"])
        loaded = load_project_instructions(self.root, cwd=self.root.parent, max_chars=12)
        self.assertEqual([item.path for item in loaded], [self.root / "AGENTS.md"])

    def test_skill_references_exclude_credentials_without_opening_them(self):
        manager = SkillManager(self.root, drudge_home=self.root / "home")
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "open") as open_file:
            references = manager._load_references(self.root, [".drudge/auth.json", ".codex/auth.json"])
            self.assertEqual(references, [])
            open_file.assert_not_called()

    def test_skill_document_alias_to_credentials_is_excluded(self):
        manager = SkillManager(self.root, drudge_home=self.root / "home")
        skill_dir = self.root / ".drudge" / "skills" / "review"
        skill_dir.mkdir(parents=True)
        alias = skill_dir / "SKILL.md"
        alias.write_text("Review files", encoding="utf-8")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            return skill_dir / "auth.json" if path == alias else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve), patch.object(Path, "open") as open_file:
            self.assertEqual(manager.discover(), {})
            open_file.assert_not_called()

    def test_search_reauthorizes_discovered_paths(self):
        # Simulate symlink resolution so this also runs on Windows without
        # requiring symlink privileges. No external or credential file is read.
        alias = self.root / "alias.txt"
        alias.write_text("should not be read", encoding="utf-8")
        original_resolve = Path.resolve
        context = ToolContext(self.root, frozenset({"file"}))
        for target in (self.root.parent / "external.txt", self.root / ".drudge" / "auth.json"):
            with self.subTest(target=target):
                def resolve(path, *args, **kwargs):
                    return target if path == alias else original_resolve(path, *args, **kwargs)

                with patch.object(Path, "resolve", resolve), patch("builtins.open") as open_file:
                    result = json.loads(registry.dispatch("search_files", {"pattern": ".", "path": "."}, context=context))
                    self.assertEqual(result["matches"], [])
                    open_file.assert_not_called()
