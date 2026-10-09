"""Offline compatibility checks for locally imported Codex-format skills."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from agent.skills import SkillManager
from tests.fakes import FakeLLM, chat_response
from tests.test_sessions_instructions_skills import configured


class CodexSkillCompatibilityTests(unittest.TestCase):
    def make_skill(self, workspace):
        root = Path(workspace, ".drudge", "skills", "codex-fixture")
        (root / "references").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "agents").mkdir()
        (root / "SKILL.md").write_text(
            "---\n"
            "name: codex-fixture\n"
            "description: Review a local fixture\n"
            "metadata:\n"
            "  short-description: Codex-format fixture\n"
            "allowed-tools: terminal\n"
            "security:\n"
            "  approval_mode: auto\n"
            "  allow_outside_workspace: true\n"
            "---\n\n"
            "# Local review\n"
            "Inspect references/guide.md before using scripts/check.py.\n"
            "An upstream example mentions sandbox_permissions=require_escalated.\n",
            encoding="utf-8",
        )
        (root / "references" / "guide.md").write_text("REFERENCE_BODY_NOT_EAGERLY_LOADED", encoding="utf-8")
        (root / "scripts" / "check.py").write_text("raise AssertionError('must not auto-run')\n", encoding="utf-8")
        (root / "agents" / "openai.yaml").write_text(
            "interface:\n  display_name: Fixture\n  default_prompt: UI_METADATA_ONLY\n",
            encoding="utf-8",
        )
        return root

    def test_codex_metadata_and_resources_load_without_automatic_execution(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = self.make_skill(workspace)
            manager = SkillManager(workspace, drudge_home=Path(workspace, "home"))
            with patch("subprocess.run") as run:
                skill = manager.get("codex-fixture")
            run.assert_not_called()
            self.assertEqual(skill.path, (root / "SKILL.md").resolve())
            self.assertEqual(skill.metadata["metadata"]["short-description"], "Codex-format fixture")
            self.assertEqual(skill.scripts, {})
            self.assertEqual(skill.references, [])
            self.assertIn(str(root.resolve()), skill.render())
            self.assertIn("references/guide.md", skill.render())
            self.assertNotIn("REFERENCE_BODY_NOT_EAGERLY_LOADED", skill.render())
            self.assertNotIn("UI_METADATA_ONLY", skill.render())

    def test_activation_and_resume_preserve_host_permissions_and_portability_notes(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.make_skill(workspace)
            config = configured(workspace, str(Path(workspace, "sessions.db")))
            config.override("toolsets", value=[])
            config.override("security", "approval_mode", value="never")
            config.override("security", "allow_terminal", value=False)
            config.override("security", "allow_network", value=False)
            agent = Agent(config)
            self.assertEqual(agent.active_skill_names, [])
            self.assertNotIn("DRUDGE SKILL EXECUTION", agent._build_system_content())
            agent.activate_skill("codex-fixture")
            agent.llm = FakeLLM([chat_response("fixture loaded")])
            self.assertEqual(asyncio.run(agent.run("Check the imported skill")), "fixture loaded")

            resumed = Agent(config)
            session = resumed.resume_session(agent.session_id)
            self.assertEqual(session["active_skills"], ["codex-fixture"])
            for current in (agent, resumed):
                with self.subTest(resumed=current is resumed):
                    system = current.get_messages()[0]["content"]
                    self.assertIn("# Local review", system)
                    self.assertIn("DRUDGE SKILL EXECUTION", system)
                    self.assertIn("not Drudge tool arguments", system)
                    self.assertIn("relative to the loaded skill's Directory", system)
                    self.assertIn("does not install dependencies", system)
                    self.assertNotIn("REFERENCE_BODY_NOT_EAGERLY_LOADED", system)
                    context = current.tool_context
                    self.assertEqual(context.approval_mode, "never")
                    self.assertFalse(context.allow_terminal)
                    self.assertFalse(context.allow_network)
                    self.assertFalse(context.allow_outside_workspace)
                    self.assertEqual(context.enabled_toolsets, frozenset())

    def test_deactivation_removes_loaded_instructions_but_keeps_discovery(self):
        with tempfile.TemporaryDirectory() as workspace:
            self.make_skill(workspace)
            config = configured(workspace, str(Path(workspace, "sessions.db")))
            config.override("storage", "enabled", value=False)
            agent = Agent(config)
            agent.activate_skill("codex-fixture")
            self.assertIn("# Local review", agent._build_system_content())
            self.assertTrue(agent.deactivate_skill("codex-fixture"))
            system = agent._build_system_content()
            self.assertNotIn("# Local review", system)
            self.assertNotIn("DRUDGE SKILL EXECUTION", system)
            self.assertIn("codex-fixture: Review a local fixture", system)
