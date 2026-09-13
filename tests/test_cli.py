from __future__ import annotations

import asyncio
import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from config import ConfigManager
from main import _handle_command, parse_args
from agent.cli_renderer import CliRenderer


class CliTests(unittest.TestCase):
    def test_codex_config_without_path_uses_default(self):
        with patch("sys.argv", ["drudge", "--codex-config"]):
            args = parse_args()

        self.assertEqual(
            Path(args.codex_config),
            ConfigManager.default_codex_config_path(),
        )

    def test_codex_config_accepts_explicit_path(self):
        with patch("sys.argv", ["drudge", "--codex-config", "custom.toml"]):
            args = parse_args()

        self.assertEqual(args.codex_config, "custom.toml")

    def test_auth_subcommand(self):
        with patch("sys.argv", ["drudge", "auth", "login", "--no-browser"]):
            args = parse_args()

        self.assertEqual(args.command, "auth")
        self.assertEqual(args.action, "login")
        self.assertTrue(args.no_browser)

    def test_doctor_subcommand(self):
        with patch("sys.argv", ["drudge", "--codex-oauth", "doctor"]):
            args = parse_args()

        self.assertEqual(args.command, "doctor")
        self.assertTrue(args.codex_oauth)

    def test_approval_mode_override(self):
        with patch("sys.argv", ["drudge", "--approval-mode", "on_request"]):
            args = parse_args()

        self.assertEqual(args.approval_mode, "on_request")

    def test_resume_and_repeatable_skills(self):
        with patch(
            "sys.argv",
            ["drudge", "--resume", "abc123", "--skill", "review", "--skill", "tests"],
        ):
            args = parse_args()

        self.assertEqual(args.resume, "abc123")
        self.assertEqual(args.skill, ["review", "tests"])

    def test_status_subcommand_json(self):
        with patch("sys.argv", ["drudge", "--codex-oauth", "status", "--json"]):
            args = parse_args()

        self.assertEqual(args.command, "status")
        self.assertTrue(args.status_json)

    def test_undo_dry_run_option_is_forwarded_and_preview_is_rendered(self):
        agent = Mock()
        agent.undo_last_file_change.return_value = {
            "id": 1, "path": "sample.txt", "undo_action": "restore",
            "undo_diff_summary": "-after\n+before", "dry_run": True,
        }
        stream = io.StringIO()
        renderer = CliRenderer(stream=stream, pretty=False)
        asyncio.run(_handle_command("/undo --dry-run", ConfigManager(), agent, renderer=renderer))
        agent.undo_last_file_change.assert_called_once_with(dry_run=True)
        self.assertIn("+before", stream.getvalue())
        self.assertIn("No files changed", stream.getvalue())

    def test_invalid_undo_options_never_execute(self):
        for command in ("/undo --force", "/undo all", "/undo --dry-run extra"):
            with self.subTest(command=command):
                agent = Mock()
                stream = io.StringIO()
                asyncio.run(_handle_command(command, ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
                agent.undo_last_file_change.assert_not_called()
                self.assertIn("Usage: /undo [--dry-run]", stream.getvalue())

    def test_undo_errors_remain_in_interactive_session(self):
        for error in (RuntimeError("file conflict"), PermissionError("policy block"), OSError("file busy")):
            with self.subTest(error=error):
                agent = Mock()
                agent.undo_last_file_change.side_effect = error
                stream = io.StringIO()
                quit_requested = asyncio.run(_handle_command("/undo", ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
                self.assertFalse(quit_requested)
                agent.undo_last_file_change.assert_called_once_with(dry_run=False)
                self.assertIn(str(error), stream.getvalue())

    def test_fork_command_accepts_title_and_explains_shared_workspace(self):
        for command, title in (("/fork", None), ("/fork approach B", "approach B")):
            with self.subTest(command=command):
                agent = Mock()
                agent.fork_session.return_value = {
                    "id": "child", "title": title or "Fork: original", "metadata": {"parent_session_id": "parent"},
                }
                stream = io.StringIO()
                asyncio.run(_handle_command(command, ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
                agent.fork_session.assert_called_once_with(title)
                self.assertIn("parent -> child", stream.getvalue())
                self.assertIn("workspace files are shared", stream.getvalue())

    def test_fork_failure_keeps_cli_alive(self):
        agent = Mock()
        agent.fork_session.side_effect = RuntimeError("Session changed in another consumer")
        stream = io.StringIO()
        result = asyncio.run(_handle_command("/fork", ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
        self.assertFalse(result)
        self.assertIn("another consumer", stream.getvalue())

    def test_compact_displays_durable_checkpoint_id(self):
        agent = Mock()
        agent.compact_context = AsyncMock(return_value={
            "before_messages": 20, "after_messages": 5, "before_tokens": 1000, "after_tokens": 200,
            "mode": "llm", "summary_tokens": 30, "checkpoint_id": 7,
        })
        stream = io.StringIO()
        asyncio.run(_handle_command("/compact", ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
        self.assertIn("checkpoint #7", stream.getvalue())
        self.assertIn("full raw history retained", stream.getvalue())

    def test_failed_or_cancelled_compaction_keeps_cli_alive(self):
        for error in (RuntimeError("checkpoint write failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                agent = Mock()
                agent.compact_context = AsyncMock(side_effect=error)
                stream = io.StringIO()
                result = asyncio.run(_handle_command("/compact", ConfigManager(), agent, renderer=CliRenderer(stream=stream, pretty=False)))
                self.assertFalse(result)
                self.assertTrue(stream.getvalue())
