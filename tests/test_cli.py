from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from config import ConfigManager
from main import parse_args


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
