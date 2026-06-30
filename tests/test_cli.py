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
