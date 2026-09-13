from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from config import tomllib
from scripts.release_check import check_archive, check_content, forbidden_path, source_hygiene


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCheckTests(unittest.TestCase):
    def test_version_matches_package_and_beta_is_explicit(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        declared = next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "VERSION" for target in node.targets))
        self.assertEqual(metadata["project"]["version"], declared)
        self.assertIn("Development Status :: 4 - Beta", metadata["project"]["classifiers"])

    def test_runtime_and_release_scripts_parse_as_python310(self):
        paths = [ROOT / name for name in ("main.py", "config.py", "utils.py")]
        for name in ("agent", "prompt", "tools", "scripts"):
            paths.extend((ROOT / name).rglob("*.py"))
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))

    def test_private_paths_and_traversal_rejected(self):
        for name in (".drudge/auth.json", "x/.CODEX/auth.json", "../secret", "C:/secret", "/secret", "pkg/.env", "pkg/.env.production", "config.local.yaml", "pkg/__pycache__/x.pyc", "data.db-wal"):
            self.assertTrue(forbidden_path(name), name)
        for name in ("agent/storage.py", ".env.example", "docs/CODEX_OAUTH.md"):
            self.assertFalse(forbidden_path(name), name)

    def test_sensitive_source_name_is_rejected_before_any_read(self):
        result = subprocess.CompletedProcess([], 0, stdout=b".drudge/auth.json\0")
        with patch("scripts.release_check.subprocess.run", return_value=result), patch.object(Path, "read_bytes") as read:
            with self.assertRaisesRegex(ValueError, "Private/generated"):
                source_hygiene(ROOT)
        read.assert_not_called()

    def test_secret_detection_reports_filename_not_secret(self):
        token = b"sk-" + b"x" * 40
        with self.assertRaises(ValueError) as raised:
            check_content("fixture.txt", token)
        self.assertIn("fixture.txt", str(raised.exception))
        self.assertNotIn(token.decode(), str(raised.exception))

    def test_archive_sensitive_entries_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(".codex/auth.json", "fixture")
            with patch.object(zipfile.ZipFile, "read") as read:
                with self.assertRaisesRegex(ValueError, "Unexpected wheel entry"):
                    check_archive(path)
            read.assert_not_called()

    def test_archive_missing_runtime_module_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("main.py", "# fixture")
            with self.assertRaisesRegex(ValueError, "missing required modules"):
                check_archive(path)

    def test_ci_covers_minimum_python_windows_and_unix(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        for required in ("'3.10'", "windows-latest", "ubuntu-latest", "macos-latest", "scripts/release_check.py", "contents: read"):
            self.assertIn(required, workflow)


if __name__ == "__main__":
    unittest.main()
