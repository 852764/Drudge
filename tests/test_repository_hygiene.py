import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryHygieneTests(unittest.TestCase):
    def test_config_has_no_embedded_openai_style_secret(self):
        source = (ROOT / "config.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{16,}", source))

    def test_python_caches_are_ignored_and_untracked(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("__pycache__/", ignore)
        self.assertIn("*.py[cod]", ignore)

        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "ls-files",
                "*.pyc",
                "**/__pycache__/**",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
        if result.returncode != 0:
            self.skipTest("git ls-files unavailable")
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()

