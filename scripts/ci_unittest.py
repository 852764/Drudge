"""Run the offline suite with durable, visible CI diagnostics."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_check import failure_excerpt


LOG = ROOT / "build" / "ci" / "unittest.log"


def _command_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode:
        excerpt = failure_excerpt(completed.stdout, max_lines=24)
        for line in excerpt.splitlines():
            print(f"::error title=Cross-platform unittest::{_command_escape(line)}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
