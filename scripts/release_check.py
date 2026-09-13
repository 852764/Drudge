"""Offline release gate: source hygiene, regressions, archives and installed CLI.

Run from a checkout with runtime dependencies, setuptools and wheel installed.
No credentials, network calls, publishing, commits or pushes are performed.
Reports/artifacts live in unique ignored build/release-check subdirectories.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/").lower())
    return (
        path.is_absolute() or ".." in path.parts or ":" in name
        or bool(set(path.parts) & {".drudge", ".codex", ".git", ".venv", "__pycache__"})
        or path.name in {"auth.json", "config.local.yaml", "config.local.yml", ".env"}
        or (path.name.startswith(".env.") and path.name != ".env.example")
        or path.name.endswith((".pyc", ".pyo", ".db", ".db-shm", ".db-wal"))
    )


def check_content(name: str, data: bytes) -> None:
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        # Report only the filename, never the matching secret.
        raise ValueError(f"Possible embedded credential in {name}")


def source_hygiene(root: Path = ROOT) -> int:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True, timeout=15,
    )
    names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    for name in names:
        if forbidden_path(name):
            raise ValueError(f"Private/generated path is included in source: {name}")
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Release source must not contain symlinks: {name}")
        if not path.exists():  # Pending tracked deletion.
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or forbidden_path(resolved.relative_to(root.resolve()).as_posix()):
            raise ValueError(f"Source path resolves outside release scope: {name}")
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Unexpected oversized source file: {name}")
        check_content(name, path.read_bytes())
    return len(names)


def check_archive(path: Path) -> dict:
    names = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if forbidden_path(info.filename) or info.file_size > 10 * 1024 * 1024:
                    raise ValueError(f"Unexpected wheel entry: {info.filename}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Wheel symlink: {info.filename}")
                check_content(info.filename, archive.read(info))
                names.append(info.filename)
        required = {"main.py", "config.py", "agent/drudge_agent.py", "tools/_windows_job.py", "tools/plan.py"}
        if not required.issubset(names):
            raise ValueError(f"Wheel missing required modules: {sorted(required - set(names))}")
    else:
        with tarfile.open(path) as archive:
            for info in archive:
                if forbidden_path(info.name) or info.issym() or info.islnk() or info.size > 10 * 1024 * 1024:
                    raise ValueError(f"Unexpected source archive entry: {info.name}")
                if info.isfile():
                    with archive.extractfile(info) as stream:
                        check_content(info.name, stream.read())
                elif not info.isdir():
                    raise ValueError(f"Unexpected archive entry type: {info.name}")
                names.append(info.name)
        if not any(name.endswith("/tests/test_persistent_plans.py") for name in names):
            raise ValueError("Source archive is missing regression tests")
    return {"file": path.name, "entries": len(names), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    base = ROOT / "build" / "release-check"
    if not base.resolve().is_relative_to(ROOT) or base.is_symlink():
        raise ValueError("Release output must stay within this workspace")
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    report = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": sys.platform,
        "scope": "offline beta release gate; not a live-model benchmark or deployment",
        "ok": False, "steps": [],
    }

    def run(name: str, command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> str:
        print(f"{name} ...", flush=True)
        started = time.monotonic()
        log = output / f"{name}.log"
        with log.open("wb") as stream:
            completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        report["steps"].append({"name": name, "exit_code": completed.returncode, "seconds": round(time.monotonic() - started, 3), "log": log.name})
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit {completed.returncode}; inspect {log}")
        return log.read_text(encoding="utf-8", errors="replace")

    try:
        report["source_files_checked"] = source_hygiene()
        report["commit"] = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip()
        report["dirty"] = bool(subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"], cwd=ROOT, timeout=15).strip())
        tests = run("unittest", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        match = re.search(r"Ran (\d+) tests? in", tests)
        report["tests_run"] = int(match.group(1)) if match else None
        skipped = re.search(r"skipped=(\d+)", tests)
        report["tests_skipped"] = int(skipped.group(1)) if skipped else 0
        if not report["tests_run"]:
            raise RuntimeError("No regression tests were executed")
        run("compileall", [sys.executable, "-m", "compileall", "-q", "agent", "config.py", "main.py", "prompt", "tools", "tests", "scripts"])
        run("diff-check", ["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--check"])
        artifacts = output / "artifacts"
        artifacts.mkdir()
        # Each backend hook gets a fresh interpreter; setuptools mutates argv.
        run("sdist", [sys.executable, "-c", "import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])", str(artifacts)])
        run("wheel", [sys.executable, "-c", "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])", str(artifacts)])
        report["artifacts"] = [check_archive(path) for path in sorted(artifacts.iterdir())]
        wheels = list(artifacts.glob("*.whl"))
        if len(wheels) != 1 or len(list(artifacts.glob("*.tar.gz"))) != 1:
            raise RuntimeError("Expected exactly one wheel and one source archive")
        environment = output / "venv"
        run("venv", [sys.executable, "-m", "venv", "--system-site-packages", str(environment)])
        binary_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")
        # An embedding host may inject a separate venv's site-packages, which
        # --system-site-packages alone does not inherit. Reuse only dependency
        # directories, never the checkout or arbitrary PYTHONPATH entries.
        host_sites = sorted({str(Path(item).resolve()) for item in sys.path if item
                             and Path(item).name in {"site-packages", "dist-packages"}
                             and Path(item).is_dir()})
        purelib = Path(run("dependency-path", [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], cwd=output).strip())
        if not purelib.resolve().is_relative_to(environment.resolve()):
            raise ValueError("Dependency shim must stay within the new virtual environment")
        # ASCII JSON-escaped paths also work with Python 3.10's locale .pth reader.
        (purelib / "drudge_release_dependencies.pth").write_text(
            "import sys; sys.path.extend(" + json.dumps(host_sites, ensure_ascii=True) + ")\n", encoding="ascii",
        )
        run("install", [str(python), "-m", "pip", "--isolated", "install", "--no-deps", "--no-index", "--force-reinstall", str(wheels[0])], cwd=output)
        # Assert all Drudge modules came from the new venv, not a host install.
        smoke = output / "installed_smoke.py"
        shutil.copyfile(ROOT / "scripts" / "installed_smoke.py", smoke)
        run("installed-smoke", [str(python), "-I", str(smoke)], cwd=output)
        cli = binary_dir / ("drudge.exe" if os.name == "nt" else "drudge")
        run("cli-help", [str(cli), "--help"], cwd=output)
        run("cli-version", [str(cli), "--version"], cwd=output)
        report["ok"] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
        print(f"Release gate failed: {exc}", file=sys.stderr)
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
