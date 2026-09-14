"""文件操作工具集 — read_file, write_file, search_files, patch"""

import json
import re
import os
from pathlib import Path
from .context import ToolContext
from .file_io import FileConflictError, atomic_write, check_expected_hash, read_bytes, sha256
from .file_io import text_diff as _diff_summary
from .registry import registry
from .result import ToolResult, normalize_tool_payload, limit_tool_result
from .risk import RiskLevel, ToolRisk
from .repository import RepositoryWalker


def _file_mutation_risk(args: dict, context: ToolContext) -> ToolRisk:
    path = str(args.get("path", "(unknown path)"))
    return ToolRisk(RiskLevel.MEDIUM, "Modify a workspace file", path)


def _resolve_path(
    path: str,
    context: ToolContext | None,
) -> Path:
    """解析路径，支持 ~/ 和相对路径"""
    if context is None:
        raise PermissionError("ToolContext is required")
    return context.resolve_path(path)


def _record_file_change(
    context: ToolContext | None,
    *,
    path: Path,
    operation: str,
    before_content: str | None,
    after_content: str | None,
    diff_summary: str,
) -> bool:
    if context is None or context.record_file_change is None:
        return False
    context.record_file_change({
        "path": str(path),
        "operation": operation,
        "before_content": before_content,
        "after_content": after_content,
        "diff_summary": diff_summary,
    })
    return True


def _commit_text_change(
    path: Path,
    before: bytes | None,
    content: str,
    context: ToolContext,
    *,
    operation: str,
) -> dict:
    after = content.encode("utf-8")
    before_text = before.decode("utf-8") if before is not None else None
    changed = before != after
    diff_summary = _diff_summary(path, before_text, content)
    result = {
        "success": True,
        "path": str(path),
        "changed": changed,
        "sha256": sha256(after),
        "before_sha256": sha256(before),
        "diff_summary": diff_summary,
        "checkpoint_created": False,
    }
    if not changed:
        return result
    atomic_write(path, after, expected=before)
    try:
        result["checkpoint_created"] = _record_file_change(
            context, path=path, operation=operation,
            before_content=before_text, after_content=content,
            diff_summary=diff_summary,
        )
    except Exception as exc:
        # The file has already changed. Do not invite a repeat mutation.
        result["warnings"] = [f"File saved, but checkpoint recording failed: {exc}"]
    return result


def _edit_failure(exc: Exception) -> ToolResult:
    if isinstance(exc, FileConflictError):
        return ToolResult.failure(
            str(exc), conflict=True,
            expected_sha256=exc.expected_sha256,
            actual_sha256=exc.actual_sha256,
        )
    return ToolResult.failure(str(exc), blocked=isinstance(exc, PermissionError))


def read_file_handler(
    path: str,
    offset: int = 1,
    limit: int = 500,
    context: ToolContext | None = None,
) -> str | ToolResult:
    """读取文件内容，返回带行号的内容"""
    try:
        filepath = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    try:
        if offset < 1 or limit < 1:
            raise ValueError("offset and limit must be positive integers")
        data = read_bytes(filepath)
        if data is None:
            return ToolResult.failure(f"File not found: {filepath}")
        all_lines = data.decode("utf-8").splitlines()
        total = len(all_lines)
        start = offset - 1
        end = min(start + limit, total)
        lines = [f"{i + 1}|{all_lines[i]}" for i in range(start, end)]
    except Exception as e:
        return _edit_failure(e)

    return json.dumps({
        "content": "\n".join(lines),
        "total_lines": total,
        "shown_lines": len(lines),
        "offset": start + 1,
        "sha256": sha256(data),
    }, ensure_ascii=False)


def read_files_handler(
    paths: list[str], offset: int = 1, limit: int = 200,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read a bounded group in one model round, preserving per-file outcomes."""
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    if not isinstance(paths, list) or not 1 <= len(paths) <= 8:
        return ToolResult.failure("paths must contain between 1 and 8 file paths")
    if any(not isinstance(path, str) or not path.strip() or len(path) > 4096 for path in paths):
        return ToolResult.failure("Each path must be a nonempty string of at most 4096 characters")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        return ToolResult.failure("offset must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        return ToolResult.failure("limit must be between 1 and 500 lines per file")
    results = []
    for path in paths:
        payload = read_file_handler(path, offset=offset, limit=limit, context=context)
        save = None
        if context.save_tool_output is not None:
            def save(content):
                return context.save_tool_output(content, tool_name="read_files.item", kind="json")
        bounded = limit_tool_result(payload, max_chars=6000, save_output=save)
        results.append({"path": path, **normalize_tool_payload(bounded)})
    failed = sum(not result["ok"] for result in results)
    content = json.dumps(results, ensure_ascii=False)
    metadata = {"files_read": len(results) - failed, "files_failed": failed}
    if failed:
        return ToolResult.failure(
            f"{failed} of {len(results)} files could not be read; inspect each result.",
            content=content, **metadata,
        )
    return ToolResult.success(content, **metadata)


def write_file_handler(
    path: str,
    content: str,
    context: ToolContext | None = None,
    expected_sha256: str | None = None,
) -> dict | ToolResult:
    """写入文件内容（覆盖）"""
    try:
        filepath = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    allowed, reason = context.mutation_allowed(f"write_file {filepath}")
    if not allowed:
        return ToolResult.failure(reason or "Write blocked", blocked=True)
    try:
        before = read_bytes(filepath)
        check_expected_hash(filepath, before, expected_sha256)
        result = _commit_text_change(filepath, before, content, context, operation="write_file")
        result["size"] = len(content)
        return result
    except Exception as e:
        return _edit_failure(e)


def search_files_handler(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    context: ToolContext | None = None,
    include_hidden: bool = False,
    literal: bool = False,
    case_sensitive: bool = False,
) -> str | ToolResult:
    """Search UTF-8 text with explicit scope, I/O budgets and completeness."""
    if not isinstance(pattern, str) or not 1 <= len(pattern) <= 4096:
        return ToolResult.failure("pattern must contain between 1 and 4096 characters")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        return ToolResult.failure("limit must be an integer between 1 and 1000")
    if any(not isinstance(value, bool) for value in (include_hidden, literal, case_sensitive)):
        return ToolResult.failure("include_hidden, literal and case_sensitive must be booleans")
    if file_glob is not None and (
        not isinstance(file_glob, str) or not 1 <= len(file_glob) <= 2048
        or len(file_glob.replace("\\", "/").split("/")) > 64
        or Path(file_glob).is_absolute() or ".." in file_glob.replace("\\", "/").split("/")
    ):
        return ToolResult.failure("file_glob must be a bounded search-root-relative glob")
    try:
        search_path = _resolve_path(path, context)
        regex = re.compile(re.escape(pattern) if literal else pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return ToolResult.failure(f"Invalid regex: {exc}")
    except (OSError, ValueError, TypeError) as exc:
        return _edit_failure(exc)
    if not search_path.exists():
        return ToolResult.failure(f"Path not found: {search_path}")
    if not search_path.is_file() and not search_path.is_dir():
        return ToolResult.failure("Search path must be an ordinary file or directory")

    matches = []
    walker = RepositoryWalker(
        search_path if search_path.is_dir() else search_path.parent,
        anchor=context.workspace, authorize=context.resolve_path,
        include_hidden=include_hidden, max_entries=context.search_max_entries,
    )
    reasons = walker.stats.incomplete_reasons
    skipped = walker.stats.skipped
    files_scanned = bytes_read = 0
    candidates = (search_path,) if search_path.is_file() else walker.files(file_glob)
    for filepath in candidates:
        if files_scanned >= context.search_max_files:
            reasons.add("file_limit")
            break
        try:
            # Reauthorize immediately before opening, independently of discovery.
            filepath = _resolve_path(str(filepath), context)
            info = filepath.stat()
            if not filepath.is_file():
                skipped["special"] += 1
                continue
            if info.st_size > context.search_max_file_bytes:
                skipped["too_large"] += 1
                reasons.add("file_size_limit")
                continue
            remaining = context.search_max_total_bytes - bytes_read
            if info.st_size > remaining:
                reasons.add("byte_limit")
                break
            files_scanned += 1
            cap = min(context.search_max_file_bytes, remaining)
            with open(filepath, "rb") as stream:
                data = stream.read(cap)
                current_size = os.fstat(stream.fileno()).st_size
            bytes_read += len(data)
            if current_size > len(data):
                skipped["changed_during_read"] += 1
                reasons.add("file_changed")
                continue
            if b"\0" in data:
                skipped["binary"] += 1
                continue
            try:
                source = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                skipped["non_utf8"] += 1
                continue
            for line_no, line in enumerate(source.splitlines(), 1):
                match = regex.search(line)
                if match is None:
                    continue
                # Look ahead by one matching line: exactly limit hits at EOF
                # is a complete result, rather than a spurious truncation.
                if len(matches) == limit:
                    reasons.add("match_limit")
                    break
                start = max(0, match.start() - 80)
                matches.append({
                    "file": str(filepath), "line": line_no,
                    "column": match.start() + 1,
                    "content": line[start:start + 200],
                    "content_start_column": start + 1,
                })
            if "match_limit" in reasons:
                break
        except (OSError, ValueError, RuntimeError) as exc:
            skipped["blocked" if isinstance(exc, PermissionError) else "unreadable"] += 1
            reasons.add("file_unreadable")
            continue

    return json.dumps({
        "matches": matches,
        "total": len(matches),
        "truncated": bool(reasons),
        "complete": not reasons,
        "incomplete_reasons": sorted(reasons),
        "files_scanned": files_scanned, "bytes_read": bytes_read,
        "entries_seen": walker.stats.entries_seen,
        "skipped": dict(sorted(skipped.items())),
        "scope": {"path": str(search_path), "file_glob": file_glob, "include_hidden": include_hidden,
                  "ignore_policy": "workspace .gitignore (explicit files bypass discovery filters)",
                  "encoding": "utf-8", "case_sensitive": case_sensitive, "literal": literal},
        "limits": {"files": context.search_max_files, "file_bytes": context.search_max_file_bytes,
                   "total_bytes": context.search_max_total_bytes, "entries": context.search_max_entries},
    }, ensure_ascii=False)


def patch_handler(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    context: ToolContext | None = None,
    expected_sha256: str | None = None,
) -> str | dict | ToolResult:
    """在文件中查找替换"""
    try:
        filepath = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    allowed, reason = context.mutation_allowed(f"patch {filepath}")
    if not allowed:
        return ToolResult.failure(reason or "Patch blocked", blocked=True)

    try:
        if not old_string:
            raise ValueError("old_string must not be empty")
        before = read_bytes(filepath)
        check_expected_hash(filepath, before, expected_sha256)
        if before is None:
            return ToolResult.failure(f"File not found: {filepath}")
        content = before.decode("utf-8")
    except Exception as exc:
        return _edit_failure(exc)

    # Adapt LF model text to uniformly CRLF sources, not mixed-line-ending files.
    without_crlf = content.replace("\r\n", "")
    if "\r\n" in content and "\n" not in without_crlf and "\r" not in without_crlf:
        if "\r" not in old_string:
            old_string = old_string.replace("\n", "\r\n")
        if "\r" not in new_string:
            new_string = new_string.replace("\n", "\r\n")

    count = content.count(old_string)
    if count == 0:
        return json.dumps({"error": "old_string not found in file"})
    if count > 1 and not replace_all:
        return json.dumps({
            "error": f"old_string found {count} times. Set replace_all=true to replace all, or provide more context for uniqueness.",
            "occurrences": count,
        })

    new_content = content.replace(old_string, new_string)
    try:
        result = _commit_text_change(filepath, before, new_content, context, operation="patch")
        result["replacements"] = count
        return result
    except Exception as e:
        return _edit_failure(e)


def apply_patch_handler(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    context: ToolContext | None = None,
    expected_sha256: str | None = None,
) -> str | dict | ToolResult:
    """First-class patch tool for targeted source edits."""
    return patch_handler(path, old_string, new_string, replace_all, context, expected_sha256)


def file_check() -> bool:
    return True


# 注册工具
registry.register(
    name="read_file",
    description="Read a text file with line numbers. Output format: 'LINE_NUM|CONTENT'. "
    "Use offset and limit for large files. Includes whole-file sha256 for guarded edits. "
    "NOTE: Cannot read images or binary files.",
    parameters={
        "path": {"type": str, "description": "Path to the file (absolute, relative, or ~/path)"},
        "offset": {"type": int, "description": "Line number to start from (1-indexed, default: 1)"},
        "limit": {"type": int, "description": "Maximum lines to read (default: 500)"},
    },
    handler=read_file_handler,
    toolset="file",
    check_fn=file_check,
    required=["path"],
)

registry.register(
    name="read_files",
    description="Read up to 8 independent text files in one call. Returns per-file line numbers, whole-file SHA-256 and errors. "
    "Prefer this to separate read_file calls when inspecting several files. Large results have bounded previews and output receipts.",
    parameters={
        "paths": {"type": list, "items": {"type": "string"}, "minItems": 1, "maxItems": 8,
                  "description": "File paths to read; each is checked against the host workspace and credential policy"},
        "offset": {"type": int, "minimum": 1, "description": "First line in each file (default: 1)"},
        "limit": {"type": int, "minimum": 1, "maximum": 500, "description": "Lines per file (default: 200, max: 500)"},
    },
    handler=read_files_handler, toolset="file", check_fn=file_check, required=["paths"],
)

registry.register(
    name="write_file",
    description="Atomically write exact UTF-8 content. Creates parent directories automatically. "
    "Pass expected_sha256 from read_file to detect stale edits, or 'missing' to create a new file only.",
    parameters={
        "path": {"type": str, "description": "Path to the file to write"},
        "content": {"type": str, "description": "Complete content to write to the file"},
        "expected_sha256": {"type": str, "description": "Expected whole-file SHA-256 from read_file, or 'missing' (optional)"},
    },
    handler=write_file_handler,
    toolset="file",
    check_fn=file_check,
    required=["path", "content"],
    risk_fn=_file_mutation_risk,
)

registry.register(
    name="search_files",
    description="Search UTF-8 files using regex or literal text. Respects workspace/nested .gitignore and reports skipped files and incomplete scans. "
    "Returns paths, one-based lines/columns, and match-centered snippets. Narrow path/glob when a host budget is reached.",
    parameters={
        "pattern": {"type": str, "description": "Regex pattern to search for"},
        "path": {"type": str, "description": "Directory or file to search in (default: current directory)"},
        "file_glob": {"type": str, "description": "Optional case-sensitive glob: '*.py' at any depth, or root-relative 'src/**/*.py'"},
        "limit": {"type": int, "minimum": 1, "maximum": 1000, "description": "Maximum matching lines (default: 50, max: 1000)"},
        "include_hidden": {"type": bool, "description": "Include ordinary hidden paths, still excluding private/runtime directories (default: false)"},
        "literal": {"type": bool, "description": "Search exact text instead of regex (default: false)"},
        "case_sensitive": {"type": bool, "description": "Match case exactly (default: false)"},
    },
    handler=search_files_handler,
    toolset="file",
    check_fn=file_check,
    required=["pattern"],
)

registry.register(
    name="patch",
    description="Find and replace text in a file. Use for targeted edits.",
    parameters={
        "path": {"type": str, "description": "File path to edit"},
        "old_string": {"type": str, "description": "Exact text to find"},
        "new_string": {"type": str, "description": "Replacement text"},
        "replace_all": {"type": bool, "description": "Replace all occurrences (default: false)"},
        "expected_sha256": {"type": str, "description": "Expected whole-file SHA-256 from read_file (optional)"},
    },
    handler=patch_handler,
    toolset="file",
    check_fn=file_check,
    required=["path", "old_string", "new_string"],
    risk_fn=_file_mutation_risk,
)

registry.register(
    name="apply_patch",
    description="Apply a targeted text patch to one file by replacing an exact old_string with new_string. Prefer this for code edits.",
    parameters={
        "path": {"type": str, "description": "File path to edit"},
        "old_string": {"type": str, "description": "Exact text to replace"},
        "new_string": {"type": str, "description": "Replacement text"},
        "replace_all": {"type": bool, "description": "Replace all occurrences (default: false)"},
        "expected_sha256": {"type": str, "description": "Expected whole-file SHA-256 from read_file (optional)"},
    },
    handler=apply_patch_handler,
    toolset="file",
    check_fn=file_check,
    required=["path", "old_string", "new_string"],
    risk_fn=_file_mutation_risk,
)
