"""文件操作工具集 — read_file, write_file, search_files, patch"""

import json
import re
from pathlib import Path
from .context import ToolContext
from .file_io import FileConflictError, atomic_write, check_expected_hash, read_bytes, sha256
from .file_io import text_diff as _diff_summary
from .registry import registry
from .result import ToolResult
from .risk import RiskLevel, ToolRisk


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
) -> str | ToolResult:
    """搜索文件内容"""
    try:
        search_path = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if not search_path.exists():
        return json.dumps({"error": f"Path not found: {search_path}"})

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}"})

    matches = []
    files_to_search = []

    if search_path.is_file():
        files_to_search = [search_path]
    else:
        glob_pattern = file_glob or "*"
        for f in search_path.rglob(glob_pattern):
            if f.is_file() and not any(p.startswith(".") for p in f.parts):
                files_to_search.append(f)

    for filepath in files_to_search[:200]:
        if len(matches) >= limit:
            break
        try:
            # Authorize discovered symlinks as well as the search directory.
            filepath = _resolve_path(str(filepath), context)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append({
                            "file": str(filepath),
                            "line": line_no,
                            "content": line.strip()[:200],
                        })
                        if len(matches) >= limit:
                            break
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

    return json.dumps({
        "matches": matches,
        "total": len(matches),
        "truncated": len(matches) >= limit,
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
    description="Search file contents using regex. Returns file paths, line numbers, and matching content.",
    parameters={
        "pattern": {"type": str, "description": "Regex pattern to search for"},
        "path": {"type": str, "description": "Directory or file to search in (default: current directory)"},
        "file_glob": {"type": str, "description": "Optional glob filter (e.g., '*.py')"},
        "limit": {"type": int, "description": "Maximum results (default: 50)"},
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
