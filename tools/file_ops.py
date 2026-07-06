"""文件操作工具集 — read_file, write_file, search_files, patch"""

import difflib
import json
import re
from pathlib import Path
from .context import ToolContext
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


def _is_sensitive_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return name == "auth.json" and ({".drudge", ".codex"} & parts)


def _safe_read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise
    except OSError:
        return None


def _diff_summary(path: Path, before: str | None, after: str | None, *, limit: int = 24) -> str:
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
        )
    )
    if not diff:
        return ""
    if len(diff) > limit:
        diff = diff[:limit] + ["... (truncated)"]
    return "\n".join(diff)


def _record_file_change(
    context: ToolContext | None,
    *,
    path: Path,
    operation: str,
    before_content: str | None,
    after_content: str | None,
    diff_summary: str,
) -> None:
    if context is None or context.record_file_change is None:
        return
    context.record_file_change({
        "path": str(path),
        "operation": operation,
        "before_content": before_content,
        "after_content": after_content,
        "diff_summary": diff_summary,
    })


def read_file_handler(
    path: str,
    offset: int = 1,
    limit: int = 500,
    context: ToolContext | None = None,
) -> str:
    """读取文件内容，返回带行号的内容"""
    try:
        filepath = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if _is_sensitive_path(filepath):
        return ToolResult.failure("Reading credential files is blocked", blocked=True)
    if not filepath.exists():
        return json.dumps({"error": f"File not found: {filepath}"})

    # 检测二进制文件
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            pass
    except UnicodeDecodeError:
        return json.dumps({"error": "Binary file detected. Cannot read as text."})

    lines = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            total = len(all_lines)
            start = max(0, offset - 1)
            end = min(start + limit, total)
            for i in range(start, end):
                lines.append(f"{i + 1}|{all_lines[i].rstrip()}")
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps({
        "content": "\n".join(lines),
        "total_lines": total,
        "shown_lines": len(lines),
        "offset": start + 1,
    }, ensure_ascii=False)


def write_file_handler(
    path: str,
    content: str,
    context: ToolContext | None = None,
) -> str:
    """写入文件内容（覆盖）"""
    try:
        filepath = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if _is_sensitive_path(filepath):
        return ToolResult.failure("Editing credential files is blocked", blocked=True)
    if context is None:
        return ToolResult.failure("ToolContext is required", blocked=True)
    allowed, reason = context.mutation_allowed(f"write_file {filepath}")
    if not allowed:
        return ToolResult.failure(reason or "Write blocked", blocked=True)
    try:
        before_content = _safe_read_text(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        diff_summary = _diff_summary(filepath, before_content, content)
        _record_file_change(
            context,
            path=filepath,
            operation="write_file",
            before_content=before_content,
            after_content=content,
            diff_summary=diff_summary,
        )
        return {
            "success": True,
            "path": str(filepath),
            "size": len(content),
            "changed": before_content != content,
            "diff_summary": diff_summary,
            "checkpoint_created": bool(context.record_file_change),
        }
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_files_handler(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    context: ToolContext | None = None,
) -> str:
    """搜索文件内容"""
    try:
        search_path = _resolve_path(path, context)
    except PermissionError as e:
        return ToolResult.failure(str(e), blocked=True)
    if _is_sensitive_path(search_path):
        return ToolResult.failure("Searching credential files is blocked", blocked=True)
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
) -> str:
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
    if _is_sensitive_path(filepath):
        return ToolResult.failure("Editing credential files is blocked", blocked=True)
    if not filepath.exists():
        return json.dumps({"error": f"File not found: {filepath}"})

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return json.dumps({"error": "Binary file detected"})

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
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        diff_summary = _diff_summary(filepath, content, new_content)
        _record_file_change(
            context,
            path=filepath,
            operation="patch",
            before_content=content,
            after_content=new_content,
            diff_summary=diff_summary,
        )
        return {
            "success": True,
            "path": str(filepath),
            "replacements": count,
            "diff_summary": diff_summary,
            "checkpoint_created": bool(context.record_file_change),
        }
    except Exception as e:
        return json.dumps({"error": str(e)})


def apply_patch_handler(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    context: ToolContext | None = None,
) -> str:
    """First-class patch tool for targeted source edits."""
    return patch_handler(path, old_string, new_string, replace_all, context)


def file_check() -> bool:
    return True


# 注册工具
registry.register(
    name="read_file",
    description="Read a text file with line numbers. Output format: 'LINE_NUM|CONTENT'. "
    "Use offset and limit for large files. NOTE: Cannot read images or binary files.",
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
    description="Write content to a file, overwriting existing content. Creates parent directories automatically.",
    parameters={
        "path": {"type": str, "description": "Path to the file to write"},
        "content": {"type": str, "description": "Complete content to write to the file"},
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
    },
    handler=apply_patch_handler,
    toolset="file",
    check_fn=file_check,
    required=["path", "old_string", "new_string"],
    risk_fn=_file_mutation_risk,
)
