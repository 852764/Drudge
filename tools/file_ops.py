"""文件操作工具集 — read_file, write_file, search_files, patch"""

import json
import re
from pathlib import Path
from .registry import registry


def _resolve_path(path: str, workspace: str | None = None) -> Path:
    """解析路径，支持 ~/ 和相对路径"""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = Path(workspace) / p
    return p.resolve()


def read_file_handler(
    path: str,
    offset: int = 1,
    limit: int = 500,
    workspace: str | None = None,
) -> str:
    """读取文件内容，返回带行号的内容"""
    filepath = _resolve_path(path, workspace)
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
    workspace: str | None = None,
) -> str:
    """写入文件内容（覆盖）"""
    filepath = _resolve_path(path, workspace)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({
            "success": True,
            "path": str(filepath),
            "size": len(content),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_files_handler(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    workspace: str | None = None,
) -> str:
    """搜索文件内容"""
    search_path = _resolve_path(path, workspace)
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
    workspace: str | None = None,
) -> str:
    """在文件中查找替换"""
    filepath = _resolve_path(path, workspace)
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
        return json.dumps({
            "success": True,
            "path": str(filepath),
            "replacements": count,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


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
)
