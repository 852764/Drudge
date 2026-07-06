"""SQLite-backed conversation storage for Drudge sessions."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ConversationStore:
    """Persist sessions, messages, and tool calls to a local SQLite database."""

    def __init__(self, db_path: str):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    turn INTEGER NOT NULL DEFAULT 0,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    turn INTEGER NOT NULL DEFAULT 0,
                    model TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TEXT
                );

                CREATE TABLE IF NOT EXISTS file_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    run_id TEXT,
                    path TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    before_content TEXT,
                    after_content TEXT,
                    diff_summary TEXT NOT NULL DEFAULT '',
                    undone INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    undone_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);

                CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id
                ON tool_calls(session_id, id);

                CREATE INDEX IF NOT EXISTS idx_runs_session_id
                ON runs(session_id, started_at);

                CREATE INDEX IF NOT EXISTS idx_run_events_run_id
                ON run_events(run_id, id);

                CREATE INDEX IF NOT EXISTS idx_tasks_session_id
                ON tasks(session_id, status, id);

                CREATE INDEX IF NOT EXISTS idx_memories_scope_namespace
                ON memories(scope, namespace, pinned, updated_at);

                CREATE INDEX IF NOT EXISTS idx_file_revisions_session_id
                ON file_revisions(session_id, undone, id);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "metadata_json" not in columns:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute("PRAGMA journal_mode = WAL")

    def create_session(
        self,
        title: str,
        model: str,
        cwd: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:12]
        clean_title = " ".join(title.split())[:80] or "Untitled session"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, title, model, cwd, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    clean_title,
                    model,
                    cwd or os.getcwd(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return session_id

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, model, cwd, metadata_json, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        return item

    def update_session_metadata(self, session_id: str, updates: dict[str, Any]) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.update(updates)
            conn.execute(
                """
                UPDATE sessions
                SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), session_id),
            )

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        *,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_call_id, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, tool_call_id, metadata_json),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    def append_tool_call(
        self,
        session_id: str,
        turn: int,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (session_id, turn, name, arguments_json, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, turn, name, json.dumps(arguments, ensure_ascii=False), result),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, model, cwd, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(
        self,
        session_id: str,
        limit: int | None = 100,
    ) -> list[dict[str, Any]]:
        limit_clause = "LIMIT ?" if limit is not None else ""
        parameters: tuple[Any, ...] = (session_id, limit) if limit is not None else (session_id,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, role, content, tool_call_id, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                {limit_clause}
                """,
                parameters,
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            messages.append(item)
        return messages

    def get_max_turn(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn), 0) AS max_turn FROM tool_calls WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["max_turn"] if row else 0)

    def start_run(
        self,
        session_id: str | None,
        prompt: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (id, session_id, prompt, model, status, metadata_json)
                VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    prompt,
                    model,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                ),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            merged_metadata = json.loads(row["metadata_json"] or "{}") if row else {}
            merged_metadata.update(metadata or {})
            conn.execute(
                """
                UPDATE runs
                SET status = ?, error = ?, metadata_json = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    error,
                    json.dumps(merged_metadata, ensure_ascii=False, default=str),
                    run_id,
                ),
            )

    def append_run_event(
        self,
        run_id: str,
        kind: str,
        *,
        turn: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_events (run_id, kind, turn, detail_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    kind,
                    int(turn),
                    json.dumps(detail or {}, ensure_ascii=False, default=str),
                ),
            )

    def append_model_call(
        self,
        run_id: str,
        *,
        turn: int,
        model: str,
        purpose: str,
        total_tokens: int,
        latency_ms: int,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_calls
                    (run_id, turn, model, purpose, total_tokens, latency_ms, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    int(turn),
                    model,
                    purpose,
                    int(total_tokens),
                    int(latency_ms),
                    status,
                    error,
                ),
            )

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        where = "WHERE session_id = ?" if session_id else ""
        params: tuple[Any, ...] = (session_id, limit) if session_id else (limit,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, prompt, model, status, error, metadata_json,
                       started_at, completed_at
                FROM runs
                {where}
                ORDER BY started_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            result.append(item)
        return result

    def get_run_trace(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            run = conn.execute(
                """
                SELECT id, session_id, prompt, model, status, error, metadata_json,
                       started_at, completed_at
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                return None
            events = conn.execute(
                """
                SELECT id, kind, turn, detail_json, created_at
                FROM run_events WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
            calls = conn.execute(
                """
                SELECT id, turn, model, purpose, total_tokens, latency_ms, status, error, created_at
                FROM model_calls WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        result = dict(run)
        result["metadata"] = json.loads(result.pop("metadata_json") or "{}")
        result["events"] = []
        for row in events:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json") or "{}")
            result["events"].append(item)
        result["model_calls"] = [dict(row) for row in calls]
        return result

    def create_task(self, session_id: str, title: str, details: str = "") -> dict[str, Any]:
        clean_title = " ".join(title.split()).strip()
        if not clean_title:
            raise ValueError("Task title cannot be empty")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks (session_id, title, details)
                VALUES (?, ?, ?)
                """,
                (session_id, clean_title[:240], details.strip()[:8000]),
            )
            task_id = int(cursor.lastrowid)
        return self.get_task(session_id, task_id)

    def get_task(self, session_id: str, task_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, title, details, status, created_at, updated_at, completed_at
                FROM tasks WHERE session_id = ? AND id = ?
                """,
                (session_id, int(task_id)),
            ).fetchone()
        if row is None:
            raise KeyError(f"Task not found: {task_id}")
        return dict(row)

    def list_tasks(
        self,
        session_id: str,
        *,
        include_closed: bool = False,
    ) -> list[dict[str, Any]]:
        condition = "" if include_closed else "AND status NOT IN ('completed', 'cancelled')"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, title, details, status, created_at, updated_at, completed_at
                FROM tasks
                WHERE session_id = ? {condition}
                ORDER BY
                    CASE status WHEN 'in_progress' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                    id
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_task(self, session_id: str, task_id: int, status: str) -> dict[str, Any]:
        allowed = {"pending", "in_progress", "completed", "cancelled"}
        if status not in allowed:
            raise ValueError(f"Invalid task status: {status}")
        completed = "CURRENT_TIMESTAMP" if status in {"completed", "cancelled"} else "NULL"
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP, completed_at = {completed}
                WHERE session_id = ? AND id = ?
                """,
                (status, session_id, int(task_id)),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Task not found: {task_id}")
        return self.get_task(session_id, task_id)

    def create_memory(
        self,
        scope: str,
        namespace: str,
        content: str,
        *,
        title: str = "",
        tags: list[str] | None = None,
        pinned: bool = False,
    ) -> dict[str, Any]:
        clean_scope = str(scope).strip().lower()
        if clean_scope not in {"project", "user"}:
            raise ValueError(f"Invalid memory scope: {scope}")
        clean_content = str(content).strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty")
        clean_tags = [str(tag).strip()[:64] for tag in (tags or []) if str(tag).strip()][:20]
        clean_title = " ".join(str(title).split()).strip()[:240]
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (scope, namespace, title, content, tags_json, pinned)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_scope,
                    str(namespace),
                    clean_title,
                    clean_content[:16000],
                    json.dumps(clean_tags, ensure_ascii=False),
                    1 if pinned else 0,
                ),
            )
            memory_id = int(cursor.lastrowid)
        return self.get_memory(memory_id)

    def get_memory(self, memory_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, scope, namespace, title, content, tags_json, pinned,
                       created_at, updated_at, last_used_at
                FROM memories WHERE id = ?
                """,
                (int(memory_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        item["pinned"] = bool(item.get("pinned"))
        return item

    def list_memories(
        self,
        *,
        scope: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            params.append(str(scope).strip().lower())
        if namespace:
            clauses.append("namespace = ?")
            params.append(str(namespace))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, scope, namespace, title, content, tags_json, pinned,
                       created_at, updated_at, last_used_at
                FROM memories
                {where}
                ORDER BY pinned DESC,
                         COALESCE(last_used_at, updated_at) DESC,
                         id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.pop("tags_json") or "[]")
            item["pinned"] = bool(item.get("pinned"))
            result.append(item)
        return result

    def update_memory(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        updates: list[str] = []
        params: list[Any] = []
        if title is not None:
            updates.append("title = ?")
            params.append(" ".join(str(title).split()).strip()[:240])
        if content is not None:
            clean_content = str(content).strip()
            if not clean_content:
                raise ValueError("Memory content cannot be empty")
            updates.append("content = ?")
            params.append(clean_content[:16000])
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(1 if pinned else 0)
        if not updates:
            return self.get_memory(memory_id)
        params.extend([int(memory_id)])
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                tuple(params),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Memory not found: {memory_id}")
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (int(memory_id),),
            )
        return cursor.rowcount > 0

    def touch_memory(self, memory_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_used_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (int(memory_id),),
            )

    def record_file_revision(
        self,
        *,
        session_id: str | None,
        run_id: str | None,
        path: str,
        operation: str,
        before_content: str | None,
        after_content: str | None,
        diff_summary: str = "",
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO file_revisions
                    (session_id, run_id, path, operation, before_content, after_content, diff_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    str(path),
                    str(operation),
                    before_content,
                    after_content,
                    str(diff_summary)[:4000],
                ),
            )
            revision_id = int(cursor.lastrowid)
        return self.get_file_revision(revision_id)

    def get_file_revision(self, revision_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, run_id, path, operation, before_content, after_content,
                       diff_summary, undone, created_at, undone_at
                FROM file_revisions WHERE id = ?
                """,
                (int(revision_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"File revision not found: {revision_id}")
        item = dict(row)
        item["undone"] = bool(item.get("undone"))
        return item

    def list_file_revisions(
        self,
        session_id: str,
        *,
        include_undone: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        condition = "" if include_undone else "AND undone = 0"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, run_id, path, operation, before_content, after_content,
                       diff_summary, undone, created_at, undone_at
                FROM file_revisions
                WHERE session_id = ? {condition}
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max(1, int(limit))),
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["undone"] = bool(item.get("undone"))
        return result

    def get_latest_file_revision(self, session_id: str) -> dict[str, Any] | None:
        items = self.list_file_revisions(session_id, include_undone=False, limit=1)
        return items[0] if items else None

    def mark_file_revision_undone(self, revision_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE file_revisions
                SET undone = 1, undone_at = CURRENT_TIMESTAMP
                WHERE id = ? AND undone = 0
                """,
                (int(revision_id),),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"File revision not found or already undone: {revision_id}")
        return self.get_file_revision(revision_id)
