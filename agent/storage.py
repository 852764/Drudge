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
