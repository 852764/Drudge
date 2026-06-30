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

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);

                CREATE INDEX IF NOT EXISTS idx_tool_calls_session_id
                ON tool_calls(session_id, id);
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
