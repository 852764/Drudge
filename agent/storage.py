"""SQLite-backed conversation storage for Drudge sessions."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any


class ConversationStore:
    """Persist sessions, messages, and tool calls to a local SQLite database."""

    def __init__(self, db_path: str):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    cwd TEXT NOT NULL,
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
                """
            )

    def create_session(self, title: str, model: str, cwd: str | None = None) -> str:
        session_id = uuid.uuid4().hex[:12]
        clean_title = " ".join(title.split())[:80] or "Untitled session"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, model, cwd) VALUES (?, ?, ?, ?)",
                (session_id, clean_title, model, cwd or os.getcwd()),
            )
        return session_id

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
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

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, tool_call_id, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            messages.append(item)
        return messages

