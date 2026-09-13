"""SQLite-backed conversation storage for Drudge sessions."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .conversation import encode_context, message_from_row
from tools.output_capture import MAX_OUTPUT_BYTES, MAX_OUTPUT_PAGE_CHARS, OUTPUT_CHUNK_CHARS, utf8_prefix
from tools.plan import normalize_plan


class ContextConflictError(RuntimeError):
    """A different consumer advanced this session's durable context."""


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

                CREATE TABLE IF NOT EXISTS context_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    through_message_id INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    messages_json TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS tool_outputs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    run_id TEXT,
                    tool_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    source_chars INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    complete INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS tool_output_chunks (
                    output_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content BLOB NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY(output_id, chunk_index),
                    FOREIGN KEY(output_id) REFERENCES tool_outputs(id)
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

                CREATE TABLE IF NOT EXISTS session_plans (
                    session_id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
                    snapshot_version INTEGER NOT NULL DEFAULT 0,
                    before_sha256 TEXT,
                    after_sha256 TEXT,
                    diff_summary TEXT NOT NULL DEFAULT '',
                    undone INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    undone_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);

                CREATE INDEX IF NOT EXISTS idx_context_checkpoints_session_id
                ON context_checkpoints(session_id, id);

                CREATE INDEX IF NOT EXISTS idx_tool_outputs_scope
                ON tool_outputs(session_id, workspace, created_at);

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
            revision_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(file_revisions)")
            }
            # Old checkpoints used platform-translated text. Do not invent
            # byte-exact hashes for them; version 0 remains distinguishable.
            for name, declaration in (
                ("snapshot_version", "INTEGER NOT NULL DEFAULT 0"),
                ("before_sha256", "TEXT"),
                ("after_sha256", "TEXT"),
            ):
                if name not in revision_columns:
                    conn.execute(f"ALTER TABLE file_revisions ADD COLUMN {name} {declaration}")
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
        expected_message_id: int | None = None,
        expected_checkpoint_id: int | None = None,
    ) -> int:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            if expected_message_id is not None:
                conn.execute("BEGIN IMMEDIATE")
                self._check_context_frontier(conn, session_id, expected_message_id, expected_checkpoint_id)
            cursor = conn.execute(
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
            return int(cursor.lastrowid)

    @staticmethod
    def _context_frontier(conn: sqlite3.Connection, session_id: str) -> tuple[int, int | None]:
        if conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        last_message = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?", (session_id,),
        ).fetchone()[0]
        last_checkpoint = conn.execute(
            "SELECT MAX(id) FROM context_checkpoints WHERE session_id = ?", (session_id,),
        ).fetchone()[0]
        return int(last_message), last_checkpoint

    @classmethod
    def _check_context_frontier(
        cls, conn: sqlite3.Connection, session_id: str,
        expected_message_id: int, expected_checkpoint_id: int | None,
    ) -> None:
        if cls._context_frontier(conn, session_id) != (expected_message_id, expected_checkpoint_id):
            raise ContextConflictError("Session changed in another consumer; resume it before continuing.")

    @staticmethod
    def _insert_context_message(
        conn: sqlite3.Connection, session_id: str, message: dict[str, Any],
        *, repaired: bool = False,
    ) -> int:
        metadata = {key: message[key] for key in ("tool_calls", "provider_items") if key in message}
        if repaired:
            metadata["repaired"] = True
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (session_id, message["role"], message.get("content"), message.get("tool_call_id"), json.dumps(metadata, ensure_ascii=False)),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _context_digest(messages_json: str, through_message_id: int, metadata_json: str) -> str:
        envelope = json.dumps(
            [1, through_message_id, metadata_json, messages_json],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(envelope.encode("utf-8")).hexdigest()

    def save_context_checkpoint(
        self, session_id: str, messages: list[dict[str, Any]], *,
        expected_message_id: int, expected_checkpoint_id: int | None,
        metadata: dict[str, Any] | None = None,
        repair_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Commit a working context and recovery audit entries as one transaction.

        Raw history is append-only. Both watermarks are checked so a stale
        compactor cannot hide another consumer's messages or newer checkpoint.
        """
        encoded = encode_context(messages)
        metadata = dict(metadata or {})
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_context_frontier(conn, session_id, expected_message_id, expected_checkpoint_id)
            through_id = expected_message_id
            for message in repair_messages or []:
                if message.get("role") != "tool":
                    raise ValueError("Recovery audit entries must be tool results")
                through_id = self._insert_context_message(conn, session_id, message, repaired=True)
            cursor = conn.execute(
                """INSERT INTO context_checkpoints
                   (session_id, through_message_id, messages_json, content_sha256, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, through_id, encoded, self._context_digest(encoded, through_id, metadata_json), metadata_json),
            )
            checkpoint_id = int(cursor.lastrowid)
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        return {"id": checkpoint_id, "through_message_id": through_id, "metadata": metadata}

    def load_session_context(self, session_id: str) -> dict[str, Any]:
        """Read checkpoint plus the raw tail from one SQLite read snapshot."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            last_message_id, last_checkpoint_id = self._context_frontier(conn, session_id)
            row = conn.execute(
                "SELECT * FROM context_checkpoints WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            messages: list[dict[str, Any]] = []
            checkpoint = None
            warning = None
            through_id = 0
            if row is not None:
                try:
                    if row["schema_version"] != 1:
                        raise ValueError("unsupported schema version")
                    if not isinstance(row["messages_json"], str):
                        raise ValueError("invalid message encoding")
                    if self._context_digest(row["messages_json"], row["through_message_id"], row["metadata_json"]) != row["content_sha256"]:
                        raise ValueError("checksum mismatch")
                    through_id = row["through_message_id"]
                    if through_id < 0 or through_id > last_message_id or (through_id and conn.execute(
                        "SELECT id FROM messages WHERE session_id = ? AND id = ?", (session_id, through_id),
                    ).fetchone() is None):
                        raise ValueError("invalid message watermark")
                    decoded = json.loads(row["messages_json"])
                    messages = json.loads(encode_context(decoded))
                    metadata = json.loads(row["metadata_json"])
                    if not isinstance(metadata, dict):
                        raise ValueError("invalid checkpoint metadata")
                    checkpoint = {"id": row["id"], "through_message_id": through_id, "metadata": metadata}
                except (ValueError, TypeError, KeyError) as exc:
                    messages = []
                    through_id = 0
                    warning = f"Context checkpoint #{row['id']} ignored ({exc}); restored raw history."
            rows = conn.execute(
                """SELECT id, role, content, tool_call_id, metadata_json FROM messages
                   WHERE session_id = ? AND id > ? ORDER BY id""",
                (session_id, through_id),
            ).fetchall()
            for item in rows:
                raw = dict(item)
                raw["metadata"] = json.loads(raw.pop("metadata_json") or "{}")
                messages.append(message_from_row(raw))
            assistant_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'assistant'", (session_id,),
            ).fetchone()[0]
        return {
            "messages": messages, "last_message_id": last_message_id,
            "checkpoint_id": last_checkpoint_id, "checkpoint": checkpoint,
            "warning": warning, "assistant_count": int(assistant_count),
            "tail_assistant_count": sum(item["role"] == "assistant" for item in rows),
        }

    def fork_session(
        self, parent_id: str, messages: list[dict[str, Any]], *,
        title: str, model: str, cwd: str,
        expected_message_id: int, expected_checkpoint_id: int | None,
        active_skills: list[str] | None = None,
        expected_plan_revision: int = 0,
    ) -> str:
        """Branch working context only; tools, approvals and file undo are not cloned."""
        messages = json.loads(encode_context(messages))
        if not messages:
            raise ValueError("An empty conversation cannot be forked")
        session_id = uuid.uuid4().hex[:12]
        metadata = {
            "parent_session_id": parent_id,
            "parent_message_id": expected_message_id,
            "parent_context_checkpoint_id": expected_checkpoint_id,
            "active_skills": list(active_skills or []),
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_context_frontier(conn, parent_id, expected_message_id, expected_checkpoint_id)
            plan = self._load_plan(conn, parent_id, cwd)
            if plan["revision"] != expected_plan_revision:
                raise ContextConflictError("Session plan changed; resume it before forking.")
            conn.execute(
                "INSERT INTO sessions (id, title, model, cwd, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (session_id, title[:200], model, cwd, json.dumps(metadata, ensure_ascii=False)),
            )
            for message in messages:
                self._insert_context_message(conn, session_id, message)
            if plan["revision"]:
                self._write_plan(conn, session_id, cwd, 1, normalize_plan(plan["plan"], plan["explanation"] or ""))
        return session_id

    @staticmethod
    def _plan_digest(session_id: str, workspace: str, revision: int, payload_json: str) -> str:
        encoded = json.dumps([1, session_id, workspace, revision, payload_json], ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def _load_plan(cls, conn: sqlite3.Connection, session_id: str, workspace: str) -> dict[str, Any]:
        row = conn.execute("SELECT * FROM session_plans WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return {"plan": [], "explanation": None, "revision": 0}
        workspace = str(Path(workspace).expanduser().resolve())
        if os.path.normcase(row["workspace"]) != os.path.normcase(workspace):
            raise RuntimeError("Session plan belongs to a different workspace")
        if row["schema_version"] != 1 or row["revision"] < 1:
            raise RuntimeError("Unsupported session plan schema or revision")
        if row["sha256"] != cls._plan_digest(session_id, row["workspace"], row["revision"], row["payload_json"]):
            raise RuntimeError("Session plan checksum mismatch")
        try:
            payload = json.loads(row["payload_json"])
            payload = normalize_plan(payload["plan"], payload["explanation"] or "")
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError("Invalid stored session plan") from exc
        return {**payload, "revision": row["revision"]}

    @classmethod
    def _write_plan(
        cls, conn: sqlite3.Connection, session_id: str, workspace: str,
        revision: int, payload: dict[str, Any],
    ) -> None:
        workspace = str(Path(workspace).expanduser().resolve())
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        conn.execute(
            """INSERT INTO session_plans (session_id, workspace, revision, payload_json, sha256)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 revision=excluded.revision, payload_json=excluded.payload_json,
                 sha256=excluded.sha256, workspace=excluded.workspace, updated_at=CURRENT_TIMESTAMP""",
            (session_id, workspace, revision, encoded, cls._plan_digest(session_id, workspace, revision, encoded)),
        )

    def load_plan(self, session_id: str, *, workspace: str) -> dict[str, Any]:
        with self._connect() as conn:
            if conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            return self._load_plan(conn, session_id, workspace)

    def save_plan(
        self, session_id: str, plan: list[dict[str, str]], explanation: str = "", *,
        workspace: str, expected_revision: int, expected_message_id: int,
        expected_checkpoint_id: int | None, run_id: str | None = None,
    ) -> dict[str, Any]:
        """CAS update and audit commit together; model arguments never set scope."""
        payload = normalize_plan(plan, explanation)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_context_frontier(conn, session_id, expected_message_id, expected_checkpoint_id)
            current = self._load_plan(conn, session_id, workspace)
            if current["revision"] != expected_revision:
                raise ContextConflictError("Session plan changed in another consumer; resume it before continuing.")
            revision = current["revision"] + 1
            self._write_plan(conn, session_id, workspace, revision, payload)
            if run_id:
                if conn.execute("SELECT id FROM runs WHERE id = ? AND session_id = ?", (run_id, session_id)).fetchone() is None:
                    raise ValueError("Plan run does not belong to the active session")
                conn.execute(
                    "INSERT INTO run_events (run_id, kind, detail_json) VALUES (?, 'plan_updated', ?)",
                    (run_id, json.dumps({**payload, "revision": revision}, ensure_ascii=False)),
                )
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        return {**payload, "revision": revision}

    def save_tool_output(
        self, content: str, *, session_id: str, workspace: str,
        run_id: str | None = None, tool_name: str = "unknown",
        source_chars: int | None = None, complete: bool = True,
        kind: str = "json", status: str = "completed",
        max_bytes: int = MAX_OUTPUT_BYTES,
    ) -> dict[str, Any]:
        """Store capped UTF-8 text with a session/workspace-scoped opaque receipt."""
        original_chars = len(content) if source_chars is None else max(len(content), int(source_chars))
        stored = utf8_prefix(content, max_bytes)
        data = stored.encode("utf-8")
        complete = bool(complete and len(stored) == original_chars)
        if kind == "json" and not complete:
            kind = "text"  # A capped JSON prefix is explicitly not a JSON document.
        receipt = {
            "id": uuid.uuid4().hex,
            "tool_name": str(tool_name)[:100],
            "kind": kind, "status": status,
            "size_bytes": len(data), "char_count": len(stored), "source_chars": original_chars,
            "sha256": hashlib.sha256(data).hexdigest(), "complete": complete,
        }
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tool_outputs
                   (id, session_id, workspace, run_id, tool_name, kind, status,
                    size_bytes, char_count, source_chars, sha256, complete)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (receipt["id"], session_id, str(Path(workspace).expanduser().resolve()), run_id,
                 receipt["tool_name"], kind, status, len(data), len(stored),
                 original_chars, receipt["sha256"], int(complete)),
            )
            for index, start in enumerate(range(0, len(stored), OUTPUT_CHUNK_CHARS)):
                chunk = stored[start:start + OUTPUT_CHUNK_CHARS].encode("utf-8")
                conn.execute(
                    "INSERT INTO tool_output_chunks (output_id, chunk_index, content, sha256) VALUES (?, ?, ?, ?)",
                    (receipt["id"], index, chunk, hashlib.sha256(chunk).hexdigest()),
                )
        return receipt

    def read_tool_output(
        self, output_id: str, *, session_id: str, workspace: str,
        offset: int = 0, limit: int = MAX_OUTPUT_PAGE_CHARS,
    ) -> dict[str, Any]:
        """Read only the fixed-size chunks covering a Unicode-character page.

        UTF-8 BLOB chunks preserve NUL characters (SQLite TEXT substr does not).
        This path does not scan or materialize earlier output to reach an offset.
        """
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_OUTPUT_PAGE_CHARS:
            raise ValueError(f"limit must be between 1 and {MAX_OUTPUT_PAGE_CHARS}")
        if not isinstance(output_id, str) or len(output_id) != 32 or any(c not in "0123456789abcdef" for c in output_id):
            raise ValueError("Invalid tool output ID")
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, tool_name, kind, status, size_bytes, char_count, source_chars,
                          sha256, complete
                   FROM tool_outputs WHERE id = ? AND session_id = ? AND workspace = ?""",
                (output_id, session_id, str(Path(workspace).expanduser().resolve())),
            ).fetchone()
            if row is None:
                raise KeyError("Tool output not found in the active session and workspace")
            first = offset // OUTPUT_CHUNK_CHARS
            last = (offset + limit - 1) // OUTPUT_CHUNK_CHARS
            chunks = conn.execute(
                "SELECT content, sha256 FROM tool_output_chunks WHERE output_id = ? AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
                (output_id, first, last),
            ).fetchall()
        for chunk in chunks:
            if hashlib.sha256(chunk["content"]).hexdigest() != chunk["sha256"]:
                raise RuntimeError("Tool output chunk integrity check failed")
        text = b"".join(chunk["content"] for chunk in chunks).decode("utf-8")
        page = text[offset % OUTPUT_CHUNK_CHARS:offset % OUTPUT_CHUNK_CHARS + limit]
        if len(page) != min(limit, max(0, row["char_count"] - offset)):
            raise RuntimeError("Tool output chunks are incomplete")
        result = dict(row)
        result["content"] = page
        result["complete"] = bool(result["complete"])
        next_offset = min(offset + len(result["content"]), result["char_count"])
        result.update({
            "offset": offset, "next_offset": next_offset,
            "eof": next_offset >= result["char_count"],
        })
        return result

    def list_tool_outputs(self, *, session_id: str, workspace: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, tool_name, kind, status, size_bytes, char_count, source_chars,
                          sha256, complete, created_at
                   FROM tool_outputs WHERE session_id = ? AND workspace = ?
                   ORDER BY created_at DESC, rowid DESC LIMIT ?""",
                (session_id, str(Path(workspace).expanduser().resolve()), max(1, min(int(limit), 100))),
            ).fetchall()
        return [{**dict(row), "complete": bool(row["complete"])} for row in rows]

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
        def digest(content: str | None) -> str | None:
            return hashlib.sha256(content.encode("utf-8")).hexdigest() if content is not None else None

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO file_revisions
                    (session_id, run_id, path, operation, before_content, after_content, diff_summary,
                     snapshot_version, before_sha256, after_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    str(path),
                    str(operation),
                    before_content,
                    after_content,
                    str(diff_summary)[:4000],
                    digest(before_content),
                    digest(after_content),
                ),
            )
            revision_id = int(cursor.lastrowid)
        return self.get_file_revision(revision_id)

    def get_file_revision(self, revision_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, run_id, path, operation, before_content, after_content,
                       diff_summary, undone, created_at, undone_at,
                       snapshot_version, before_sha256, after_sha256
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
                       diff_summary, undone, created_at, undone_at,
                       snapshot_version, before_sha256, after_sha256
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

    def apply_file_revision_undo(
        self, revision_id: int, apply_change: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Serialize undo consumers; roll back the marker if file restoration fails.

        Filesystem and SQLite commits are not a distributed transaction. In
        particular, a process crash between them still requires manual review.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM file_revisions WHERE id = ?", (int(revision_id),)).fetchone()
            if row is None or row["undone"]:
                raise RuntimeError(f"File revision not found or already undone: {revision_id}")
            apply_change(dict(row))
            conn.execute(
                "UPDATE file_revisions SET undone = 1, undone_at = CURRENT_TIMESTAMP WHERE id = ?",
                (int(revision_id),),
            )
        return self.get_file_revision(revision_id)

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
