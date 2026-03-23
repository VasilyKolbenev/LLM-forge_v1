"""Durable session store backed by the assistant_sessions SQLite table."""

import json
import logging
from datetime import datetime
from typing import Any

from pulsar_ai.storage.database import Database, get_database

logger = logging.getLogger(__name__)


class SessionStore:
    """Persist chat sessions (assistant + site_chat) in SQLite.

    Args:
        db: Database instance. Uses singleton if not provided.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_database()

    def get_or_create(
        self,
        session_id: str,
        session_type: str = "assistant",
        ttl_hours: int = 24,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Return existing session or create a new one.

        Args:
            session_id: Unique session identifier.
            session_type: 'assistant' or 'site_chat'.
            ttl_hours: Time-to-live in hours.
            user_id: Owner user ID.

        Returns:
            Session dict with id, session_type, messages, created_at, updated_at.
        """
        row = self._db.fetch_one("SELECT * FROM assistant_sessions WHERE id = ?", (session_id,))
        if row:
            return self._row_to_dict(row)

        now = datetime.now().isoformat()
        self._db.execute(
            "INSERT INTO assistant_sessions "
            "(id, session_type, messages, created_at, updated_at, ttl_hours, user_id) "
            "VALUES (?, ?, '[]', ?, ?, ?, ?)",
            (session_id, session_type, now, now, ttl_hours, user_id),
        )
        self._db.commit()
        return {
            "id": session_id,
            "session_type": session_type,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "ttl_hours": ttl_hours,
        }

    def get(self, session_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.
            user_id: Optional user filter.

        Returns:
            Session dict or None.
        """
        sql = "SELECT * FROM assistant_sessions WHERE id = ?"
        params: list[Any] = [session_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        row = self._db.fetch_one(sql, tuple(params))
        return self._row_to_dict(row) if row else None

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Get message history for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of message dicts with role and content.
        """
        row = self._db.fetch_one(
            "SELECT messages FROM assistant_sessions WHERE id = ?", (session_id,)
        )
        if not row:
            return []
        return json.loads(row["messages"])

    def append_message(
        self, session_id: str, role: str, content: str, max_messages: int = 0
    ) -> None:
        """Append a message and update the session timestamp.

        Args:
            session_id: Session identifier.
            role: Message role (user, assistant, system).
            content: Message text.
            max_messages: Trim to this many messages if > 0.
        """
        messages = self.get_messages(session_id)
        messages.append({"role": role, "content": content})
        if max_messages > 0 and len(messages) > max_messages:
            messages = messages[-max_messages:]
        now = datetime.now().isoformat()
        self._db.execute(
            "UPDATE assistant_sessions SET messages = ?, updated_at = ? WHERE id = ?",
            (json.dumps(messages, ensure_ascii=False), now, session_id),
        )
        self._db.commit()

    def set_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """Replace all messages in a session.

        Args:
            session_id: Session identifier.
            messages: Full message list.
        """
        now = datetime.now().isoformat()
        self._db.execute(
            "UPDATE assistant_sessions SET messages = ?, updated_at = ? WHERE id = ?",
            (json.dumps(messages, ensure_ascii=False), now, session_id),
        )
        self._db.commit()

    def delete(self, session_id: str, user_id: str | None = None) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.
            user_id: Optional user filter.

        Returns:
            True if deleted, False if not found.
        """
        sql = "DELETE FROM assistant_sessions WHERE id = ?"
        params: list[Any] = [session_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        self._db.execute(sql, tuple(params))
        self._db.commit()
        return self._db.execute("SELECT changes()").fetchone()[0] > 0

    def list_sessions(
        self,
        session_type: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by type and user.

        Args:
            session_type: Filter by session type.
            user_id: Optional user filter.

        Returns:
            List of session dicts (newest first).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_type:
            clauses.append("session_type = ?")
            params.append(session_type)

        sql = "SELECT * FROM assistant_sessions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC"

        rows = self._db.fetch_all(sql, tuple(params))
        return [self._row_to_dict(r) for r in rows]

    def cleanup_expired(self) -> int:
        """Delete sessions past their TTL.

        Returns:
            Number of deleted sessions.
        """
        # SQLite datetime arithmetic: updated_at + ttl_hours < now
        now = datetime.now().isoformat()
        self._db.execute(
            "DELETE FROM assistant_sessions "
            "WHERE datetime(updated_at, '+' || ttl_hours || ' hours') < datetime(?)",
            (now,),
        )
        self._db.commit()
        count = self._db.execute("SELECT changes()").fetchone()[0]
        if count:
            logger.info("Cleaned up %d expired sessions", count)
        return count

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert a database row to a session dict."""
        return {
            "id": row["id"],
            "session_type": row["session_type"],
            "messages": json.loads(row["messages"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "ttl_hours": row["ttl_hours"],
        }
