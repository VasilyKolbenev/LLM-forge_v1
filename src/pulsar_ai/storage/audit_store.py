"""Audit log store for governance trail.

Records all state-changing operations for compliance and debugging.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


class AuditStore:
    """Store and query audit logs."""

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database
            self._db = get_database()

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str = "",
        user_id: str = "",
        workspace_id: str = "",
        details: str = "{}",
        ip_address: str = "",
    ) -> None:
        """Write an audit log entry.

        Args:
            action: HTTP method or operation name.
            resource_type: Type of resource affected.
            resource_id: ID of the resource.
            user_id: Acting user ID.
            workspace_id: Workspace context.
            details: JSON string with additional context.
            ip_address: Client IP address.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """INSERT INTO audit_logs
               (user_id, workspace_id, action, resource_type,
                resource_id, details, ip_address, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, workspace_id, action, resource_type,
             resource_id, details, ip_address, now),
        )
        self._db.commit()

    def query(
        self,
        workspace_id: str | None = None,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query audit logs with filters.

        Args:
            workspace_id: Filter by workspace.
            user_id: Filter by user.
            action: Filter by action (POST, DELETE, etc.).
            resource_type: Filter by resource type.
            from_date: Start date (ISO format).
            to_date: End date (ISO format).
            limit: Max results.
            offset: Skip N results.

        Returns:
            List of audit log dicts.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if resource_type:
            clauses.append("resource_type = ?")
            params.append(resource_type)
        if from_date:
            clauses.append("timestamp >= ?")
            params.append(from_date)
        if to_date:
            clauses.append("timestamp <= ?")
            params.append(to_date)

        sql = "SELECT * FROM audit_logs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._db.fetch_all(sql, tuple(params))
        return [dict(r) for r in rows]

    def count(
        self,
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count audit log entries."""
        clauses: list[str] = []
        params: list[Any] = []

        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)

        sql = "SELECT COUNT(*) as cnt FROM audit_logs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        row = self._db.fetch_one(sql, tuple(params))
        return row["cnt"] if row else 0
