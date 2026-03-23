"""Approval request store for governance workflows.

Manages the lifecycle of approval requests: creation, review,
and status tracking for high-risk operations.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


class ApprovalStore:
    """Store and manage approval requests."""

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database
            self._db = get_database()

    def create_request(
        self,
        resource_type: str,
        resource_id: str,
        requester_id: str,
        action: str = "execute",
        workspace_id: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Create a new approval request.

        Args:
            resource_type: Type of resource (workflow, experiment, etc.).
            resource_id: ID of the resource.
            requester_id: User ID of the requester.
            action: Action requiring approval.
            workspace_id: Workspace context.
            reason: Why approval is needed.

        Returns:
            Created request dict.
        """
        request_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        self._db.execute(
            """INSERT INTO approval_requests
               (id, workspace_id, resource_type, resource_id, action,
                requester_id, status, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (request_id, workspace_id, resource_type, resource_id,
             action, requester_id, reason, now),
        )
        self._db.commit()

        logger.info(
            "Created approval request %s: %s/%s by %s",
            request_id, resource_type, resource_id, requester_id,
        )
        return {
            "id": request_id, "workspace_id": workspace_id,
            "resource_type": resource_type, "resource_id": resource_id,
            "action": action, "requester_id": requester_id,
            "status": "pending", "reason": reason, "created_at": now,
        }

    def get_by_id(self, request_id: str) -> dict[str, Any] | None:
        """Get a single approval request."""
        row = self._db.fetch_one(
            "SELECT * FROM approval_requests WHERE id = ?", (request_id,),
        )
        return dict(row) if row else None

    def list_requests(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        requester_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List approval requests with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if requester_id:
            clauses.append("requester_id = ?")
            params.append(requester_id)

        sql = "SELECT * FROM approval_requests"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._db.fetch_all(sql, tuple(params))
        return [dict(r) for r in rows]

    def review(
        self,
        request_id: str,
        approver_id: str,
        decision: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Review (approve/reject) an approval request.

        Args:
            request_id: ID of the approval request.
            approver_id: User ID of the reviewer.
            decision: "approved" or "rejected".
            note: Optional review note.

        Returns:
            Updated request dict.

        Raises:
            ValueError: If request not found or already decided.
        """
        req = self.get_by_id(request_id)
        if req is None:
            raise ValueError(f"Approval request not found: {request_id}")
        if req["status"] != "pending":
            raise ValueError(f"Request already {req['status']}")
        if decision not in ("approved", "rejected"):
            raise ValueError(f"Invalid decision: {decision}")

        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """UPDATE approval_requests
               SET status = ?, approver_id = ?, review_note = ?, decided_at = ?
               WHERE id = ?""",
            (decision, approver_id, note, now, request_id),
        )
        self._db.commit()

        logger.info(
            "Approval %s %s by %s: %s/%s",
            request_id, decision, approver_id,
            req["resource_type"], req["resource_id"],
        )
        req.update({
            "status": decision, "approver_id": approver_id,
            "review_note": note, "decided_at": now,
        })
        return req

    def get_pending_count(self, workspace_id: str = "") -> int:
        """Count pending approval requests."""
        row = self._db.fetch_one(
            "SELECT COUNT(*) as cnt FROM approval_requests WHERE workspace_id = ? AND status = 'pending'",
            (workspace_id,),
        )
        return row["cnt"] if row else 0
