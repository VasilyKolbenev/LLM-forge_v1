"""Workspace management store.

Handles workspace CRUD and membership operations for multi-tenant
isolation of resources.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Convert name to URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "workspace"


class WorkspaceStore:
    """Store and manage workspaces and memberships."""

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database
            self._db = get_database()

    def create_workspace(
        self,
        name: str,
        owner_id: str,
    ) -> dict[str, Any]:
        """Create a workspace and add the owner as admin member.

        Args:
            name: Workspace display name.
            owner_id: User ID of the workspace owner.

        Returns:
            Created workspace dict.
        """
        workspace_id = uuid.uuid4().hex[:12]
        slug = f"{_slugify(name)}-{workspace_id[:4]}"
        now = datetime.now(timezone.utc).isoformat()

        self._db.execute(
            """INSERT INTO workspaces (id, name, slug, owner_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (workspace_id, name, slug, owner_id, now, now),
        )

        # Auto-add owner as admin
        self._db.execute(
            """INSERT INTO workspace_members (workspace_id, user_id, role, joined_at)
               VALUES (?, ?, 'admin', ?)""",
            (workspace_id, owner_id, now),
        )
        self._db.commit()

        logger.info("Created workspace %s (%s) for user %s", name, workspace_id, owner_id)
        return {
            "id": workspace_id, "name": name, "slug": slug,
            "owner_id": owner_id, "created_at": now, "updated_at": now,
        }

    def get_by_id(self, workspace_id: str) -> dict[str, Any] | None:
        """Get workspace by ID."""
        row = self._db.fetch_one(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,),
        )
        return dict(row) if row else None

    def list_workspaces(
        self,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List workspaces, optionally filtered by user membership.

        Args:
            user_id: If provided, only return workspaces the user belongs to.
            limit: Max results.

        Returns:
            List of workspace dicts.
        """
        if user_id:
            rows = self._db.fetch_all(
                """SELECT w.* FROM workspaces w
                   JOIN workspace_members wm ON w.id = wm.workspace_id
                   WHERE wm.user_id = ?
                   ORDER BY w.updated_at DESC LIMIT ?""",
                (user_id, limit),
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM workspaces ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in rows]

    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "member",
    ) -> dict[str, Any]:
        """Add a user to a workspace."""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """INSERT OR IGNORE INTO workspace_members
               (workspace_id, user_id, role, joined_at) VALUES (?, ?, ?, ?)""",
            (workspace_id, user_id, role, now),
        )
        self._db.commit()
        return {"workspace_id": workspace_id, "user_id": user_id, "role": role}

    def remove_member(self, workspace_id: str, user_id: str) -> bool:
        """Remove a user from a workspace."""
        self._db.execute(
            "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        self._db.commit()
        return True

    def update_member_role(
        self, workspace_id: str, user_id: str, role: str,
    ) -> bool:
        """Update a member's role in a workspace."""
        self._db.execute(
            "UPDATE workspace_members SET role = ? WHERE workspace_id = ? AND user_id = ?",
            (role, workspace_id, user_id),
        )
        self._db.commit()
        return True

    def get_members(self, workspace_id: str) -> list[dict[str, Any]]:
        """Get members of a workspace with user details."""
        rows = self._db.fetch_all(
            """SELECT wm.user_id, wm.role, wm.joined_at,
                      u.email, u.name as user_name
               FROM workspace_members wm
               JOIN users u ON wm.user_id = u.id
               WHERE wm.workspace_id = ?
               ORDER BY wm.joined_at""",
            (workspace_id,),
        )
        return [dict(r) for r in rows]

    def is_member(self, workspace_id: str, user_id: str) -> bool:
        """Check if a user is a member of a workspace."""
        row = self._db.fetch_one(
            "SELECT 1 FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        return row is not None

    def get_member_role(self, workspace_id: str, user_id: str) -> str | None:
        """Get a user's role in a workspace."""
        row = self._db.fetch_one(
            "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        return row["role"] if row else None

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all memberships."""
        self._db.execute("DELETE FROM workspace_members WHERE workspace_id = ?", (workspace_id,))
        self._db.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        self._db.commit()
        return True
