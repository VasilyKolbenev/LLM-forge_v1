"""RBAC permission system for Pulsar AI.

Defines roles, permissions, and a FastAPI dependency for route-level
authorization checks. Roles are hierarchical: admin > manager > member > viewer.

Permissions are defined as code (not DB) for simplicity and performance.
"""

import logging
import os
from functools import lru_cache
from typing import Callable

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)

# Role hierarchy (higher = more privileges)
ROLE_LEVELS = {
    "admin": 40,
    "manager": 30,
    "member": 20,
    "viewer": 10,
}

# Cumulative permissions per role
_VIEWER_PERMS = {
    "*.read",
    "traces.read",
    "benchmarks.read",
    "audit.read",
}

_MEMBER_PERMS = _VIEWER_PERMS | {
    "experiments.create",
    "experiments.update",
    "workflows.create",
    "workflows.update",
    "workflows.execute",
    "prompts.create",
    "prompts.update",
    "datasets.upload",
    "datasets.create",
    "models.evaluate",
    "benchmarks.run",
    "traces.create",
}

_MANAGER_PERMS = _MEMBER_PERMS | {
    "experiments.delete",
    "workflows.delete",
    "prompts.delete",
    "datasets.delete",
    "models.export",
    "approvals.review",
    "users.read",
    "workspaces.update",
    "compute.manage",
}

_ADMIN_PERMS = _MANAGER_PERMS | {
    "users.create",
    "users.update",
    "users.delete",
    "workspaces.create",
    "workspaces.delete",
    "settings.update",
    "apikeys.manage",
    "approvals.override",
}

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": _VIEWER_PERMS,
    "member": _MEMBER_PERMS,
    "manager": _MANAGER_PERMS,
    "admin": _ADMIN_PERMS,
}


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission.

    Supports wildcard matching: *.read matches experiments.read.

    Args:
        role: User role string.
        permission: Permission to check (e.g., "experiments.create").

    Returns:
        True if the role has the permission.
    """
    perms = ROLE_PERMISSIONS.get(role, set())

    # Direct match
    if permission in perms:
        return True

    # Wildcard match: check if *.action is in perms
    parts = permission.split(".")
    if len(parts) == 2:
        wildcard = f"*.{parts[1]}"
        if wildcard in perms:
            return True

    return False


def get_role_level(role: str) -> int:
    """Get the hierarchy level for a role."""
    return ROLE_LEVELS.get(role, 0)


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return os.environ.get("PULSAR_AUTH_ENABLED", "false").lower() == "true"


def require_permission(permission: str) -> Depends:
    """Create a FastAPI dependency that checks for a specific permission.

    Usage in routes:
        @router.post("/experiments")
        async def create(request: Request, _=require_permission("experiments.create")):
            ...

    When auth is disabled, all permissions are granted.

    Args:
        permission: Required permission string.

    Returns:
        FastAPI Depends object.
    """
    async def _check(request: Request) -> None:
        # When auth is disabled, allow everything
        if not _is_auth_enabled():
            return

        user = getattr(request.state, "user", None)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")

        role = user.get("role", "viewer")
        if not has_permission(role, permission):
            logger.warning(
                "Permission denied: user=%s role=%s needs=%s",
                user.get("email", "?"), role, permission,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission}",
            )

    return Depends(_check)
