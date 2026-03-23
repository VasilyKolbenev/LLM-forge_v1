"""Admin API routes for user management and system health.

All endpoints require ``role=admin``. Non-admin users receive 403.
"""

import logging
import os
import secrets
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

import bcrypt
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pulsar_ai.storage.database import get_database
from pulsar_ai.storage.user_store import UserStore
from pulsar_ai.ui.auth import get_current_user
from pulsar_ai.ui.jwt_utils import blacklist_user_tokens, cleanup_expired_blacklist

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_users = UserStore()


# ── Admin guard ───────────────────────────────────────────────────


def _require_admin(request: Request) -> dict:
    """Extract current user and verify admin role.

    Args:
        request: FastAPI request.

    Returns:
        Authenticated admin user dict.

    Raises:
        HTTPException: 403 if user is not an admin.
    """
    user = get_current_user(request)
    if user.get("role") != "admin":
        logger.warning(
            "Admin access denied: user=%s role=%s path=%s",
            user.get("email", "?"),
            user.get("role", "?"),
            request.url.path,
        )
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Request models ────────────────────────────────────────────────


class UpdateUserRequest(BaseModel):
    """Fields allowed for admin user update."""

    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ── User Management ──────────────────────────────────────────────


@router.get("/users")
async def list_users(request: Request) -> dict[str, Any]:
    """List all users (including inactive) with key fields."""
    _require_admin(request)
    db = get_database()
    rows = db.fetch_all(
        "SELECT id, email, name, role, is_active, created_at, last_login_at "
        "FROM users ORDER BY created_at DESC"
    )
    return {"users": rows, "total": len(rows)}


@router.get("/users/{user_id}")
async def get_user_detail(user_id: str, request: Request) -> dict[str, Any]:
    """Get user detail with activity stats."""
    _require_admin(request)
    db = get_database()
    user = db.fetch_one(
        "SELECT id, email, name, role, is_active, created_at, last_login_at "
        "FROM users WHERE id = ?",
        (user_id,),
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    experiment_count = db.fetch_one(
        "SELECT COUNT(*) as cnt FROM experiments WHERE user_id = ?",
        (user_id,),
    )
    dataset_count = db.fetch_one(
        "SELECT COUNT(*) as cnt FROM datasets WHERE user_id = ?",
        (user_id,),
    )

    return {
        **user,
        "experiment_count": experiment_count["cnt"] if experiment_count else 0,
        "dataset_count": dataset_count["cnt"] if dataset_count else 0,
    }


@router.put("/users/{user_id}")
async def update_user(
    user_id: str, body: UpdateUserRequest, request: Request,
) -> dict[str, Any]:
    """Update user name, role, or active status."""
    admin = _require_admin(request)
    db = get_database()

    existing = db.fetch_one("SELECT id FROM users WHERE id = ?", (user_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")

    updates: list[str] = []
    params: list[Any] = []
    if body.name is not None:
        updates.append("name = ?")
        params.append(body.name)
    if body.role is not None:
        if body.role not in ("admin", "manager", "member", "viewer"):
            raise HTTPException(status_code=400, detail="Invalid role")
        updates.append("role = ?")
        params.append(body.role)
    if body.is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if body.is_active else 0)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(user_id)
    db.execute(
        f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
        tuple(params),
    )
    db.commit()
    logger.info(
        "Admin %s updated user %s: %s",
        admin.get("id"), user_id, updates,
    )
    return {"status": "updated", "user_id": user_id}


@router.post("/users/{user_id}/reset-password")
async def reset_password(user_id: str, request: Request) -> dict[str, Any]:
    """Admin reset password — generates a temporary password."""
    _require_admin(request)
    db = get_database()

    existing = db.fetch_one("SELECT id, email FROM users WHERE id = ?", (user_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")

    temp_password = secrets.token_urlsafe(12)
    password_hash = bcrypt.hashpw(
        temp_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    db.commit()

    # Force logout after password reset
    blacklist_user_tokens(user_id)

    logger.info("Admin reset password for user %s (%s)", user_id, existing["email"])
    return {
        "status": "password_reset",
        "user_id": user_id,
        "temp_password": temp_password,
    }


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(user_id: str, request: Request) -> dict[str, Any]:
    """Deactivate a user account."""
    admin = _require_admin(request)
    if admin.get("id") == user_id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    db = get_database()
    cursor = db.execute(
        "UPDATE users SET is_active = 0 WHERE id = ? AND is_active = 1",
        (user_id,),
    )
    db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found or already inactive")

    blacklist_user_tokens(user_id)
    logger.info("Admin %s deactivated user %s", admin.get("id"), user_id)
    return {"status": "deactivated", "user_id": user_id}


@router.post("/users/{user_id}/activate")
async def activate_user(user_id: str, request: Request) -> dict[str, Any]:
    """Reactivate a user account."""
    _require_admin(request)
    db = get_database()
    cursor = db.execute(
        "UPDATE users SET is_active = 1 WHERE id = ? AND is_active = 0",
        (user_id,),
    )
    db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found or already active")

    logger.info("Admin activated user %s", user_id)
    return {"status": "activated", "user_id": user_id}


@router.delete("/users/{user_id}/mfa")
async def disable_mfa(user_id: str, request: Request) -> dict[str, Any]:
    """Admin disable MFA for a user."""
    _require_admin(request)
    db = get_database()

    existing = db.fetch_one("SELECT id FROM users WHERE id = ?", (user_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")

    db.execute(
        "UPDATE users SET mfa_secret = NULL, mfa_enabled = 0 WHERE id = ?",
        (user_id,),
    )
    db.commit()
    logger.info("Admin disabled MFA for user %s", user_id)
    return {"status": "mfa_disabled", "user_id": user_id}


@router.post("/users/{user_id}/force-logout")
async def force_logout(user_id: str, request: Request) -> dict[str, Any]:
    """Blacklist all tokens for a user (force logout everywhere)."""
    _require_admin(request)
    db = get_database()

    existing = db.fetch_one("SELECT id FROM users WHERE id = ?", (user_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")

    blacklist_user_tokens(user_id)
    logger.info("Admin force-logged out user %s", user_id)
    return {"status": "logged_out", "user_id": user_id}


# ── System Health ────────────────────────────────────────────────


@router.get("/system/health")
async def system_health(request: Request) -> dict[str, Any]:
    """Detailed health check: DB, Redis, S3, disk, memory."""
    _require_admin(request)
    checks: dict[str, Any] = {}

    # Database
    try:
        db = get_database()
        db.fetch_one("SELECT 1")
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}

    # Redis
    redis_url = os.environ.get("PULSAR_REDIS_URL", "").strip()
    if redis_url:
        try:
            import redis

            r = redis.from_url(redis_url, socket_timeout=3)
            r.ping()
            checks["redis"] = {"status": "ok"}
        except Exception as exc:
            checks["redis"] = {"status": "error", "detail": str(exc)}
    else:
        checks["redis"] = {"status": "not_configured"}

    # S3
    s3_bucket = os.environ.get("PULSAR_S3_BUCKET", "").strip()
    if s3_bucket:
        try:
            import boto3

            s3 = boto3.client("s3")
            s3.head_bucket(Bucket=s3_bucket)
            checks["s3"] = {"status": "ok", "bucket": s3_bucket}
        except Exception as exc:
            checks["s3"] = {"status": "error", "detail": str(exc)}
    else:
        checks["s3"] = {"status": "not_configured"}

    # Disk space
    try:
        usage = shutil.disk_usage(".")
        checks["disk"] = {
            "status": "ok",
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
            "percent_used": round(usage.used / usage.total * 100, 1),
        }
    except Exception as exc:
        checks["disk"] = {"status": "error", "detail": str(exc)}

    # Memory (psutil optional)
    try:
        import psutil

        mem = psutil.virtual_memory()
        checks["memory"] = {
            "status": "ok",
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "percent_used": mem.percent,
        }
    except ImportError:
        checks["memory"] = {"status": "unavailable", "detail": "psutil not installed"}
    except Exception as exc:
        checks["memory"] = {"status": "error", "detail": str(exc)}

    overall = "ok" if all(
        c.get("status") in ("ok", "not_configured", "unavailable")
        for c in checks.values()
    ) else "degraded"

    return {"status": overall, "checks": checks}


@router.get("/system/stats")
async def system_stats(request: Request) -> dict[str, Any]:
    """System statistics: counts of key entities."""
    _require_admin(request)
    db = get_database()

    def _count(table: str) -> int:
        row = db.fetch_one(f"SELECT COUNT(*) as cnt FROM {table}")  # noqa: S608
        return row["cnt"] if row else 0

    total_users = _count("users")
    active_users = db.fetch_one(
        "SELECT COUNT(*) as cnt FROM users WHERE is_active = 1"
    )

    stats: dict[str, Any] = {
        "total_users": total_users,
        "active_users": active_users["cnt"] if active_users else 0,
    }

    # These tables may not exist in all deployments — count safely
    for table in ("experiments", "datasets", "workflows"):
        try:
            stats[f"total_{table}"] = _count(table)
        except Exception:
            stats[f"total_{table}"] = 0

    # Active jobs (running experiments)
    try:
        running = db.fetch_one(
            "SELECT COUNT(*) as cnt FROM experiments WHERE status = 'running'"
        )
        stats["active_jobs"] = running["cnt"] if running else 0
    except Exception:
        stats["active_jobs"] = 0

    # Database file size (SQLite)
    try:
        db_size = db.fetch_one("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        if db_size and db_size.get("size"):
            stats["db_size_mb"] = round(db_size["size"] / (1024 * 1024), 2)
    except Exception:
        stats["db_size_mb"] = None

    return stats


@router.post("/system/cleanup")
async def system_cleanup(request: Request) -> dict[str, Any]:
    """Cleanup expired tokens, old login attempts, expired sessions."""
    _require_admin(request)
    db = get_database()
    now_iso = datetime.now(timezone.utc).isoformat()
    results: dict[str, int] = {}

    # Expired blacklist tokens
    results["expired_blacklist_tokens"] = cleanup_expired_blacklist()

    # Old login attempts (older than 30 days)
    try:
        cursor = db.execute(
            "DELETE FROM login_attempts WHERE attempted_at < datetime(?, '-30 days')",
            (now_iso,),
        )
        db.commit()
        results["old_login_attempts"] = cursor.rowcount
    except Exception:
        results["old_login_attempts"] = 0

    # Expired sessions
    try:
        cursor = db.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (now_iso,),
        )
        db.commit()
        results["expired_sessions"] = cursor.rowcount
    except Exception:
        results["expired_sessions"] = 0

    logger.info("System cleanup results: %s", results)
    return {"status": "completed", "cleaned": results}


@router.get("/system/config")
async def system_config(request: Request) -> dict[str, Any]:
    """Show active configuration (no secrets)."""
    _require_admin(request)

    db_url = os.environ.get("PULSAR_DB_URL", "").strip()
    if db_url:
        # Mask credentials in URL
        db_backend = "postgresql" if "postgres" in db_url.lower() else "sqlite"
    else:
        db_backend = "sqlite"

    redis_url = os.environ.get("PULSAR_REDIS_URL", "").strip()

    return {
        "environment": os.environ.get("PULSAR_ENV", "development"),
        "database_backend": db_backend,
        "redis_configured": bool(redis_url),
        "s3_bucket": os.environ.get("PULSAR_S3_BUCKET", "") or None,
        "auth_enabled": os.environ.get("PULSAR_AUTH_ENABLED", "false").lower() == "true",
        "stand_mode": os.environ.get("PULSAR_STAND_MODE", "dev"),
        "cors_origins": os.environ.get("PULSAR_CORS_ORIGINS", ""),
    }
