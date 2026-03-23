"""API key and JWT authentication middleware for Pulsar AI UI.

Provides:
- ApiKeyStore: SQLite-backed hashed key storage with audit trail
- ApiKeyMiddleware: FastAPI middleware for Bearer token auth
- JWTAuthMiddleware: JWT-based user authentication
- DemoModeMiddleware: Read-only mode for investor demos

Enable auth via PULSAR_AUTH_ENABLED=true.
Enable demo via PULSAR_STAND_MODE=demo.
"""

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from pulsar_ai.storage.database import Database, get_database
from pulsar_ai.ui.jwt_utils import verify_token

_ANONYMOUS_USER: dict = {"id": "__anonymous__", "email": "", "role": "admin", "name": "Anonymous"}


def _is_auth_enabled() -> bool:
    """Check whether JWT/API-key auth is turned on."""
    return os.getenv("PULSAR_AUTH_ENABLED", "").lower() in ("1", "true", "yes")


def get_current_user(request: Request) -> dict:
    """Extract the authenticated user from request state.

    When auth is disabled returns a sentinel anonymous user with admin role
    so that all data is visible (backward-compatible single-user mode).

    Args:
        request: FastAPI request.

    Returns:
        User dict with at least ``id``, ``email``, ``role`` keys.

    Raises:
        HTTPException: 401 if auth is enabled but no user is attached.
    """
    user = getattr(request.state, "user", None)
    if user:
        return user
    if not _is_auth_enabled():
        return _ANONYMOUS_USER
    raise HTTPException(status_code=401, detail="Authentication required")


def get_user_id(request: Request) -> str:
    """Shortcut — return only the user ID string."""
    return get_current_user(request)["id"]


def get_scoped_user_id(request: Request) -> str | None:
    """Return user_id for data scoping, or None for admins (see all).

    Admins get ``None`` which stores interpret as "skip user_id filter".
    Regular users get their actual ID for strict isolation.
    """
    user = get_current_user(request)
    if user.get("role") == "admin":
        return None
    return user["id"]

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = frozenset(
    {
        "/api/v1/health",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v1/auth/mfa/verify",
        "/api/v1/auth/oidc/authorize",
        "/api/v1/auth/oidc/callback",
        "/api/v1/auth/oidc/config",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


class ApiKeyStore:
    """Manages hashed API keys in SQLite with audit trail.

    Keys are stored as SHA-256 hashes — plaintext is only shown once
    at generation time.  Every significant action (create, verify, revoke,
    auth failure) is logged to the ``api_key_events`` table.

    Args:
        db: Database instance.  Uses the module singleton when *None*.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_database()

    # ── Audit helpers ─────────────────────────────────────────────

    def _log_event(self, key_id: str, event_type: str, ip_address: str = "") -> None:
        """Write an audit event to api_key_events."""
        self._db.execute(
            "INSERT INTO api_key_events (key_id, event_type, timestamp, ip_address) "
            "VALUES (?, ?, ?, ?)",
            (key_id, event_type, datetime.now().isoformat(), ip_address),
        )

    def get_events(
        self,
        key_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit events with optional filters.

        Args:
            key_id: Filter by key ID.
            event_type: Filter by event type.
            limit: Max rows to return.

        Returns:
            List of event dicts, newest first.
        """
        clauses: list[str] = []
        params: list[str | int] = []
        if key_id:
            clauses.append("key_id = ?")
            params.append(key_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._db.fetch_all(
            f"SELECT * FROM api_key_events {where} ORDER BY id DESC LIMIT ?",
            tuple(params),
        )
        return [dict(r) for r in rows]

    # ── Key CRUD ──────────────────────────────────────────────────

    def generate_key(self, name: str = "default", user_id: str = "") -> str:
        """Generate a new API key, store its hash, return plaintext.

        Args:
            name: Human-readable name for the key.
            user_id: Owner user ID for key scoping.

        Returns:
            Plaintext API key (only shown once).
        """
        raw_key = f"pulsar_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = str(uuid.uuid4())[:8]
        now_iso = datetime.now().isoformat()

        self._db.execute(
            """
            INSERT INTO api_keys (id, name, key_hash, created_at, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key_id, name, key_hash, now_iso, user_id),
        )
        self._log_event(key_id, "created")
        self._db.commit()
        logger.info("Generated API key '%s'", name)
        return raw_key

    def verify(self, key: str, ip_address: str = "") -> bool:
        """Check if a key matches any stored non-revoked hash.

        Args:
            key: Plaintext API key to verify.
            ip_address: Request origin for audit trail.

        Returns:
            True if the key is valid.
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        row = self._db.fetch_one(
            "SELECT id FROM api_keys WHERE key_hash = ? AND revoked = 0",
            (key_hash,),
        )
        if row:
            self._db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"]),
            )
            self._log_event(row["id"], "verified", ip_address)
            self._db.commit()
            return True

        # Log failed attempt with empty key_id
        self._log_event("", "auth_failed", ip_address)
        self._db.commit()
        return False

    def list_keys(self, user_id: str | None = None) -> list[dict]:
        """List key metadata (names only, no hashes).

        Args:
            user_id: When provided, only return keys owned by this user.

        Returns:
            List of dicts with 'name' field.
        """
        query = "SELECT name FROM api_keys WHERE revoked = 0"
        params: tuple = ()
        if user_id is not None:
            query += " AND user_id = ?"
            params = (user_id,)
        rows = self._db.fetch_all(query, params)
        return [{"name": r["name"]} for r in rows]

    def revoke(self, name: str, user_id: str | None = None) -> bool:
        """Revoke all keys with a given name.

        Args:
            name: Key name to revoke.
            user_id: When provided, only revoke keys owned by this user.

        Returns:
            True if any keys were revoked.
        """
        # Find key IDs before revoking (for audit trail)
        find_query = "SELECT id FROM api_keys WHERE name = ? AND revoked = 0"
        find_params: tuple = (name,)
        if user_id is not None:
            find_query += " AND user_id = ?"
            find_params = (name, user_id)
        rows = self._db.fetch_all(find_query, find_params)

        now_iso = datetime.now().isoformat()
        update_query = (
            "UPDATE api_keys SET revoked = 1, revoked_at = ? "
            "WHERE name = ? AND revoked = 0"
        )
        update_params: tuple = (now_iso, name)
        if user_id is not None:
            update_query += " AND user_id = ?"
            update_params = (now_iso, name, user_id)
        cursor = self._db.execute(update_query, update_params)
        for row in rows:
            self._log_event(row["id"], "revoked")
        self._db.commit()
        if cursor.rowcount > 0:
            logger.info("Revoked API key '%s'", name)
        return cursor.rowcount > 0


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for JWT-based user authentication.

    Validates ``Authorization: Bearer <jwt>`` headers for API routes.
    On success, attaches user info to ``request.state.user``.
    Falls through to ``ApiKeyMiddleware`` if token is not a valid JWT.

    Args:
        app: FastAPI/Starlette application.
        enabled: Whether JWT auth is active.
    """

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        """Attempt JWT auth, set request.state.user on success."""
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        # Skip for public endpoints and non-API routes
        if path in PUBLIC_PATHS or not path.startswith("/api/"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            # verify_token checks expiry, type, and blacklist status.
            # Returns None for invalid/expired/blacklisted tokens AND
            # non-JWT strings (API keys), so we pass through on None
            # to let ApiKeyMiddleware handle fallback auth.
            payload = verify_token(token, expected_type="access")
            if payload:
                # Attach user info for downstream routes
                request.state.user = {
                    "id": payload["sub"],
                    "email": payload.get("email", ""),
                    "role": payload.get("role", "user"),
                }

        # Always pass through — ApiKeyMiddleware handles fallback auth
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that validates Authorization: Bearer tokens.

    Skips authentication for public paths and non-API routes (static files).
    Also skips if JWTAuthMiddleware already authenticated the user.
    Can be disabled entirely via the enabled flag.

    Args:
        app: FastAPI/Starlette application.
        key_store: ApiKeyStore instance.
        enabled: Whether auth is active.
    """

    def __init__(self, app, key_store: ApiKeyStore, enabled: bool = True):
        super().__init__(app)
        self.key_store = key_store
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        """Process request, checking auth for API routes."""
        if not self.enabled:
            return await call_next(request)

        path = request.url.path

        # Skip auth for public endpoints
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for non-API routes (static files, frontend)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Skip if JWT middleware already authenticated
        if getattr(request.state, "user", None):
            return await call_next(request)

        ip = request.client.host if request.client else ""

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
            )

        token = auth[7:]
        if not self.key_store.verify(token, ip_address=ip):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)


# Paths always allowed in demo mode (reads + chat)
_DEMO_SAFE_PATHS = frozenset(
    {
        "/api/v1/health",
        "/api/v1/settings",
        "/assistant/chat",
        "/site/chat",
    }
)


class DemoModeMiddleware(BaseHTTPMiddleware):
    """Block mutating API requests when in demo mode.

    Allows GET requests and a whitelist of safe POST paths (chat).
    All other POST/PUT/DELETE on ``/api/`` are rejected with 403.

    Args:
        app: FastAPI/Starlette application.
    """

    async def dispatch(self, request: Request, call_next):
        """Reject writes in demo mode."""
        path = request.url.path
        method = request.method

        # Allow all reads
        if method == "GET":
            return await call_next(request)

        # Allow non-API routes
        if not path.startswith("/api/") and path not in ("/assistant/chat", "/site/chat"):
            return await call_next(request)

        # Allow whitelisted paths
        if path in _DEMO_SAFE_PATHS:
            return await call_next(request)

        return JSONResponse(
            status_code=403,
            content={"detail": "Demo mode: read-only. Mutations are disabled."},
        )
