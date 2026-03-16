"""API key authentication for Pulsar AI UI.

Provides:
- ApiKeyStore: SQLite-backed hashed key storage
- ApiKeyMiddleware: FastAPI middleware for Bearer token auth

Enable via PULSAR_AUTH_ENABLED=true environment variable.
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from pulsar_ai.storage.database import Database, get_database

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = frozenset({
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class ApiKeyStore:
    """Manages hashed API keys in SQLite.

    Keys are stored as SHA-256 hashes — plaintext is only shown once
    at generation time.

    Args:
        db: Database instance.  Uses the module singleton when *None*.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_database()

    def generate_key(self, name: str = "default") -> str:
        """Generate a new API key, store its hash, return plaintext.

        Args:
            name: Human-readable name for the key.

        Returns:
            Plaintext API key (only shown once).
        """
        raw_key = f"pulsar_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = str(uuid.uuid4())[:8]
        now_iso = datetime.now().isoformat()

        self._db.execute(
            """
            INSERT INTO api_keys (id, name, key_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key_id, name, key_hash, now_iso),
        )
        self._db.commit()
        logger.info("Generated API key '%s'", name)
        return raw_key

    def verify(self, key: str) -> bool:
        """Check if a key matches any stored non-revoked hash.

        Args:
            key: Plaintext API key to verify.

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
            self._db.commit()
        return row is not None

    def list_keys(self) -> list[dict]:
        """List key metadata (names only, no hashes).

        Returns:
            List of dicts with 'name' field.
        """
        rows = self._db.fetch_all(
            "SELECT name FROM api_keys WHERE revoked = 0"
        )
        return [{"name": r["name"]} for r in rows]

    def revoke(self, name: str) -> bool:
        """Revoke all keys with a given name.

        Args:
            name: Key name to revoke.

        Returns:
            True if any keys were revoked.
        """
        now_iso = datetime.now().isoformat()
        cursor = self._db.execute(
            """
            UPDATE api_keys
            SET revoked = 1, revoked_at = ?
            WHERE name = ? AND revoked = 0
            """,
            (now_iso, name),
        )
        self._db.commit()
        if cursor.rowcount > 0:
            logger.info("Revoked API key '%s'", name)
        return cursor.rowcount > 0


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that validates Authorization: Bearer tokens.

    Skips authentication for public paths and non-API routes (static files).
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

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
            )

        token = auth[7:]
        if not self.key_store.verify(token):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)
