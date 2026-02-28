"""API key authentication for llm-forge UI.

Provides:
- ApiKeyStore: JSON-based hashed key storage
- ApiKeyMiddleware: FastAPI middleware for Bearer token auth

Enable via FORGE_AUTH_ENABLED=true environment variable.
"""

import hashlib
import json
import logging
import secrets
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

DEFAULT_KEYS_PATH = Path("./data/api_keys.json")

# Paths that don't require authentication
PUBLIC_PATHS = frozenset({
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


class ApiKeyStore:
    """Manages hashed API keys in a JSON file.

    Keys are stored as SHA-256 hashes — plaintext is only shown once
    at generation time.

    Args:
        store_path: Path to the JSON key store file.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path or DEFAULT_KEYS_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._save({"keys": []})

    def generate_key(self, name: str = "default") -> str:
        """Generate a new API key, store its hash, return plaintext.

        Args:
            name: Human-readable name for the key.

        Returns:
            Plaintext API key (only shown once).
        """
        raw_key = f"forge_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        data = self._load()
        data["keys"].append({"name": name, "hash": key_hash})
        self._save(data)
        logger.info("Generated API key '%s'", name)
        return raw_key

    def verify(self, key: str) -> bool:
        """Check if a key matches any stored hash.

        Args:
            key: Plaintext API key to verify.

        Returns:
            True if the key is valid.
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        data = self._load()
        return any(k["hash"] == key_hash for k in data["keys"])

    def list_keys(self) -> list[dict]:
        """List key metadata (names only, no hashes).

        Returns:
            List of dicts with 'name' field.
        """
        return [{"name": k["name"]} for k in self._load()["keys"]]

    def revoke(self, name: str) -> bool:
        """Revoke all keys with a given name.

        Args:
            name: Key name to revoke.

        Returns:
            True if any keys were revoked.
        """
        data = self._load()
        original = len(data["keys"])
        data["keys"] = [k for k in data["keys"] if k["name"] != name]
        self._save(data)
        revoked = len(data["keys"]) < original
        if revoked:
            logger.info("Revoked API key '%s'", name)
        return revoked

    def _load(self) -> dict:
        """Load key store from disk."""
        with open(self.store_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        """Save key store to disk."""
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


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
