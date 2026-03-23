"""Audit middleware for automatic logging of state-changing API operations.

Captures POST/PUT/DELETE requests to /api/ paths and logs them
to the audit_logs table after successful responses.
"""

import logging
import re
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pulsar_ai.storage.audit_store import AuditStore

logger = logging.getLogger(__name__)

# Paths to skip (auth operations, health checks)
SKIP_PATTERNS = [
    re.compile(r"/api/v1/auth/"),
    re.compile(r"/api/v1/health"),
    re.compile(r"/api/v1/metrics"),
    re.compile(r"/docs"),
    re.compile(r"/openapi"),
]

# Parse resource type and ID from URL path
# e.g., /api/v1/experiments/abc123 -> ("experiments", "abc123")
RESOURCE_PATTERN = re.compile(r"/api/v1/([a-z_-]+)(?:/([a-zA-Z0-9_-]+))?")


def _parse_resource(path: str) -> tuple[str, str]:
    """Extract resource type and ID from URL path."""
    match = RESOURCE_PATTERN.search(path)
    if match:
        return match.group(1), match.group(2) or ""
    return "unknown", ""


class AuditMiddleware(BaseHTTPMiddleware):
    """Automatically logs state-changing API operations.

    Only logs POST, PUT, DELETE requests to /api/ paths
    that return 2xx status codes.
    """

    def __init__(self, app: Any, enabled: bool = True) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._store = AuditStore()

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request and log mutations."""
        if not self._enabled:
            return await call_next(request)

        # Only log mutations
        if request.method not in ("POST", "PUT", "DELETE"):
            return await call_next(request)

        # Only log API paths
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        # Skip auth and system paths
        for pattern in SKIP_PATTERNS:
            if pattern.search(path):
                return await call_next(request)

        # Execute the request
        response = await call_next(request)

        # Only log successful operations
        if 200 <= response.status_code < 300:
            try:
                user = getattr(request.state, "user", None) or {}
                user_id = user.get("id", "") if isinstance(user, dict) else ""
                workspace_id = getattr(request.state, "workspace_id", "") or ""
                resource_type, resource_id = _parse_resource(path)
                ip_address = request.client.host if request.client else ""

                self._store.log(
                    action=request.method,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    ip_address=ip_address,
                )
            except Exception as exc:
                logger.warning("Audit log failed: %s", exc)

        return response
