"""Security middleware for hardening HTTP responses.

Adds standard security headers, enforces request body size limits,
and applies request timeouts for non-streaming endpoints.
"""

import asyncio
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# 100 MB request body limit
MAX_REQUEST_BODY_BYTES: int = 100 * 1024 * 1024

# 5 minute timeout for regular requests
REQUEST_TIMEOUT_SECONDS: float = 300.0


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject standard security headers into every HTTP response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and attach security headers to response.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            Response with security headers applied.
        """
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'"
        )
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds a configurable limit.

    Args:
        app: ASGI application.
        max_bytes: Maximum allowed request body size in bytes.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check Content-Length and reject oversized requests.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            Response or 413 if the body is too large.
        """
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            logger.warning(
                "Request body too large: %s bytes (limit %s) from %s %s",
                content_length,
                self.max_bytes,
                request.method,
                request.url.path,
            )
            return Response(
                content="Request body too large",
                status_code=413,
                media_type="text/plain",
            )
        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Cancel requests that exceed a configurable timeout.

    Streaming responses (SSE / text/event-stream) are excluded because
    training and long-running endpoints rely on them.

    Args:
        app: ASGI application.
        timeout_seconds: Maximum seconds before the request is cancelled.
    """

    def __init__(
        self, app: ASGIApp, timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    ) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Wrap the downstream handler in an asyncio timeout.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            Response or 504 on timeout.
        """
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                "Request timed out after %ss: %s %s",
                self.timeout_seconds,
                request.method,
                request.url.path,
            )
            return Response(
                content="Request timed out",
                status_code=504,
                media_type="text/plain",
            )

        # Don't enforce timeout on streaming responses (SSE endpoints)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return response

        return response
