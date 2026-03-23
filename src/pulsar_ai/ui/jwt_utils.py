"""JWT token creation and verification for Pulsar AI.

Uses HS256 symmetric signing with a configurable secret.
Access tokens are short-lived (30 min), refresh tokens longer (7 days).
Supports token blacklisting for secure logout and revocation.
"""

import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from pulsar_ai.storage.database import get_database

logger = logging.getLogger(__name__)

# Configurable via environment
_JWT_SECRET: Optional[str] = None
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


def _get_secret() -> str:
    """Get the JWT secret, generating a random one if not configured."""
    global _JWT_SECRET  # noqa: PLW0603
    if _JWT_SECRET is None:
        _JWT_SECRET = os.environ.get("PULSAR_JWT_SECRET", "").strip()
        if not _JWT_SECRET:
            _JWT_SECRET = secrets.token_urlsafe(48)
            logger.warning(
                "PULSAR_JWT_SECRET not set — using random secret. "
                "Tokens will be invalidated on restart."
            )
    return _JWT_SECRET


def _generate_jti() -> str:
    """Generate a unique JWT ID (jti claim)."""
    return uuid.uuid4().hex[:16]


def create_access_token(
    user_id: str,
    email: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a short-lived access token.

    Args:
        user_id: User ID to encode in the token.
        email: User email.
        role: User role.
        expires_delta: Custom expiration. Defaults to 30 minutes.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "jti": _generate_jti(),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token.

    Args:
        user_id: User ID to encode.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": _generate_jti(),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def verify_token(token: str, expected_type: str = "access") -> Optional[dict]:
    """Verify and decode a JWT token.

    Checks signature, expiration, type, and blacklist status.

    Args:
        token: Encoded JWT string.
        expected_type: Expected token type (``access`` or ``refresh``).

    Returns:
        Decoded payload dict, or None if invalid/expired/blacklisted.
    """
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != expected_type:
            logger.debug(
                "Token type mismatch: expected %s, got %s",
                expected_type,
                payload.get("type"),
            )
            return None
        # Check blacklist if jti is present
        jti = payload.get("jti")
        if jti:
            iat_dt = payload.get("iat")
            iat_iso = ""
            if isinstance(iat_dt, (int, float)):
                iat_iso = datetime.fromtimestamp(iat_dt, tz=timezone.utc).isoformat()
            elif isinstance(iat_dt, datetime):
                iat_iso = iat_dt.isoformat()
            if is_token_blacklisted(jti, user_id=payload.get("sub", ""), iat=iat_iso):
                logger.debug("Token jti=%s is blacklisted", jti)
                return None
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.debug("Invalid token: %s", exc)
        return None


# ── Token Blacklist ──────────────────────────────────────────────────


def is_token_blacklisted(jti: str, user_id: str = "", iat: str = "") -> bool:
    """Check whether a token has been revoked.

    Checks both per-token blacklist (by jti) and user-level revocation
    (any ``all_*`` sentinel entry created after the token's ``iat``).

    Args:
        jti: JWT ID claim to check.
        user_id: Token owner, used for user-level revocation check.
        iat: ISO-8601 issued-at timestamp of the token.

    Returns:
        True if the token is blacklisted.
    """
    db = get_database()
    # Check exact jti match
    row = db.fetch_one(
        "SELECT 1 FROM token_blacklist WHERE jti = ?",
        (jti,),
    )
    if row is not None:
        return True
    # Check user-level revocation (sentinel entries created after token iat)
    if user_id and iat:
        row = db.fetch_one(
            "SELECT 1 FROM token_blacklist "
            "WHERE user_id = ? AND jti LIKE 'all_%' AND revoked_at > ?",
            (user_id, iat),
        )
        if row is not None:
            return True
    return False


def blacklist_token(jti: str, user_id: str, expires_at: str) -> None:
    """Add a single token to the blacklist.

    Args:
        jti: JWT ID claim of the token to revoke.
        user_id: Owner of the token.
        expires_at: ISO-8601 expiration timestamp (for cleanup).
    """
    db = get_database()
    now_iso = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT OR IGNORE INTO token_blacklist (jti, user_id, expires_at, revoked_at) "
        "VALUES (?, ?, ?, ?)",
        (jti, user_id, expires_at, now_iso),
    )
    db.commit()
    logger.info("Blacklisted token jti=%s for user=%s", jti, user_id)


def blacklist_user_tokens(user_id: str) -> None:
    """Blacklist all active tokens for a user (force logout everywhere).

    This inserts a sentinel row that downstream checks can use.
    Existing per-token rows are unaffected.

    Args:
        user_id: User whose tokens should be invalidated.
    """
    now = datetime.now(timezone.utc)
    # Insert a wildcard sentinel — expires far in the future so it persists
    sentinel_jti = f"all_{user_id}_{uuid.uuid4().hex[:8]}"
    far_future = (now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)).isoformat()
    db = get_database()
    db.execute(
        "INSERT OR IGNORE INTO token_blacklist (jti, user_id, expires_at, revoked_at) "
        "VALUES (?, ?, ?, ?)",
        (sentinel_jti, user_id, far_future, now.isoformat()),
    )
    db.commit()
    logger.info("Blacklisted all tokens for user=%s (sentinel=%s)", user_id, sentinel_jti)


def cleanup_expired_blacklist() -> int:
    """Remove blacklist entries whose tokens have already expired.

    Returns:
        Number of rows deleted.
    """
    db = get_database()
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        "DELETE FROM token_blacklist WHERE expires_at < ?",
        (now_iso,),
    )
    db.commit()
    deleted = cursor.rowcount
    if deleted:
        logger.info("Cleaned up %d expired blacklist entries", deleted)
    return deleted
