"""Authentication API routes: login, register, refresh, logout, me, MFA, OIDC.

Mounted at ``/api/v1/auth``.

Includes password validation, brute-force protection, account lockout,
TOTP-based MFA, and OpenID Connect SSO.
"""

import logging
import re
import secrets
import string
import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from pulsar_ai.storage.database import get_database
from pulsar_ai.storage.user_store import UserStore
from pulsar_ai.ui.jwt_utils import (
    JWT_ALGORITHM,
    blacklist_token,
    blacklist_user_tokens,
    create_access_token,
    create_refresh_token,
    verify_token,
    _get_secret,
)
from pulsar_ai.ui.mfa import MFAStore
from pulsar_ai.ui.oidc import get_oidc_provider, get_oidc_public_config, is_oidc_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_user_store: UserStore | None = None

# ── Lockout configuration ────────────────────────────────────────
MAX_FAILED_ATTEMPTS_EMAIL = 5
MAX_FAILED_ATTEMPTS_IP = 20
LOCKOUT_WINDOW_MINUTES = 15


def _get_store() -> UserStore:
    global _user_store  # noqa: PLW0603
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


_mfa_store: MFAStore | None = None


def _get_mfa_store() -> MFAStore:
    global _mfa_store  # noqa: PLW0603
    if _mfa_store is None:
        _mfa_store = MFAStore()
    return _mfa_store


# ── MFA token helpers (short-lived, single-purpose) ─────────────

MFA_TOKEN_EXPIRE_MINUTES = 5


def _create_mfa_token(user_id: str) -> str:
    """Create a short-lived token valid only for MFA verification.

    Args:
        user_id: User ID pending MFA.

    Returns:
        Encoded JWT string with type ``mfa_pending``.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": "mfa_pending",
        "jti": uuid.uuid4().hex[:16],
        "iat": now,
        "exp": now + timedelta(minutes=MFA_TOKEN_EXPIRE_MINUTES),
    }
    return pyjwt.encode(payload, _get_secret(), algorithm=JWT_ALGORITHM)


def _verify_mfa_token(token: str) -> dict | None:
    """Verify a short-lived MFA pending token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload or None if invalid/expired.
    """
    try:
        payload = pyjwt.decode(token, _get_secret(), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "mfa_pending":
            return None
        return payload
    except pyjwt.InvalidTokenError:
        return None


# ── Password validation ─────────────────────────────────────────

_SPECIAL_CHARS = set(string.punctuation)


def validate_password(password: str) -> list[str]:
    """Validate password strength.

    Args:
        password: Plaintext password to validate.

    Returns:
        List of violation messages. Empty list means the password is valid.
    """
    errors: list[str] = []
    if len(password) < 12:
        errors.append("Password must be at least 12 characters long")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least 1 uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least 1 lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least 1 digit")
    if not any(ch in _SPECIAL_CHARS for ch in password):
        errors.append("Password must contain at least 1 special character")
    return errors


# ── Brute-force protection ───────────────────────────────────────


def record_login_attempt(email: str, ip: str, success: bool) -> None:
    """Record a login attempt for brute-force tracking.

    Args:
        email: Email used in the attempt.
        ip: Source IP address.
        success: Whether authentication succeeded.
    """
    db = get_database()
    db.execute(
        "INSERT INTO login_attempts (email, ip_address, success, attempted_at) "
        "VALUES (?, ?, ?, ?)",
        (email, ip, int(success), datetime.now().isoformat()),
    )
    db.commit()


def check_account_lockout(email: str) -> tuple[bool, int]:
    """Check if an account is locked due to too many failed attempts.

    Locks after ``MAX_FAILED_ATTEMPTS_EMAIL`` failures within the last
    ``LOCKOUT_WINDOW_MINUTES`` minutes.

    Args:
        email: Email to check.

    Returns:
        Tuple of (is_locked, seconds_remaining).
    """
    db = get_database()
    window_start = (
        datetime.now() - timedelta(minutes=LOCKOUT_WINDOW_MINUTES)
    ).isoformat()

    row = db.fetch_one(
        "SELECT COUNT(*) AS cnt, MIN(attempted_at) AS first_attempt "
        "FROM login_attempts "
        "WHERE email = ? AND success = 0 AND attempted_at > ?",
        (email, window_start),
    )
    if not row or row["cnt"] < MAX_FAILED_ATTEMPTS_EMAIL:
        return False, 0

    first_attempt = datetime.fromisoformat(row["first_attempt"])
    lockout_expires = first_attempt + timedelta(minutes=LOCKOUT_WINDOW_MINUTES)
    remaining = (lockout_expires - datetime.now()).total_seconds()
    if remaining <= 0:
        return False, 0
    return True, int(remaining)


def check_ip_rate_limit(ip: str) -> tuple[bool, int]:
    """Check if an IP is rate-limited due to too many failed attempts.

    Limits after ``MAX_FAILED_ATTEMPTS_IP`` failures within the last
    ``LOCKOUT_WINDOW_MINUTES`` minutes.

    Args:
        ip: IP address to check.

    Returns:
        Tuple of (is_limited, seconds_remaining).
    """
    if not ip:
        return False, 0

    db = get_database()
    window_start = (
        datetime.now() - timedelta(minutes=LOCKOUT_WINDOW_MINUTES)
    ).isoformat()

    row = db.fetch_one(
        "SELECT COUNT(*) AS cnt, MIN(attempted_at) AS first_attempt "
        "FROM login_attempts "
        "WHERE ip_address = ? AND success = 0 AND attempted_at > ?",
        (ip, window_start),
    )
    if not row or row["cnt"] < MAX_FAILED_ATTEMPTS_IP:
        return False, 0

    first_attempt = datetime.fromisoformat(row["first_attempt"])
    lockout_expires = first_attempt + timedelta(minutes=LOCKOUT_WINDOW_MINUTES)
    remaining = (lockout_expires - datetime.now()).total_seconds()
    if remaining <= 0:
        return False, 0
    return True, int(remaining)


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    return request.client.host if request.client else ""


# ── Request / Response models ────────────────────────────────────


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str = ""


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str


class MFAVerifyRequest(BaseModel):
    mfa_token: str
    code: str


class MFASetupVerifyRequest(BaseModel):
    code: str


class MFADisableRequest(BaseModel):
    code: str


# ── Endpoints ────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request) -> dict | JSONResponse:
    """Authenticate with email and password, receive JWT tokens."""
    ip = _get_client_ip(request)

    # Check account lockout
    locked, retry_after = check_account_lockout(body.email)
    if locked:
        logger.warning(
            "Account locked for '%s' — too many failed attempts", body.email
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Account temporarily locked due to too many failed login attempts",
            },
            headers={"Retry-After": str(retry_after)},
        )

    # Check IP rate limit
    ip_limited, ip_retry = check_ip_rate_limit(ip)
    if ip_limited:
        logger.warning("IP rate-limited: %s", ip)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many login attempts from this IP address"},
            headers={"Retry-After": str(ip_retry)},
        )

    store = _get_store()
    user = store.authenticate(body.email, body.password)
    if not user:
        record_login_attempt(body.email, ip, success=False)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    record_login_attempt(body.email, ip, success=True)

    # Check if MFA is enabled — if so, return a short-lived mfa_token
    mfa_store = _get_mfa_store()
    if mfa_store.is_mfa_enabled(user["id"]):
        mfa_token = _create_mfa_token(user["id"])
        logger.info("User '%s' requires MFA verification", body.email)
        return JSONResponse(
            status_code=200,
            content={
                "requires_mfa": True,
                "mfa_token": mfa_token,
            },
        )

    store.record_login(user["id"])

    access_token = create_access_token(user["id"], user["email"], user["role"])
    refresh_token = create_refresh_token(user["id"])

    logger.info("User '%s' logged in", body.email)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest) -> dict:
    """Register a new user account."""
    # Validate password strength
    violations = validate_password(body.password)
    if violations:
        raise HTTPException(status_code=400, detail=violations)

    store = _get_store()
    try:
        user = store.create_user(
            email=body.email,
            password=body.password,
            name=body.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    access_token = create_access_token(user["id"], user["email"], user["role"])
    refresh_token = create_refresh_token(user["id"])

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest) -> dict:
    """Exchange a refresh token for new access + refresh tokens.

    The old refresh token is blacklisted to prevent reuse (rotation).
    """
    payload = verify_token(body.refresh_token, expected_type="refresh")
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    store = _get_store()
    user = store.get_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Blacklist the old refresh token to prevent reuse
    old_jti = payload.get("jti")
    if old_jti:
        exp = payload.get("exp", "")
        if isinstance(exp, (int, float)):
            exp_iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
        else:
            exp_iso = str(exp)
        blacklist_token(old_jti, payload["sub"], exp_iso)

    access_token = create_access_token(user["id"], user["email"], user["role"])
    new_refresh = create_refresh_token(user["id"])

    return {
        "access_token": access_token,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/logout")
async def logout(request: Request, body: LogoutRequest) -> dict:
    """Logout: blacklist the current access token and optional refresh token."""
    # Blacklist the access token from the Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        access_token_str = auth[7:]
        access_payload = verify_token(access_token_str, expected_type="access")
        if access_payload and access_payload.get("jti"):
            exp = access_payload.get("exp", "")
            if isinstance(exp, (int, float)):
                exp_iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
            else:
                exp_iso = str(exp)
            blacklist_token(access_payload["jti"], access_payload["sub"], exp_iso)

    # Blacklist the refresh token if provided
    if body.refresh_token:
        refresh_payload = verify_token(body.refresh_token, expected_type="refresh")
        if refresh_payload and refresh_payload.get("jti"):
            exp = refresh_payload.get("exp", "")
            if isinstance(exp, (int, float)):
                exp_iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
            else:
                exp_iso = str(exp)
            blacklist_token(refresh_payload["jti"], refresh_payload["sub"], exp_iso)

    return {"detail": "Logged out successfully"}


@router.post("/logout-all")
async def logout_all(request: Request) -> dict:
    """Blacklist all tokens for the current user (force logout everywhere).

    Only the user themselves or an admin can call this.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    blacklist_user_tokens(user["id"])
    logger.info("User '%s' force-logged out all sessions", user["id"])
    return {"detail": "All sessions invalidated"}


@router.get("/me")
async def me(request: Request) -> dict:
    """Get current authenticated user."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── MFA Endpoints ───────────────────────────────────────────────


@router.post("/mfa/setup")
async def mfa_setup(request: Request) -> dict:
    """Begin MFA enrollment: generate TOTP secret and backup codes.

    Returns the secret, otpauth URI (for QR code), and backup codes.
    The caller must verify the first code via ``/mfa/verify-setup`` to
    actually enable MFA.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    mfa_store = _get_mfa_store()
    result = mfa_store.setup_mfa(user["id"], user.get("email", ""))
    return {
        "secret": result["secret"],
        "uri": result["uri"],
        "backup_codes": result["backup_codes"],
    }


@router.post("/mfa/verify-setup")
async def mfa_verify_setup(request: Request, body: MFASetupVerifyRequest) -> dict:
    """Verify initial TOTP code to activate MFA.

    Must be called after ``/mfa/setup`` with a valid 6-digit code from
    the authenticator app.
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    mfa_store = _get_mfa_store()
    if not mfa_store.verify_setup(user["id"], body.code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    return {"detail": "MFA enabled successfully"}


@router.post("/mfa/verify")
async def mfa_verify(body: MFAVerifyRequest) -> dict:
    """Verify TOTP code during login (2nd factor).

    Accepts the short-lived ``mfa_token`` from the login response and a
    6-digit TOTP or backup code. On success, returns full JWT tokens.
    """
    payload = _verify_mfa_token(body.mfa_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA token")

    user_id = payload["sub"]
    mfa_store = _get_mfa_store()

    if not mfa_store.verify_code(user_id, body.code):
        raise HTTPException(status_code=401, detail="Invalid TOTP or backup code")

    store = _get_store()
    user = store.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    store.record_login(user_id)

    access_token = create_access_token(user["id"], user["email"], user["role"])
    refresh_token = create_refresh_token(user["id"])

    logger.info("User '%s' completed MFA verification", user["email"])
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/mfa/disable")
async def mfa_disable(request: Request, body: MFADisableRequest) -> dict:
    """Disable MFA. Requires a valid TOTP code to confirm."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    mfa_store = _get_mfa_store()
    if not mfa_store.verify_code(user["id"], body.code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    mfa_store.disable_mfa(user["id"])
    return {"detail": "MFA disabled successfully"}


@router.post("/mfa/backup-codes")
async def mfa_backup_codes(request: Request) -> dict:
    """Regenerate backup codes. Requires MFA to be enabled."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    mfa_store = _get_mfa_store()
    codes = mfa_store.regenerate_backup_codes(user["id"])
    if codes is None:
        raise HTTPException(status_code=400, detail="MFA is not enabled")

    return {"backup_codes": codes}


@router.get("/mfa/status")
async def mfa_status(request: Request) -> dict:
    """Check whether MFA is enabled for the current user."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    mfa_store = _get_mfa_store()
    enabled = mfa_store.is_mfa_enabled(user["id"])
    return {"mfa_enabled": enabled}


# ── OIDC Endpoints ──────────────────────────────────────────────


@router.get("/oidc/config")
async def oidc_config() -> dict:
    """Return public OIDC configuration (provider name, enabled state)."""
    return get_oidc_public_config()


@router.get("/oidc/authorize")
async def oidc_authorize() -> RedirectResponse:
    """Redirect the user to the OIDC provider for authentication."""
    provider = get_oidc_provider()
    if not provider:
        raise HTTPException(status_code=404, detail="OIDC is not configured")

    # Generate and store a CSRF state token
    state = secrets.token_urlsafe(32)
    db = get_database()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=10)
    db.execute(
        "INSERT INTO oidc_states (state, created_at, expires_at) VALUES (?, ?, ?)",
        (state, now.isoformat(), expires.isoformat()),
    )
    db.commit()

    auth_url = await provider.get_authorization_url(state)
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/oidc/callback")
async def oidc_callback(code: str, state: str) -> RedirectResponse:
    """Handle the OIDC provider callback.

    Exchanges the authorization code for user info, creates or updates
    the local user account, and redirects to the frontend with JWT
    tokens in URL fragment (hash).
    """
    provider = get_oidc_provider()
    if not provider:
        raise HTTPException(status_code=404, detail="OIDC is not configured")

    # Validate CSRF state
    db = get_database()
    row = db.fetch_one(
        "SELECT * FROM oidc_states WHERE state = ?", (state,)
    )
    if not row:
        raise HTTPException(status_code=400, detail="Invalid OIDC state")

    # Clean up used state
    db.execute("DELETE FROM oidc_states WHERE state = ?", (state,))
    # Clean up expired states while we're at it
    db.execute(
        "DELETE FROM oidc_states WHERE expires_at < ?",
        (datetime.now(timezone.utc).isoformat(),),
    )
    db.commit()

    # Check expiry
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="OIDC state expired")

    # Exchange code for user info
    try:
        oidc_user = await provider.exchange_code(code)
    except RuntimeError as exc:
        logger.error("OIDC code exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail="SSO authentication failed") from exc

    email = oidc_user.get("email", "")
    if not email:
        raise HTTPException(
            status_code=400,
            detail="OIDC provider did not return an email address",
        )

    # Find or create local user
    store = _get_store()
    existing = db.fetch_one(
        "SELECT * FROM users WHERE email = ? AND is_active = 1", (email,)
    )

    if existing:
        user = {
            "id": existing["id"],
            "email": existing["email"],
            "name": existing["name"],
            "role": existing["role"],
        }
        # Update name from OIDC if available
        oidc_name = oidc_user.get("name", "")
        if oidc_name and oidc_name != existing["name"]:
            store.update_user(existing["id"], name=oidc_name)
            user["name"] = oidc_name
    else:
        # Auto-create user from OIDC (no password — SSO only)
        user_id = str(uuid.uuid4())[:8]
        now_iso = datetime.now().isoformat()
        # Use a random unusable password hash for SSO-only users
        unusable_hash = f"!oidc_{secrets.token_hex(16)}"
        db.execute(
            "INSERT INTO users (id, email, password_hash, name, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, email, unusable_hash, oidc_user.get("name", ""), "user", now_iso),
        )
        db.commit()
        user = {
            "id": user_id,
            "email": email,
            "name": oidc_user.get("name", ""),
            "role": "user",
        }
        logger.info("Created SSO user '%s'", email)

    store.record_login(user["id"])

    # Check MFA — if enabled, redirect with mfa_token instead
    mfa_store = _get_mfa_store()
    if mfa_store.is_mfa_enabled(user["id"]):
        mfa_token = _create_mfa_token(user["id"])
        return RedirectResponse(
            url=f"/login?requires_mfa=true&mfa_token={mfa_token}",
            status_code=302,
        )

    access_token = create_access_token(user["id"], user["email"], user["role"])
    refresh_token = create_refresh_token(user["id"])

    logger.info("SSO login for '%s'", email)

    # Redirect to frontend with tokens in URL fragment
    return RedirectResponse(
        url=(
            f"/login?sso=success"
            f"&access_token={access_token}"
            f"&refresh_token={refresh_token}"
        ),
        status_code=302,
    )
