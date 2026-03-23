"""TOTP-based MFA support for Pulsar AI.

Implements RFC 6238 TOTP using only Python stdlib (hmac, hashlib, struct,
time, base64).  No external ``pyotp`` dependency required.

Provides:
- TOTP secret generation (base32-encoded)
- ``otpauth://`` URI generation for QR codes
- 6-digit code verification with +/- 1 time-step tolerance
- One-time backup code generation
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import time
from datetime import datetime
from typing import Optional

from pulsar_ai.storage.database import Database, get_database

logger = logging.getLogger(__name__)

# TOTP parameters (RFC 6238 defaults)
TOTP_DIGITS = 6
TOTP_PERIOD = 30
TOTP_ALGORITHM = "sha1"
TOTP_TOLERANCE = 1  # accept codes from [-1, 0, +1] time steps


# ── TOTP Core ───────────────────────────────────────────────────────


def generate_totp_secret(length: int = 20) -> str:
    """Generate a random TOTP secret (base32-encoded).

    Args:
        length: Number of random bytes (default 20 = 160-bit).

    Returns:
        Base32-encoded secret string (no padding).
    """
    raw = os.urandom(length)
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def get_totp_uri(secret: str, email: str, issuer: str = "Pulsar AI") -> str:
    """Generate an ``otpauth://`` URI for QR code provisioning.

    Args:
        secret: Base32-encoded TOTP secret.
        email: User email (used as account label).
        issuer: Service name shown in authenticator apps.

    Returns:
        URI string suitable for encoding as a QR code.
    """
    from urllib.parse import quote

    label = f"{quote(issuer)}:{quote(email)}"
    params = (
        f"secret={secret}&issuer={quote(issuer)}"
        f"&algorithm=SHA1&digits={TOTP_DIGITS}&period={TOTP_PERIOD}"
    )
    return f"otpauth://totp/{label}?{params}"


def _hotp(secret_b32: str, counter: int) -> str:
    """Compute a single HOTP code (RFC 4226).

    Args:
        secret_b32: Base32-encoded secret (padding optional).
        counter: 8-byte counter value.

    Returns:
        Zero-padded HOTP code string.
    """
    # Decode base32 secret (add padding if needed)
    padded = secret_b32 + "=" * (-len(secret_b32) % 8)
    key = base64.b32decode(padded, casefold=True)

    # HMAC-SHA1 over the 8-byte big-endian counter
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()

    # Dynamic truncation
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset : offset + 4])[0]
    code_int &= 0x7FFFFFFF
    code_int %= 10**TOTP_DIGITS

    return str(code_int).zfill(TOTP_DIGITS)


def _current_totp(secret: str, time_offset: int = 0) -> str:
    """Compute TOTP code for the current (or offset) time step.

    Args:
        secret: Base32-encoded TOTP secret.
        time_offset: Number of time steps to shift (e.g. -1, 0, +1).

    Returns:
        6-digit TOTP code string.
    """
    counter = int(time.time()) // TOTP_PERIOD + time_offset
    return _hotp(secret, counter)


def verify_totp(secret: str, code: str) -> bool:
    """Verify a 6-digit TOTP code against the secret.

    Allows a tolerance window of +/- ``TOTP_TOLERANCE`` time steps to
    account for clock skew between server and authenticator app.

    Args:
        secret: Base32-encoded TOTP secret.
        code: 6-digit code to verify.

    Returns:
        True if the code matches any valid time step.
    """
    if not code or len(code) != TOTP_DIGITS or not code.isdigit():
        return False

    for offset in range(-TOTP_TOLERANCE, TOTP_TOLERANCE + 1):
        if hmac.compare_digest(_current_totp(secret, offset), code):
            return True
    return False


# ── Backup Codes ────────────────────────────────────────────────────


def generate_backup_codes(count: int = 10) -> list[str]:
    """Generate one-time backup codes.

    Each code is 8 hex characters (32 bits of entropy).

    Args:
        count: Number of backup codes to generate.

    Returns:
        List of unique backup code strings.
    """
    return [secrets.token_hex(4) for _ in range(count)]


def verify_backup_code(stored_codes_json: str, code: str) -> tuple[bool, str]:
    """Verify and consume a backup code.

    Args:
        stored_codes_json: JSON array of remaining backup codes.
        code: Code to verify.

    Returns:
        Tuple of (is_valid, updated_codes_json). If valid, the code
        is removed from the list.
    """
    try:
        codes: list[str] = json.loads(stored_codes_json)
    except (json.JSONDecodeError, TypeError):
        return False, stored_codes_json

    normalised = code.strip().lower()
    for i, stored in enumerate(codes):
        if hmac.compare_digest(stored.lower(), normalised):
            codes.pop(i)
            return True, json.dumps(codes)
    return False, stored_codes_json


# ── MFA Store (SQLite) ──────────────────────────────────────────────


class MFAStore:
    """Manages per-user MFA state in the ``user_mfa`` table.

    Args:
        db: Database instance. Falls back to the module singleton.
    """

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db or get_database()

    def get_mfa(self, user_id: str) -> Optional[dict]:
        """Fetch MFA record for a user.

        Args:
            user_id: User ID.

        Returns:
            MFA dict or None if not enrolled.
        """
        row = self._db.fetch_one(
            "SELECT * FROM user_mfa WHERE user_id = ?",
            (user_id,),
        )
        return dict(row) if row else None

    def is_mfa_enabled(self, user_id: str) -> bool:
        """Check whether MFA is enabled for a user.

        Args:
            user_id: User ID.

        Returns:
            True if MFA is active and verified.
        """
        mfa = self.get_mfa(user_id)
        return bool(mfa and mfa.get("enabled"))

    def setup_mfa(self, user_id: str, email: str) -> dict:
        """Begin MFA enrollment: generate secret + backup codes.

        If MFA was previously set up but not verified, the old record
        is replaced.

        Args:
            user_id: User ID.
            email: User email (for the ``otpauth://`` URI label).

        Returns:
            Dict with ``secret``, ``uri``, and ``backup_codes``.
        """
        secret = generate_totp_secret()
        uri = get_totp_uri(secret, email)
        backup_codes = generate_backup_codes()

        now_iso = datetime.now().isoformat()

        # Upsert: replace any existing un-verified setup
        self._db.execute(
            "INSERT OR REPLACE INTO user_mfa "
            "(user_id, totp_secret, backup_codes, enabled, verified_at, created_at) "
            "VALUES (?, ?, ?, 0, NULL, ?)",
            (user_id, secret, json.dumps(backup_codes), now_iso),
        )
        self._db.commit()

        logger.info("MFA setup initiated for user=%s", user_id)
        return {
            "secret": secret,
            "uri": uri,
            "backup_codes": backup_codes,
        }

    def verify_setup(self, user_id: str, code: str) -> bool:
        """Verify the initial TOTP code to activate MFA.

        Args:
            user_id: User ID.
            code: 6-digit TOTP code from authenticator app.

        Returns:
            True if code is valid and MFA is now enabled.
        """
        mfa = self.get_mfa(user_id)
        if not mfa:
            return False

        if not verify_totp(mfa["totp_secret"], code):
            return False

        now_iso = datetime.now().isoformat()
        self._db.execute(
            "UPDATE user_mfa SET enabled = 1, verified_at = ? WHERE user_id = ?",
            (now_iso, user_id),
        )
        self._db.commit()
        logger.info("MFA enabled for user=%s", user_id)
        return True

    def verify_code(self, user_id: str, code: str) -> bool:
        """Verify a TOTP code or backup code during login.

        If the code matches a backup code, it is consumed (one-time use).

        Args:
            user_id: User ID.
            code: 6-digit TOTP or backup code.

        Returns:
            True if the code is valid.
        """
        mfa = self.get_mfa(user_id)
        if not mfa or not mfa.get("enabled"):
            return False

        # Try TOTP first
        if verify_totp(mfa["totp_secret"], code):
            return True

        # Try backup code
        valid, updated_json = verify_backup_code(
            mfa.get("backup_codes", "[]"), code
        )
        if valid:
            self._db.execute(
                "UPDATE user_mfa SET backup_codes = ? WHERE user_id = ?",
                (updated_json, user_id),
            )
            self._db.commit()
            logger.info(
                "Backup code used for user=%s (remaining: %d)",
                user_id,
                len(json.loads(updated_json)),
            )
            return True

        return False

    def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for a user (deletes the record).

        Args:
            user_id: User ID.

        Returns:
            True if MFA was disabled.
        """
        cursor = self._db.execute(
            "DELETE FROM user_mfa WHERE user_id = ?", (user_id,)
        )
        self._db.commit()
        if cursor.rowcount > 0:
            logger.info("MFA disabled for user=%s", user_id)
        return cursor.rowcount > 0

    def regenerate_backup_codes(self, user_id: str) -> Optional[list[str]]:
        """Generate a new set of backup codes.

        Args:
            user_id: User ID.

        Returns:
            New backup codes list, or None if MFA is not enabled.
        """
        mfa = self.get_mfa(user_id)
        if not mfa or not mfa.get("enabled"):
            return None

        codes = generate_backup_codes()
        self._db.execute(
            "UPDATE user_mfa SET backup_codes = ? WHERE user_id = ?",
            (json.dumps(codes), user_id),
        )
        self._db.commit()
        logger.info("Backup codes regenerated for user=%s", user_id)
        return codes
