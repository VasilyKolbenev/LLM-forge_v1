"""Generic OIDC client for Pulsar AI SSO integration.

Supports any OpenID Connect provider (Azure AD, Google, Okta, Auth0, etc.)
by discovering endpoints from the provider's ``/.well-known/openid-configuration``.

Configuration via environment variables:
- ``PULSAR_OIDC_PROVIDER_URL``: Base URL of the OIDC provider
  (e.g. ``https://login.microsoftonline.com/<tenant>/v2.0``)
- ``PULSAR_OIDC_CLIENT_ID``: OAuth2 client ID
- ``PULSAR_OIDC_CLIENT_SECRET``: OAuth2 client secret
- ``PULSAR_OIDC_REDIRECT_URI``: Callback URI registered with the provider
"""

import logging
import os
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

_DISCOVERY_SUFFIX = "/.well-known/openid-configuration"


def _env(key: str) -> str:
    """Read a required environment variable.

    Args:
        key: Environment variable name.

    Returns:
        Value string (may be empty if not set).
    """
    return os.environ.get(key, "").strip()


def is_oidc_enabled() -> bool:
    """Check whether OIDC SSO is configured.

    Returns:
        True if all required OIDC env vars are present.
    """
    return bool(
        _env("PULSAR_OIDC_PROVIDER_URL")
        and _env("PULSAR_OIDC_CLIENT_ID")
        and _env("PULSAR_OIDC_CLIENT_SECRET")
    )


def get_oidc_public_config() -> dict:
    """Return non-secret OIDC config for the frontend.

    Returns:
        Dict with ``enabled``, ``provider_url``, and ``provider_name``.
    """
    provider_url = _env("PULSAR_OIDC_PROVIDER_URL")
    # Derive a human-readable name from the URL
    name = "SSO"
    if "microsoft" in provider_url or "login.microsoftonline" in provider_url:
        name = "Microsoft"
    elif "google" in provider_url or "accounts.google" in provider_url:
        name = "Google"
    elif "okta" in provider_url:
        name = "Okta"
    elif "auth0" in provider_url:
        name = "Auth0"

    return {
        "enabled": is_oidc_enabled(),
        "provider_url": provider_url,
        "provider_name": name,
    }


# ── OIDC Provider ──────────────────────────────────────────────────


class OIDCProvider:
    """Generic OIDC client supporting Azure AD, Google, Okta, etc.

    Lazily discovers endpoints from the provider's well-known config.

    Args:
        provider_url: Base URL of the OIDC provider.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        redirect_uri: Registered callback URI.
    """

    def __init__(
        self,
        provider_url: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        self.provider_url = provider_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._discovery: Optional[dict] = None

    async def _discover(self) -> dict:
        """Fetch and cache the OpenID Connect discovery document.

        Returns:
            Discovery document dict.

        Raises:
            RuntimeError: If discovery fails.
        """
        if self._discovery is not None:
            return self._discovery

        url = self.provider_url + _DISCOVERY_SUFFIX
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OIDC discovery failed: {resp.status_code} from {url}"
                )
            self._discovery = resp.json()
            logger.info("OIDC discovery loaded from %s", url)
            return self._discovery

    async def get_authorization_url(self, state: str) -> str:
        """Build the authorization URL to redirect the user to the IdP.

        Args:
            state: Opaque state parameter for CSRF protection.

        Returns:
            Full authorization URL string.
        """
        disco = await self._discover()
        auth_endpoint = disco["authorization_endpoint"]

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "state": state,
            "nonce": secrets.token_urlsafe(16),
        }
        return f"{auth_endpoint}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict:
        """Exchange an authorization code for tokens and user info.

        Args:
            code: Authorization code from the callback.

        Returns:
            Dict with ``access_token``, ``id_token``, and user info fields
            (``email``, ``name``, ``sub``).

        Raises:
            RuntimeError: If the token exchange fails.
        """
        disco = await self._discover()
        token_endpoint = disco["token_endpoint"]

        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                token_endpoint,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                error_body = resp.text
                logger.error(
                    "OIDC token exchange failed: %d — %s",
                    resp.status_code,
                    error_body,
                )
                raise RuntimeError(
                    f"OIDC token exchange failed: {resp.status_code}"
                )
            token_data = resp.json()

        # Fetch user info from the userinfo endpoint
        user_info = await self.get_userinfo(token_data["access_token"])

        return {
            "access_token": token_data["access_token"],
            "id_token": token_data.get("id_token", ""),
            "email": user_info.get("email", ""),
            "name": user_info.get("name", ""),
            "sub": user_info.get("sub", ""),
        }

    async def get_userinfo(self, access_token: str) -> dict:
        """Fetch user profile from the OIDC userinfo endpoint.

        Args:
            access_token: Bearer token from the token exchange.

        Returns:
            User info dict with at least ``email``, ``name``, ``sub``.

        Raises:
            RuntimeError: If the userinfo request fails.
        """
        disco = await self._discover()
        userinfo_endpoint = disco.get("userinfo_endpoint")
        if not userinfo_endpoint:
            logger.warning("OIDC provider has no userinfo_endpoint")
            return {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OIDC userinfo failed: {resp.status_code}"
                )
            return resp.json()


# ── Singleton ───────────────────────────────────────────────────────

_provider: Optional[OIDCProvider] = None


def get_oidc_provider() -> Optional[OIDCProvider]:
    """Get the configured OIDC provider singleton.

    Returns:
        OIDCProvider instance, or None if OIDC is not configured.
    """
    global _provider  # noqa: PLW0603
    if _provider is not None:
        return _provider

    if not is_oidc_enabled():
        return None

    _provider = OIDCProvider(
        provider_url=_env("PULSAR_OIDC_PROVIDER_URL"),
        client_id=_env("PULSAR_OIDC_CLIENT_ID"),
        client_secret=_env("PULSAR_OIDC_CLIENT_SECRET"),
        redirect_uri=_env("PULSAR_OIDC_REDIRECT_URI"),
    )
    logger.info(
        "OIDC provider configured: %s", _env("PULSAR_OIDC_PROVIDER_URL")
    )
    return _provider
