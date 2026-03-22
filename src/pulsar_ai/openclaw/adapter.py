"""OpenClaw runtime connector for Pulsar AI.

Provides session management, agent execution, and trace retrieval
through the OpenClaw agent runtime API.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class OpenClawSession:
    """Represents an active OpenClaw agent session."""

    session_id: str
    agent_name: str
    model: str
    status: str  # created, running, completed, failed
    created_at: str
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict.

        Returns:
            Session as a dict.
        """
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "tools": self.tools,
            "metadata": self.metadata,
        }


class OpenClawAdapter:
    """Connects Pulsar AI to OpenClaw agent runtime.

    OpenClaw provides real agent execution with tool use,
    multi-step workflows, and runtime behavior beyond pure inference.

    Args:
        base_url: OpenClaw API base URL.
        api_key: Optional API key for authentication.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._sessions: dict[str, OpenClawSession] = {}

    def create_session(self, agent_config: dict[str, Any]) -> OpenClawSession:
        """Create a new OpenClaw agent session.

        Args:
            agent_config: Agent configuration with keys: name, model,
                tools, system_prompt.

        Returns:
            Session object with session_id.
        """
        payload = {
            "agent_name": agent_config.get("name", "default"),
            "model": agent_config.get("model", "default"),
            "tools": agent_config.get("tools", []),
            "system_prompt": agent_config.get("system_prompt", ""),
        }

        try:
            data = self._request("POST", "/api/v1/sessions", json=payload)
        except ConnectionError:
            logger.warning("OpenClaw unavailable, creating local session stub")
            data = {
                "session_id": uuid.uuid4().hex[:12],
                "status": "created",
            }

        session = OpenClawSession(
            session_id=data.get("session_id", uuid.uuid4().hex[:12]),
            agent_name=payload["agent_name"],
            model=payload["model"],
            status=data.get("status", "created"),
            created_at=datetime.now(timezone.utc).isoformat(),
            tools=payload["tools"],
            metadata=agent_config.get("metadata", {}),
        )
        self._sessions[session.session_id] = session
        logger.info("Created OpenClaw session %s", session.session_id)
        return session

    def run(self, session_id: str, user_input: str) -> dict[str, Any]:
        """Execute agent action in a session.

        Args:
            session_id: Active session ID.
            user_input: User query to process.

        Returns:
            Dict with keys: response, trace, tools_used, tokens, latency_ms.
        """
        session = self._sessions.get(session_id)
        if session:
            session.status = "running"

        try:
            data = self._request(
                "POST",
                f"/api/v1/sessions/{session_id}/run",
                json={"input": user_input},
            )
        except ConnectionError:
            logger.error("OpenClaw unavailable for session %s", session_id)
            if session:
                session.status = "failed"
            return {
                "response": "",
                "trace": [],
                "tools_used": [],
                "tokens": 0,
                "latency_ms": 0,
                "error": "OpenClaw runtime unavailable",
            }

        if session:
            session.status = data.get("status", "completed")

        return {
            "response": data.get("response", ""),
            "trace": data.get("trace", []),
            "tools_used": data.get("tools_used", []),
            "tokens": data.get("tokens", 0),
            "latency_ms": data.get("latency_ms", 0),
        }

    def get_session(self, session_id: str) -> OpenClawSession | None:
        """Get session status and details.

        Args:
            session_id: Session ID to look up.

        Returns:
            Session object or None if not found.
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        try:
            data = self._request("GET", f"/api/v1/sessions/{session_id}")
            session = OpenClawSession(
                session_id=data.get("session_id", session_id),
                agent_name=data.get("agent_name", ""),
                model=data.get("model", ""),
                status=data.get("status", "unknown"),
                created_at=data.get("created_at", ""),
                tools=data.get("tools", []),
                metadata=data.get("metadata", {}),
            )
            self._sessions[session_id] = session
            return session
        except (ConnectionError, RuntimeError):
            logger.debug("Could not fetch session %s from OpenClaw", session_id)
            return None

    def list_sessions(self, status: str | None = None) -> list[OpenClawSession]:
        """List all sessions, optionally filtered by status.

        Args:
            status: Optional status filter (created, running, completed, failed).

        Returns:
            List of session objects.
        """
        try:
            params: dict[str, str] = {}
            if status:
                params["status"] = status
            data = self._request("GET", "/api/v1/sessions", params=params)
            sessions = []
            for item in data.get("sessions", []):
                s = OpenClawSession(
                    session_id=item.get("session_id", ""),
                    agent_name=item.get("agent_name", ""),
                    model=item.get("model", ""),
                    status=item.get("status", ""),
                    created_at=item.get("created_at", ""),
                    tools=item.get("tools", []),
                    metadata=item.get("metadata", {}),
                )
                self._sessions[s.session_id] = s
                sessions.append(s)
            return sessions
        except ConnectionError:
            logger.warning("OpenClaw unavailable, returning cached sessions")
            cached = list(self._sessions.values())
            if status:
                cached = [s for s in cached if s.status == status]
            return cached

    def stop_session(self, session_id: str) -> bool:
        """Stop and cleanup a session.

        Args:
            session_id: Session ID to stop.

        Returns:
            True if stopped successfully, False otherwise.
        """
        try:
            self._request("DELETE", f"/api/v1/sessions/{session_id}")
        except (ConnectionError, RuntimeError) as exc:
            logger.error("Failed to stop session %s: %s", session_id, exc)
            return False

        session = self._sessions.pop(session_id, None)
        if session:
            session.status = "completed"
        logger.info("Stopped OpenClaw session %s", session_id)
        return True

    def get_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Get execution trace from a session.

        Args:
            session_id: Session ID to get trace for.

        Returns:
            List of trace step dicts.
        """
        try:
            data = self._request("GET", f"/api/v1/sessions/{session_id}/trace")
            return data.get("trace", [])
        except (ConnectionError, RuntimeError) as exc:
            logger.error("Failed to get trace for %s: %s", session_id, exc)
            return []

    def health_check(self) -> dict[str, Any]:
        """Check OpenClaw runtime health.

        Returns:
            Dict with status and version info.
        """
        try:
            data = self._request("GET", "/api/v1/health")
            return {"status": "healthy", **data}
        except ConnectionError:
            return {"status": "unavailable", "error": "Cannot connect to OpenClaw"}
        except RuntimeError as exc:
            return {"status": "unhealthy", "error": str(exc)}

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make HTTP request to OpenClaw API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: API path (e.g. /api/v1/sessions).
            **kwargs: Additional arguments passed to requests.

        Returns:
            Parsed JSON response dict.

        Raises:
            ConnectionError: If the server is unreachable.
            RuntimeError: If the API returns a non-2xx status.
        """
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                timeout=30,
                **kwargs,
            )
        except requests.ConnectionError as exc:
            raise ConnectionError(f"Cannot connect to OpenClaw at {self._base_url}: {exc}") from exc
        except requests.Timeout as exc:
            raise ConnectionError(f"Request to OpenClaw timed out: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(f"OpenClaw API error {resp.status_code}: {resp.text}")

        if resp.status_code == 204:
            return {}
        return resp.json()
