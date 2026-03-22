"""Built-in OpenClaw-compatible agent runtime.

Uses Pulsar AI's ReAct agent as the execution engine,
exposed via OpenClaw API contract.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSession:
    """An active agent session in the runtime."""

    session_id: str
    agent_name: str
    model: str
    tools: list[str]
    system_prompt: str
    status: str = "created"  # created, running, completed, failed
    created_at: float = field(default_factory=time.time)
    trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenClawRuntime:
    """Built-in OpenClaw-compatible agent runtime.

    Wraps Pulsar AI's ReAct agent to provide session-based
    agent execution with trace collection.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, RuntimeSession] = {}

    def create_session(self, config: dict[str, Any]) -> RuntimeSession:
        """Create a new agent session.

        Args:
            config: Agent configuration with keys: name, model,
                tools, system_prompt.

        Returns:
            RuntimeSession instance.
        """
        session_id = uuid.uuid4().hex[:12]
        session = RuntimeSession(
            session_id=session_id,
            agent_name=config.get("name", "default-agent"),
            model=config.get("model", ""),
            tools=config.get("tools", []),
            system_prompt=config.get("system_prompt", "You are a helpful AI assistant."),
            status="created",
        )
        self._sessions[session_id] = session
        logger.info("Created session %s (agent: %s)", session_id, session.agent_name)
        return session

    def run(self, session_id: str, user_input: str) -> dict[str, Any]:
        """Execute agent action in a session.

        Uses Pulsar AI ReAct agent under the hood.

        Args:
            session_id: Active session ID.
            user_input: User query to process.

        Returns:
            Dict with keys: response, trace, tools_used, tokens, latency_ms.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.status = "running"
        start = time.time()

        try:
            from pulsar_ai.agent.base import BaseAgent

            agent_config: dict[str, Any] = {
                "agent": {
                    "name": session.agent_name,
                    "system_prompt": session.system_prompt,
                },
                "model": {
                    "base_url": "http://localhost:11434/v1",
                    "name": session.model or "llama3",
                },
            }

            agent = BaseAgent.from_config(agent_config)
            response = agent.run(user_input)
            trace = agent.trace

            latency_ms = int((time.time() - start) * 1000)
            tools_used = [step.get("tool", "") for step in trace if step.get("type") == "tool_call"]

            # Update session state
            session.trace.extend(trace)
            session.messages.append({"role": "user", "content": user_input})
            session.messages.append({"role": "assistant", "content": response})
            session.status = "running"  # still active for further queries

            result: dict[str, Any] = {
                "response": response,
                "trace": trace,
                "tools_used": tools_used,
                "tokens": getattr(agent, "_total_tokens", 0),
                "latency_ms": latency_ms,
                "session_id": session_id,
            }

            logger.info("Session %s: ran query (%dms)", session_id, latency_ms)
            return result

        except Exception:
            session.status = "failed"
            latency_ms = int((time.time() - start) * 1000)
            logger.exception("Session %s failed", session_id)
            return {
                "response": "",
                "trace": [],
                "tools_used": [],
                "tokens": 0,
                "latency_ms": latency_ms,
                "error": "Agent execution failed",
            }

    def get_session(self, session_id: str) -> RuntimeSession | None:
        """Get session by ID.

        Args:
            session_id: Session ID to look up.

        Returns:
            RuntimeSession or None if not found.
        """
        return self._sessions.get(session_id)

    def list_sessions(self, status: str | None = None) -> list[RuntimeSession]:
        """List all sessions, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of RuntimeSession objects.
        """
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def stop_session(self, session_id: str) -> bool:
        """Stop and mark a session as completed.

        Args:
            session_id: Session ID to stop.

        Returns:
            True if stopped, False if not found.
        """
        session = self._sessions.get(session_id)
        if session:
            session.status = "completed"
            return True
        return False

    def get_trace(self, session_id: str) -> list[dict[str, Any]]:
        """Get execution trace from a session.

        Args:
            session_id: Session ID to get trace for.

        Returns:
            List of trace step dicts.
        """
        session = self._sessions.get(session_id)
        return session.trace if session else []

    def health(self) -> dict[str, Any]:
        """Check runtime health.

        Returns:
            Dict with status, version, and session counts.
        """
        return {
            "status": "ok",
            "runtime": "pulsar-builtin",
            "version": "0.1.0",
            "active_sessions": len([s for s in self._sessions.values() if s.status == "running"]),
            "total_sessions": len(self._sessions),
        }


# Global singleton
_runtime: OpenClawRuntime | None = None


def get_runtime() -> OpenClawRuntime:
    """Get or create the global OpenClaw runtime singleton.

    Returns:
        OpenClawRuntime instance.
    """
    global _runtime  # noqa: PLW0603
    if _runtime is None:
        _runtime = OpenClawRuntime()
    return _runtime
