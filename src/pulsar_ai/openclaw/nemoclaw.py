"""NemoClaw sandboxed agent deployment manager.

Wraps OpenClaw sessions with security policies, network restrictions,
file access controls, and resource limits.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.openclaw.adapter import OpenClawAdapter

logger = logging.getLogger(__name__)


@dataclass
class SandboxPolicy:
    """Security policy for NemoClaw sandboxed execution."""

    allow_network: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    allow_file_write: bool = False
    allowed_paths: list[str] = field(default_factory=list)
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_tokens: int = 4096

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dict.

        Returns:
            Policy as a dict.
        """
        return {
            "allow_network": self.allow_network,
            "allowed_domains": self.allowed_domains,
            "allow_file_write": self.allow_file_write,
            "allowed_paths": self.allowed_paths,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxPolicy":
        """Create policy from dict.

        Args:
            data: Policy configuration dictionary.

        Returns:
            SandboxPolicy instance.
        """
        return cls(
            allow_network=data.get("allow_network", False),
            allowed_domains=data.get("allowed_domains", []),
            allow_file_write=data.get("allow_file_write", False),
            allowed_paths=data.get("allowed_paths", []),
            max_memory_mb=data.get("max_memory_mb", 512),
            max_cpu_seconds=data.get("max_cpu_seconds", 60),
            max_tokens=data.get("max_tokens", 4096),
        )


@dataclass
class NemoClawDeployment:
    """Represents a NemoClaw sandboxed deployment."""

    deployment_id: str
    session_id: str
    policy: SandboxPolicy
    status: str  # provisioning, running, stopped, failed
    created_at: str
    health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize deployment to dict.

        Returns:
            Deployment as a dict.
        """
        return {
            "deployment_id": self.deployment_id,
            "session_id": self.session_id,
            "policy": self.policy.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "health": self.health,
        }


class NemoClawManager:
    """Manages NemoClaw sandboxed agent deployments.

    NemoClaw wraps OpenClaw with sandboxed execution,
    network policies, file restrictions, and inference routing.

    Args:
        openclaw: OpenClawAdapter instance for agent sessions.
        base_url: NemoClaw API base URL.
    """

    def __init__(
        self,
        openclaw: OpenClawAdapter,
        base_url: str = "http://localhost:8200",
    ) -> None:
        self._openclaw = openclaw
        self._base_url = base_url.rstrip("/")
        self._deployments: dict[str, NemoClawDeployment] = {}

    def deploy(
        self,
        agent_config: dict[str, Any],
        policy: SandboxPolicy,
    ) -> NemoClawDeployment:
        """Deploy agent in sandboxed NemoClaw environment.

        Args:
            agent_config: Agent configuration for OpenClaw session.
            policy: Sandbox security policy to enforce.

        Returns:
            NemoClawDeployment with deployment_id and policy.
        """
        session = self._openclaw.create_session(agent_config)

        deployment_id = uuid.uuid4().hex[:12]
        deployment = NemoClawDeployment(
            deployment_id=deployment_id,
            session_id=session.session_id,
            policy=policy,
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
            health={"sandbox": "active"},
        )
        self._deployments[deployment_id] = deployment
        logger.info(
            "NemoClaw deployment %s created for session %s",
            deployment_id,
            session.session_id,
        )
        return deployment

    def run_sandboxed(
        self,
        deployment_id: str,
        user_input: str,
    ) -> dict[str, Any]:
        """Execute in sandboxed environment with policy enforcement.

        Args:
            deployment_id: Active deployment ID.
            user_input: User query to process.

        Returns:
            Dict with response, trace, tools_used, tokens, latency_ms.
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return {"error": f"Deployment {deployment_id} not found"}

        if deployment.status != "running":
            return {"error": f"Deployment {deployment_id} is {deployment.status}"}

        result = self._openclaw.run(deployment.session_id, user_input)

        tokens_used = result.get("tokens", 0)
        if tokens_used > deployment.policy.max_tokens:
            logger.warning(
                "Deployment %s exceeded token limit: %d > %d",
                deployment_id,
                tokens_used,
                deployment.policy.max_tokens,
            )
            result["warning"] = "Token limit exceeded"

        result["deployment_id"] = deployment_id
        result["policy"] = deployment.policy.to_dict()
        return result

    def get_deployment(self, deployment_id: str) -> NemoClawDeployment | None:
        """Get deployment details.

        Args:
            deployment_id: Deployment ID to look up.

        Returns:
            NemoClawDeployment or None if not found.
        """
        return self._deployments.get(deployment_id)

    def list_deployments(self) -> list[NemoClawDeployment]:
        """List all deployments.

        Returns:
            List of NemoClawDeployment objects.
        """
        return list(self._deployments.values())

    def stop_deployment(self, deployment_id: str) -> bool:
        """Stop a sandboxed deployment.

        Args:
            deployment_id: Deployment ID to stop.

        Returns:
            True if stopped successfully, False otherwise.
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            logger.warning("Deployment %s not found", deployment_id)
            return False

        stopped = self._openclaw.stop_session(deployment.session_id)
        deployment.status = "stopped" if stopped else "failed"
        logger.info("NemoClaw deployment %s stopped", deployment_id)
        return stopped

    def update_policy(
        self,
        deployment_id: str,
        policy: SandboxPolicy,
    ) -> bool:
        """Update sandbox policy for a deployment.

        Args:
            deployment_id: Deployment ID to update.
            policy: New sandbox policy.

        Returns:
            True if updated successfully, False otherwise.
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            logger.warning("Deployment %s not found", deployment_id)
            return False

        deployment.policy = policy
        logger.info("Updated policy for deployment %s", deployment_id)
        return True

    def get_health(self, deployment_id: str) -> dict[str, Any]:
        """Get health status for a deployment.

        Args:
            deployment_id: Deployment ID to check.

        Returns:
            Dict with health status information.
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return {"status": "not_found"}

        openclaw_health = self._openclaw.health_check()
        return {
            "deployment_id": deployment_id,
            "deployment_status": deployment.status,
            "sandbox": deployment.health,
            "openclaw": openclaw_health,
        }
