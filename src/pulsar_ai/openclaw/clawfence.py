"""ClawFence governance layer for OpenClaw agent execution.

Provides policy enforcement, approval gates, and audit logging
specifically for OpenClaw sessions and NemoClaw deployments.
Builds on the P2.3 Enterprise Governance foundation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from pulsar_ai.storage.approval_store import ApprovalStore
from pulsar_ai.storage.audit_store import AuditStore

logger = logging.getLogger(__name__)

_approval_store: ApprovalStore | None = None
_audit_store: AuditStore | None = None


def _get_approval_store() -> ApprovalStore:
    global _approval_store
    if _approval_store is None:
        _approval_store = ApprovalStore()
    return _approval_store


def _get_audit_store() -> AuditStore:
    global _audit_store
    if _audit_store is None:
        _audit_store = AuditStore()
    return _audit_store


@dataclass
class ClawFencePolicy:
    """Governance policy for an OpenClaw session or deployment.

    Controls what an agent can do and whether approval is needed.
    """

    approval_required: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    network_access: bool = False
    file_write_access: bool = False
    audit_level: str = "standard"  # minimal, standard, verbose
    risk_level: str = "low"  # low, medium, high, critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_required": self.approval_required,
            "allowed_tools": self.allowed_tools,
            "max_tokens": self.max_tokens,
            "network_access": self.network_access,
            "file_write_access": self.file_write_access,
            "audit_level": self.audit_level,
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClawFencePolicy":
        return cls(
            approval_required=data.get("approval_required", False),
            allowed_tools=data.get("allowed_tools", []),
            max_tokens=data.get("max_tokens", 4096),
            network_access=data.get("network_access", False),
            file_write_access=data.get("file_write_access", False),
            audit_level=data.get("audit_level", "standard"),
            risk_level=data.get("risk_level", "low"),
        )

    @classmethod
    def from_sandbox_policy(cls, sandbox_policy: Any) -> "ClawFencePolicy":
        """Create ClawFencePolicy from a NemoClaw SandboxPolicy."""
        is_high_risk = (
            getattr(sandbox_policy, "allow_network", False)
            or getattr(sandbox_policy, "allow_file_write", False)
        )
        return cls(
            approval_required=is_high_risk,
            max_tokens=getattr(sandbox_policy, "max_tokens", 4096),
            network_access=getattr(sandbox_policy, "allow_network", False),
            file_write_access=getattr(sandbox_policy, "allow_file_write", False),
            risk_level="high" if is_high_risk else "low",
        )


def enforce_policy(
    session_config: dict[str, Any],
    policy: ClawFencePolicy,
) -> list[str]:
    """Validate session config against ClawFence policy.

    Args:
        session_config: Agent session configuration.
        policy: ClawFence policy to enforce.

    Returns:
        List of violation messages (empty if compliant).
    """
    violations: list[str] = []

    # Check allowed tools
    if policy.allowed_tools:
        requested_tools = session_config.get("tools", [])
        for tool in requested_tools:
            if tool not in policy.allowed_tools:
                violations.append(f"Tool '{tool}' not in allowed list")

    return violations


def check_approval_required(
    policy: ClawFencePolicy,
    session_id: str,
    workspace_id: str = "",
) -> dict[str, Any] | None:
    """Check if approval is required and create request if needed.

    Args:
        policy: ClawFence policy.
        session_id: Session being checked.
        workspace_id: Workspace context.

    Returns:
        Approval request dict if approval needed, None if no approval required.
    """
    if not policy.approval_required:
        return None

    store = _get_approval_store()

    # Check if already approved
    existing = store.list_requests(
        workspace_id=workspace_id,
        status="approved",
    )
    for req in existing:
        if req["resource_id"] == session_id and req["resource_type"] == "openclaw-session":
            logger.info("Session %s already approved: %s", session_id, req["id"])
            return None

    # Check if pending
    pending = store.list_requests(
        workspace_id=workspace_id,
        status="pending",
    )
    for req in pending:
        if req["resource_id"] == session_id and req["resource_type"] == "openclaw-session":
            logger.info("Session %s has pending approval: %s", session_id, req["id"])
            return req

    # Create new approval request
    request = store.create_request(
        resource_type="openclaw-session",
        resource_id=session_id,
        requester_id="",
        action="execute",
        workspace_id=workspace_id,
        reason=f"Session requires approval (risk_level={policy.risk_level})",
    )
    logger.info("Created approval request %s for session %s", request["id"], session_id)
    return request


def audit_session_event(
    event: str,
    session_id: str,
    details: dict[str, Any] | None = None,
    user_id: str = "",
    workspace_id: str = "",
) -> None:
    """Log a session governance event to audit trail.

    Args:
        event: Event type (create, run, stop, policy_violation, etc.)
        session_id: Session ID.
        details: Optional extra context.
        user_id: Acting user.
        workspace_id: Workspace context.
    """
    import json
    store = _get_audit_store()
    store.log(
        action=event,
        resource_type="openclaw-session",
        resource_id=session_id,
        user_id=user_id,
        workspace_id=workspace_id,
        details=json.dumps(details or {}),
    )


def log_policy_violation(
    session_id: str,
    violation: str,
    details: dict[str, Any] | None = None,
    workspace_id: str = "",
) -> None:
    """Log a policy violation to audit trail.

    Args:
        session_id: Session that violated policy.
        violation: Description of the violation.
        details: Additional context.
        workspace_id: Workspace context.
    """
    import json
    logger.warning("Policy violation in session %s: %s", session_id, violation)
    store = _get_audit_store()
    store.log(
        action="policy_violation",
        resource_type="openclaw-session",
        resource_id=session_id,
        workspace_id=workspace_id,
        details=json.dumps({"violation": violation, **(details or {})}),
    )
