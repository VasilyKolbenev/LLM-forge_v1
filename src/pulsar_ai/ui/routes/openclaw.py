"""API routes for OpenClaw runtime and NemoClaw sandbox management."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pulsar_ai.openclaw.adapter import OpenClawAdapter
from pulsar_ai.openclaw.nemoclaw import NemoClawManager, SandboxPolicy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openclaw", tags=["openclaw"])

# ── In-memory state ───────────────────────────────────────────────

_adapter: OpenClawAdapter | None = None
_manager: NemoClawManager | None = None


def _get_adapter() -> OpenClawAdapter:
    """Get or create the global OpenClaw adapter.

    Returns:
        OpenClawAdapter instance.
    """
    global _adapter
    if _adapter is None:
        _adapter = OpenClawAdapter()
    return _adapter


def _get_manager() -> NemoClawManager:
    """Get or create the global NemoClaw manager.

    Returns:
        NemoClawManager instance.
    """
    global _manager
    if _manager is None:
        _manager = NemoClawManager(_get_adapter())
    return _manager


# ── Pydantic models ───────────────────────────────────────────────


class SessionCreateRequest(BaseModel):
    """Request body for creating an OpenClaw session."""

    name: str = "default"
    model: str = "default"
    tools: list[str] = Field(default_factory=list)
    system_prompt: str = ""


class SessionRunRequest(BaseModel):
    """Request body for running an agent in a session."""

    input: str


class SandboxPolicyModel(BaseModel):
    """Pydantic model for sandbox policy."""

    allow_network: bool = False
    allowed_domains: list[str] = Field(default_factory=list)
    allow_file_write: bool = False
    allowed_paths: list[str] = Field(default_factory=list)
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    max_tokens: int = 4096


class DeploymentCreateRequest(BaseModel):
    """Request body for creating a NemoClaw deployment."""

    name: str = "default"
    model: str = "default"
    tools: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    policy: SandboxPolicyModel = Field(default_factory=SandboxPolicyModel)


class PolicyUpdateRequest(BaseModel):
    """Request body for updating a deployment policy."""

    policy: SandboxPolicyModel


# ── Session routes ────────────────────────────────────────────────


@router.get("/health")
def openclaw_health() -> dict[str, Any]:
    """Check OpenClaw runtime health."""
    adapter = _get_adapter()
    return adapter.health_check()


@router.get("/sessions")
def list_sessions(status: str | None = None) -> dict[str, Any]:
    """List all OpenClaw sessions."""
    adapter = _get_adapter()
    sessions = adapter.list_sessions(status=status)
    return {"sessions": [s.to_dict() for s in sessions]}


@router.post("/sessions")
def create_session(body: SessionCreateRequest) -> dict[str, Any]:
    """Create a new OpenClaw agent session."""
    adapter = _get_adapter()
    config = {
        "name": body.name,
        "model": body.model,
        "tools": body.tools,
        "system_prompt": body.system_prompt,
    }
    session = adapter.create_session(config)
    return session.to_dict()


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Get session detail."""
    adapter = _get_adapter()
    session = adapter.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.post("/sessions/{session_id}/run")
def run_session(session_id: str, body: SessionRunRequest) -> dict[str, Any]:
    """Run agent in a session."""
    adapter = _get_adapter()
    session = adapter.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return adapter.run(session_id, body.input)


@router.delete("/sessions/{session_id}")
def stop_session(session_id: str) -> dict[str, Any]:
    """Stop and cleanup a session."""
    adapter = _get_adapter()
    success = adapter.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop session")
    return {"status": "stopped", "session_id": session_id}


@router.get("/sessions/{session_id}/trace")
def get_session_trace(session_id: str) -> dict[str, Any]:
    """Get execution trace from a session."""
    adapter = _get_adapter()
    trace = adapter.get_trace(session_id)
    return {"session_id": session_id, "trace": trace}


# ── Deployment routes ─────────────────────────────────────────────


@router.get("/deployments")
def list_deployments() -> dict[str, Any]:
    """List all NemoClaw deployments."""
    manager = _get_manager()
    deployments = manager.list_deployments()
    return {"deployments": [d.to_dict() for d in deployments]}


@router.post("/deployments")
def create_deployment(body: DeploymentCreateRequest) -> dict[str, Any]:
    """Deploy agent with sandbox policy."""
    manager = _get_manager()
    agent_config = {
        "name": body.name,
        "model": body.model,
        "tools": body.tools,
        "system_prompt": body.system_prompt,
    }
    policy = SandboxPolicy(
        allow_network=body.policy.allow_network,
        allowed_domains=body.policy.allowed_domains,
        allow_file_write=body.policy.allow_file_write,
        allowed_paths=body.policy.allowed_paths,
        max_memory_mb=body.policy.max_memory_mb,
        max_cpu_seconds=body.policy.max_cpu_seconds,
        max_tokens=body.policy.max_tokens,
    )
    deployment = manager.deploy(agent_config, policy)
    return deployment.to_dict()


@router.get("/deployments/{deployment_id}")
def get_deployment(deployment_id: str) -> dict[str, Any]:
    """Get deployment detail."""
    manager = _get_manager()
    deployment = manager.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment.to_dict()


@router.delete("/deployments/{deployment_id}")
def stop_deployment(deployment_id: str) -> dict[str, Any]:
    """Stop a NemoClaw deployment."""
    manager = _get_manager()
    success = manager.stop_deployment(deployment_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop deployment")
    return {"status": "stopped", "deployment_id": deployment_id}


@router.put("/deployments/{deployment_id}/policy")
def update_deployment_policy(
    deployment_id: str,
    body: PolicyUpdateRequest,
) -> dict[str, Any]:
    """Update sandbox policy for a deployment."""
    manager = _get_manager()
    policy = SandboxPolicy(
        allow_network=body.policy.allow_network,
        allowed_domains=body.policy.allowed_domains,
        allow_file_write=body.policy.allow_file_write,
        allowed_paths=body.policy.allowed_paths,
        max_memory_mb=body.policy.max_memory_mb,
        max_cpu_seconds=body.policy.max_cpu_seconds,
        max_tokens=body.policy.max_tokens,
    )
    success = manager.update_policy(deployment_id, policy)
    if not success:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment = manager.get_deployment(deployment_id)
    return deployment.to_dict() if deployment else {"status": "updated"}
