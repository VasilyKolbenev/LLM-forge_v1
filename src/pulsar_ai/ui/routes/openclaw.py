"""API routes for OpenClaw runtime and NemoClaw sandbox management."""

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pulsar_ai.openclaw.clawfence import (
    ClawFencePolicy,
    audit_session_event,
    check_approval_required,
    enforce_policy,
    log_policy_violation,
)
from pulsar_ai.openclaw.nemoclaw import NemoClawManager, SandboxPolicy
from pulsar_ai.openclaw.runtime import get_runtime
from pulsar_ai.openclaw.trace_ingestion import OpenClawTraceIngester
from pulsar_ai.storage.trace_store import TraceStore
from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id, get_user_id
from pulsar_ai.ui.permissions import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openclaw", tags=["openclaw"])

# ── In-memory state ───────────────────────────────────────────────

_manager: NemoClawManager | None = None
_ingester: OpenClawTraceIngester | None = None


def _get_manager() -> NemoClawManager:
    """Get or create the global NemoClaw manager.

    Returns:
        NemoClawManager instance backed by the built-in runtime.
    """
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = NemoClawManager(get_runtime())
    return _manager


def _get_ingester() -> OpenClawTraceIngester:
    """Get or create the global trace ingester.

    Returns:
        OpenClawTraceIngester instance backed by the built-in runtime.
    """
    global _ingester  # noqa: PLW0603
    if _ingester is None:
        _ingester = OpenClawTraceIngester(get_runtime(), TraceStore())
    return _ingester


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
    runtime = get_runtime()
    return runtime.health()


@router.get("/sessions")
def list_sessions(request: Request, status: str | None = None) -> dict[str, Any]:
    """List all OpenClaw sessions."""
    runtime = get_runtime()
    sessions = runtime.list_sessions(status=status)
    return {"sessions": [asdict(s) for s in sessions]}


@router.post("/sessions")
def create_session(body: SessionCreateRequest) -> dict[str, Any]:
    """Create a new OpenClaw agent session with governance checks."""
    runtime = get_runtime()
    config = {
        "name": body.name,
        "model": body.model,
        "tools": body.tools,
        "system_prompt": body.system_prompt,
    }
    session = runtime.create_session(config)

    # ClawFence: audit session creation
    audit_session_event("create", session.session_id, details={"agent": body.name, "model": body.model})

    return asdict(session)


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Get session detail."""
    runtime = get_runtime()
    session = runtime.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return asdict(session)


@router.post("/sessions/{session_id}/run")
def run_session(session_id: str, body: SessionRunRequest, request: Request) -> dict[str, Any]:
    """Run agent in a session with ClawFence governance."""
    user_id = get_user_id(request)
    runtime = get_runtime()
    session = runtime.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ClawFence: check if approval is required
    policy = ClawFencePolicy.from_dict(session.metadata.get("clawfence_policy", {}))
    approval = check_approval_required(policy, session_id)
    if approval and approval.get("status") == "pending":
        return {
            "status": "pending_approval",
            "approval_id": approval["id"],
            "message": "This session requires approval before execution.",
        }

    # ClawFence: audit execution
    audit_session_event("run", session_id, details={"input_length": len(body.input)})

    result = runtime.run(session_id, body.input)

    # ClawFence: check token usage against policy
    tokens_used = result.get("tokens", 0)
    if policy.max_tokens > 0 and tokens_used > policy.max_tokens:
        log_policy_violation(
            session_id,
            f"Token limit exceeded: {tokens_used}/{policy.max_tokens}",
            details={"tokens_used": tokens_used, "limit": policy.max_tokens},
        )

    return result


@router.delete("/sessions/{session_id}")
def stop_session(session_id: str) -> dict[str, Any]:
    """Stop and cleanup a session."""
    runtime = get_runtime()
    success = runtime.stop_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    audit_session_event("stop", session_id)
    return {"status": "stopped", "session_id": session_id}


@router.get("/sessions/{session_id}/trace")
def get_session_trace(session_id: str) -> dict[str, Any]:
    """Get execution trace from a session."""
    runtime = get_runtime()
    trace = runtime.get_trace(session_id)
    return {"session_id": session_id, "trace": trace}


# ── Trace ingestion routes ────────────────────────────────────────


@router.post("/sessions/{session_id}/ingest")
def ingest_session_traces(session_id: str) -> dict[str, Any]:
    """Ingest traces from an OpenClaw session into TraceStore.

    Args:
        session_id: Session to ingest traces from.

    Returns:
        Dict with session_id and list of ingested trace_ids.
    """
    ingester = _get_ingester()
    trace_ids = ingester.ingest_session(session_id)
    return {"session_id": session_id, "trace_ids": trace_ids}


@router.post("/sessions/ingest-all")
def ingest_all_traces(status: str = "completed") -> dict[str, Any]:
    """Ingest traces from all sessions with given status.

    Args:
        status: Filter sessions by this status.

    Returns:
        Dict with sessions_processed and traces_ingested counts.
    """
    ingester = _get_ingester()
    return ingester.ingest_all_sessions(status=status)


# ── Deployment routes ─────────────────────────────────────────────


@router.get("/deployments")
def list_deployments() -> dict[str, Any]:
    """List all NemoClaw deployments."""
    manager = _get_manager()
    deployments = manager.list_deployments()
    return {"deployments": [d.to_dict() for d in deployments]}


@router.post("/deployments")
def create_deployment(body: DeploymentCreateRequest, _=require_permission("workflows.execute")) -> dict[str, Any]:
    """Deploy agent with sandbox policy and ClawFence governance."""
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
    audit_session_event(
        "deploy", deployment.session_id,
        details={"deployment_id": deployment.deployment_id, "agent": body.name},
    )
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
    audit_session_event(
        "policy_update", deployment_id,
        details={"new_policy": body.policy.model_dump()},
    )
    deployment = manager.get_deployment(deployment_id)
    return deployment.to_dict() if deployment else {"status": "updated"}


# ── Ollama model routes ──────────────────────────────────────────


@router.get("/models")
async def list_available_models() -> dict[str, Any]:
    """List models available in Ollama for agent sessions."""
    from pulsar_ai.agent.ollama_helper import get_ollama_status

    status = get_ollama_status()
    return status


@router.post("/models/pull")
async def pull_model(request: Request) -> dict[str, Any]:
    """Pull a model from Ollama registry."""
    from pulsar_ai.agent.ollama_helper import ensure_ollama_model

    body = await request.json()
    model_name = body.get("model", "qwen3.5:4b")
    success = ensure_ollama_model(model_name)
    return {"success": success, "model": model_name}
