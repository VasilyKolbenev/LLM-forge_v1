"""API routes for enterprise governance: approvals, audit, workspaces."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pulsar_ai.storage.approval_store import ApprovalStore
from pulsar_ai.storage.audit_store import AuditStore
from pulsar_ai.storage.workspace_store import WorkspaceStore
from pulsar_ai.ui.permissions import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["governance"])

_approvals = ApprovalStore()
_audit = AuditStore()
_workspaces = WorkspaceStore()


# ── Approvals ──────────────────────────────────────────────────


class CreateApprovalRequest(BaseModel):
    resource_type: str
    resource_id: str
    action: str = "execute"
    reason: str = ""


class ReviewApprovalRequest(BaseModel):
    decision: str  # "approved" or "rejected"
    note: str = ""


@router.get("/governance/approvals")
async def list_approvals(
    status: str | None = None,
    limit: int = 50,
    request: Request = None,
) -> dict[str, Any]:
    """List approval requests."""
    workspace_id = getattr(request.state, "workspace_id", "") if request else ""
    approvals = _approvals.list_requests(
        workspace_id=workspace_id, status=status, limit=limit,
    )
    pending = _approvals.get_pending_count(workspace_id or "")
    return {"approvals": approvals, "total": len(approvals), "pending_count": pending}


@router.post("/governance/approvals")
async def create_approval(
    body: CreateApprovalRequest,
    request: Request,
) -> dict[str, Any]:
    """Create an approval request."""
    user = getattr(request.state, "user", None) or {}
    workspace_id = getattr(request.state, "workspace_id", "") or ""

    result = _approvals.create_request(
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        requester_id=user.get("id", ""),
        action=body.action,
        workspace_id=workspace_id,
        reason=body.reason,
    )
    return result


@router.get("/governance/approvals/{request_id}")
async def get_approval(request_id: str) -> dict[str, Any]:
    """Get a single approval request."""
    req = _approvals.get_by_id(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return req


@router.post("/governance/approvals/{request_id}/review")
async def review_approval(
    request_id: str,
    body: ReviewApprovalRequest,
    request: Request,
    _=require_permission("approvals.review"),
) -> dict[str, Any]:
    """Approve or reject an approval request."""
    user = getattr(request.state, "user", None) or {}
    try:
        result = _approvals.review(
            request_id=request_id,
            approver_id=user.get("id", ""),
            decision=body.decision,
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


# ── Audit Logs ─────────────────────────────────────────────────


@router.get("/governance/audit")
async def query_audit(
    user_id: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
    request: Request = None,
) -> dict[str, Any]:
    """Query audit logs with filters."""
    workspace_id = getattr(request.state, "workspace_id", "") if request else ""
    logs = _audit.query(
        workspace_id=workspace_id or None,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )
    total = _audit.count(workspace_id=workspace_id or None, user_id=user_id)
    return {"logs": logs, "total": total}


# ── Workspaces ─────────────────────────────────────────────────


class CreateWorkspaceRequest(BaseModel):
    name: str


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "member"


class UpdateMemberRoleRequest(BaseModel):
    role: str


@router.get("/governance/workspaces")
async def list_workspaces(request: Request) -> dict[str, Any]:
    """List workspaces for the current user."""
    user = getattr(request.state, "user", None) or {}
    user_id = user.get("id")
    workspaces = _workspaces.list_workspaces(user_id=user_id)
    return {"workspaces": workspaces, "total": len(workspaces)}


@router.post("/governance/workspaces")
async def create_workspace(
    body: CreateWorkspaceRequest,
    request: Request,
    _=require_permission("workspaces.create"),
) -> dict[str, Any]:
    """Create a new workspace."""
    user = getattr(request.state, "user", None) or {}
    result = _workspaces.create_workspace(
        name=body.name, owner_id=user.get("id", ""),
    )
    return result


@router.get("/governance/workspaces/{workspace_id}/members")
async def get_workspace_members(workspace_id: str) -> dict[str, Any]:
    """List members of a workspace."""
    members = _workspaces.get_members(workspace_id)
    return {"members": members, "total": len(members)}


@router.post("/governance/workspaces/{workspace_id}/members")
async def add_workspace_member(
    workspace_id: str,
    body: AddMemberRequest,
    _=require_permission("workspaces.update"),
) -> dict[str, Any]:
    """Add a member to a workspace."""
    result = _workspaces.add_member(
        workspace_id=workspace_id,
        user_id=body.user_id,
        role=body.role,
    )
    return result


@router.delete("/governance/workspaces/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    _=require_permission("workspaces.update"),
) -> dict[str, Any]:
    """Remove a member from a workspace."""
    _workspaces.remove_member(workspace_id, user_id)
    return {"removed": user_id}


@router.put("/governance/workspaces/{workspace_id}/members/{user_id}")
async def update_member_role(
    workspace_id: str,
    user_id: str,
    body: UpdateMemberRoleRequest,
    _=require_permission("workspaces.update"),
) -> dict[str, Any]:
    """Change a member's role in a workspace."""
    _workspaces.update_member_role(workspace_id, user_id, body.role)
    return {"user_id": user_id, "role": body.role}
