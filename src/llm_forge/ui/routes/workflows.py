"""API routes for visual workflow management."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_forge.ui.workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])

_store = WorkflowStore()


class SaveWorkflowRequest(BaseModel):
    name: str
    nodes: list[dict]
    edges: list[dict]
    workflow_id: str | None = None


@router.get("")
async def list_workflows() -> list[dict]:
    """List all saved workflows."""
    return _store.list_all()


@router.post("")
async def save_workflow(req: SaveWorkflowRequest) -> dict:
    """Save or update a workflow."""
    return _store.save(
        name=req.name,
        nodes=req.nodes,
        edges=req.edges,
        workflow_id=req.workflow_id,
    )


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    """Get a single workflow by ID."""
    wf = _store.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict:
    """Delete a workflow."""
    if _store.delete(workflow_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Workflow not found")


@router.post("/{workflow_id}/run")
async def run_workflow(workflow_id: str) -> dict:
    """Convert workflow to pipeline config and mark as run.

    Returns the pipeline config (actual execution is handled by the caller).
    """
    config = _store.to_pipeline_config(workflow_id)
    if not config:
        raise HTTPException(status_code=404, detail="Workflow not found")
    _store.mark_run(workflow_id)
    return {"status": "started", "pipeline_config": config}


@router.get("/{workflow_id}/config")
async def get_workflow_config(workflow_id: str) -> dict:
    """Get the pipeline config for a workflow without running it."""
    config = _store.to_pipeline_config(workflow_id)
    if not config:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return config
