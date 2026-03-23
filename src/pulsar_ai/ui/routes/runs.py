"""API routes for experiment run tracking."""

import logging

from fastapi import APIRouter, HTTPException, Request

from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
async def list_runs(
    request: Request,
    project: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List tracked experiment runs."""
    from pulsar_ai.tracking import list_runs

    uid = get_scoped_user_id(request)
    return list_runs(project=project, status=status, limit=limit, user_id=uid)


@router.get("/{run_id}")
async def get_run(request: Request, run_id: str) -> dict:
    """Get a specific run by ID."""
    from pulsar_ai.tracking import get_run

    uid = get_scoped_user_id(request)
    run = get_run(run_id, user_id=uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/compare")
async def compare_runs(run_ids: list[str]) -> dict:
    """Compare multiple runs side by side."""
    from pulsar_ai.tracking import compare_runs

    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 run IDs")
    return compare_runs(run_ids)
