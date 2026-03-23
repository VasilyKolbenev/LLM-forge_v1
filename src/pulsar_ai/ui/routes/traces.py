"""API routes for agent trace operations.

Provides endpoints for listing, inspecting, and annotating traces,
as well as exporting training datasets from selected traces.
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pulsar_ai.storage.trace_store import TraceStore
from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])

_store = TraceStore()


# ── Request / response models ────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Body for adding feedback to a trace."""

    feedback_type: str = "thumbs"
    rating: float = 1.0
    reason: str = ""
    user_id: str = ""


class BuildDatasetRequest(BaseModel):
    """Body for building a training dataset from traces."""

    trace_ids: list[str]
    format: str = "sft"
    name: str = "traces-dataset"
    output_dir: str = "./data/generated"


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/traces")
async def list_traces(
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
    model_name: str | None = None,
    status: str | None = None,
    min_rating: float | None = None,
    has_feedback: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List traces with filters and pagination.

    Args:
        date_from: ISO date lower bound.
        date_to: ISO date upper bound.
        model_name: Filter by model.
        status: Filter by status.
        min_rating: Minimum average rating.
        has_feedback: Filter by feedback presence.
        limit: Page size.
        offset: Page offset.

    Returns:
        Dict with traces list, total count, limit, and offset.
    """
    user_id = get_scoped_user_id(request)
    traces = _store.list_traces(
        date_from=date_from or "",
        date_to=date_to or "",
        model_name=model_name or "",
        status=status or "",
        min_rating=min_rating,
        has_feedback=has_feedback,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )
    # Get total without pagination for the current filters
    all_traces = _store.list_traces(
        date_from=date_from or "",
        date_to=date_to or "",
        model_name=model_name or "",
        status=status or "",
        min_rating=min_rating,
        has_feedback=has_feedback,
        limit=10_000,
        offset=0,
        user_id=user_id,
    )
    return {
        "traces": traces,
        "total": len(all_traces),
        "limit": limit,
        "offset": offset,
    }


@router.get("/traces/stats")
async def get_trace_stats(request: Request, days: int = 30) -> dict[str, Any]:
    """Get trace statistics for dashboard.

    Args:
        days: How many days back to aggregate.

    Returns:
        Statistics dict with totals, averages, and breakdowns.
    """
    user_id = get_scoped_user_id(request)
    return _store.get_stats(days=days, user_id=user_id)


@router.get("/traces/{trace_id}")
async def get_trace_detail(trace_id: str, request: Request) -> dict[str, Any]:
    """Get full trace with feedback.

    Args:
        trace_id: The trace identifier.

    Returns:
        Trace dict augmented with a ``feedback`` list.

    Raises:
        HTTPException: 404 if trace not found.
    """
    user_id = get_scoped_user_id(request)
    trace = _store.get_trace(trace_id, user_id=user_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    trace["feedback"] = _store.get_feedback(trace_id)
    return trace


@router.post("/traces/{trace_id}/feedback")
async def add_trace_feedback(
    trace_id: str,
    body: FeedbackRequest,
) -> dict[str, str]:
    """Add feedback to a trace.

    Args:
        trace_id: The trace to annotate.
        body: Feedback details.

    Returns:
        Dict with the generated feedback_id.

    Raises:
        HTTPException: 404 if trace not found.
    """
    trace = _store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    feedback_id = _store.add_feedback(
        trace_id,
        body.feedback_type,
        body.rating,
        reason=body.reason,
        user_id=body.user_id,
    )
    return {"feedback_id": feedback_id}


@router.post("/traces/build-dataset")
async def build_dataset(body: BuildDatasetRequest) -> dict[str, Any]:
    """Build a training dataset from selected traces.

    Supports SFT (single examples) and DPO (preference pairs).
    Writes JSONL to the specified output directory.

    Args:
        body: Dataset build parameters.

    Returns:
        Dict with output path, format, and example count.

    Raises:
        HTTPException: 400 on invalid format or empty result.
    """
    if body.format not in ("sft", "dpo"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {body.format}. Use 'sft' or 'dpo'.",
        )

    if body.format == "sft":
        examples = _store.export_as_sft(body.trace_ids)
    else:
        examples = _store.auto_pair_dpo(body.trace_ids)

    if not examples:
        raise HTTPException(
            status_code=400,
            detail="No examples could be generated from the selected traces.",
        )

    out_dir = Path(body.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{body.name}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(
        "Built %s dataset with %d examples at %s",
        body.format,
        len(examples),
        out_path,
    )
    return {
        "path": str(out_path),
        "format": body.format,
        "num_examples": len(examples),
    }
