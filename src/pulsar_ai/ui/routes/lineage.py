"""API routes for lineage graph and lifecycle timeline.

Provides endpoints for visualising data lineage across experiments,
models, traces, and feedback — plus a chronological event timeline.
"""

import logging
from typing import Any

from fastapi import APIRouter

from pulsar_ai.storage.lineage_store import LineageStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lineage"])

_store = LineageStore()


@router.get("/lineage/graph")
async def get_lineage_graph(model_name: str | None = None) -> dict[str, Any]:
    """Get lineage graph for visualization.

    Args:
        model_name: Optional filter to scope the graph to a single model.

    Returns:
        Dict with ``nodes`` and ``edges`` lists.
    """
    return _store.get_lineage_graph(model_name)


@router.get("/lineage/timeline")
async def get_timeline(limit: int = 50) -> dict[str, Any]:
    """Get lifecycle timeline events.

    Args:
        limit: Maximum number of events to return.

    Returns:
        Dict with ``events`` list.
    """
    return {"events": _store.get_timeline(limit)}
