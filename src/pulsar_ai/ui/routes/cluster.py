"""API routes for cluster management and distributed training monitoring."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pulsar_ai.compute.cluster import ClusterManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cluster"])

_cluster = ClusterManager()


@router.get("/cluster/status")
async def cluster_status() -> dict[str, Any]:
    """Get aggregated cluster status: all targets, GPUs, health."""
    return _cluster.get_cluster_status()


@router.get("/cluster/metrics/{target_id}")
async def target_gpu_metrics(target_id: str) -> dict[str, Any]:
    """Get live GPU metrics for a specific target."""
    try:
        gpus = _cluster.poll_gpu_metrics(target_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"target_id": target_id, "gpus": gpus}


class PreflightRequest(BaseModel):
    """Request body for pre-flight check."""

    config: dict[str, Any]
    target_ids: list[str]


@router.post("/cluster/preflight")
async def preflight_check(body: PreflightRequest) -> dict[str, Any]:
    """Run pre-flight validation before distributed training.

    Checks connectivity, VRAM sufficiency, GPU compatibility.
    """
    return _cluster.preflight_check(body.config, body.target_ids)


@router.get("/cluster/jobs/{job_id}/metrics")
async def distributed_job_metrics(
    job_id: str,
    limit: int = 500,
) -> dict[str, Any]:
    """Get per-rank distributed metrics for a training job."""
    metrics = _cluster.get_distributed_metrics(job_id, limit=limit)
    return {"job_id": job_id, "metrics": metrics, "total": len(metrics)}


@router.get("/cluster/configs")
async def list_cluster_configs() -> dict[str, Any]:
    """List saved cluster configuration presets."""
    configs = _cluster.list_cluster_configs()
    return {"configs": configs, "total": len(configs)}


class SaveClusterConfigRequest(BaseModel):
    """Request body for saving a cluster config."""

    name: str
    target_ids: list[str]
    master_target_id: str
    strategy: str = "fsdp_qlora"


@router.post("/cluster/configs")
async def save_cluster_config(body: SaveClusterConfigRequest) -> dict[str, Any]:
    """Save a cluster configuration preset."""
    config_id = _cluster.save_cluster_config(
        name=body.name,
        target_ids=body.target_ids,
        master_target_id=body.master_target_id,
        strategy=body.strategy,
    )
    return {"config_id": config_id}
