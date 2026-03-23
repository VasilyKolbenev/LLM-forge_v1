"""API routes for benchmark operations.

Provides endpoints for listing, running, comparing, and ranking
model benchmarks.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from pulsar_ai.benchmark.baselines import seed_baselines
from pulsar_ai.benchmark.runner import run_benchmark
from pulsar_ai.benchmark.store import BenchmarkStore
from pulsar_ai.ui.auth import get_current_user, get_scoped_user_id, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["benchmarks"])

_store = BenchmarkStore()

# Seed baselines on first import
_baselines_seeded = False


def _ensure_baselines() -> None:
    global _baselines_seeded
    if not _baselines_seeded:
        try:
            count = seed_baselines(_store)
            if count > 0:
                logger.info("Seeded %d baseline benchmarks", count)
        except Exception as exc:
            logger.warning("Failed to seed baselines: %s", exc)
        _baselines_seeded = True


class RunBenchmarkRequest(BaseModel):
    """Body for running a benchmark."""

    model_path: str
    model_name: str = ""
    experiment_id: str = ""
    eval_data_path: str = ""
    gpu_cost_per_hour: float = 2.0
    tags: list[str] = []


@router.get("/benchmarks")
async def list_benchmarks(
    request: Request,
    model_name: str | None = None,
    experiment_id: str | None = None,
    is_baseline: bool | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List benchmark results with optional filters."""
    _ensure_baselines()
    user_id = get_scoped_user_id(request)
    benchmarks = _store.list_all(
        model_name=model_name,
        experiment_id=experiment_id,
        is_baseline=is_baseline,
        limit=limit,
        user_id=user_id,
    )
    return {"benchmarks": benchmarks, "total": len(benchmarks)}


@router.get("/benchmarks/leaderboard")
async def leaderboard(
    metric: str = "tokens_per_sec",
    order: str = "desc",
    limit: int = 20,
) -> dict[str, Any]:
    """Get ranked list of benchmarks by a metric."""
    _ensure_baselines()
    try:
        results = _store.leaderboard(metric=metric, order=order, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"leaderboard": results, "metric": metric, "order": order}


@router.get("/benchmarks/compare")
async def compare_benchmarks(ids: str, request: Request) -> dict[str, Any]:
    """Compare multiple benchmark runs.

    Args:
        ids: Comma-separated benchmark IDs.
    """
    user_id = get_scoped_user_id(request)
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 comma-separated IDs to compare",
        )
    try:
        comparison = _store.compare(id_list, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return comparison


@router.get("/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str) -> dict[str, Any]:
    """Get a single benchmark result."""
    _ensure_baselines()
    result = _store.get(benchmark_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return result


@router.post("/benchmarks/run")
async def run_benchmark_endpoint(body: RunBenchmarkRequest, request: Request) -> dict[str, Any]:
    """Run a benchmark on a model.

    Returns summary metrics from the benchmark run.
    """
    config: dict[str, Any] = {}
    if body.model_name:
        config["model_name"] = body.model_name
    if body.eval_data_path:
        config["eval_data"] = body.eval_data_path
    config["gpu_cost_per_hour"] = body.gpu_cost_per_hour

    user_id = get_user_id(request)
    try:
        result = run_benchmark(
            model_path=body.model_path,
            config=config,
            experiment_id=body.experiment_id,
            tags=body.tags,
            store=_store,
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception("Benchmark run failed")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}")

    return {
        "benchmark_id": result.id,
        "model_name": result.model_name,
        "tokens_per_sec": result.tokens_per_sec,
        "time_to_first_token_ms": result.time_to_first_token_ms,
        "peak_vram_gb": result.peak_vram_gb,
        "model_size_params": result.model_size_params,
        "perplexity": result.perplexity,
        "estimated_cost_per_1m_tokens": result.estimated_cost_per_1m_tokens,
        "status": result.status,
    }


@router.delete("/benchmarks/{benchmark_id}")
async def delete_benchmark(benchmark_id: str, request: Request) -> dict[str, Any]:
    """Delete a benchmark result."""
    user_id = get_scoped_user_id(request)
    existing = _store.get(benchmark_id, user_id=user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    _store.delete(benchmark_id, user_id=user_id)
    return {"deleted": benchmark_id}
