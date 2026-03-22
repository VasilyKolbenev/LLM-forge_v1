"""Recipe Hub API routes: browse and run recipe templates."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from pulsar_ai.recipes import RecipeRegistry
from pulsar_ai.ui.experiment_store import ExperimentStore
from pulsar_ai.ui.jobs import submit_training_job

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recipes"])

_registry = RecipeRegistry()
_store = ExperimentStore()


@router.get("/recipes")
async def list_recipes(
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """List available recipes with optional filters.

    Args:
        task_type: Filter by task type (sft, dpo, grpo, ...).
        tag: Filter by tag.
        difficulty: Filter by difficulty level.

    Returns:
        List of recipe metadata dicts.
    """
    return _registry.list_recipes(task_type=task_type, tag=tag, difficulty=difficulty)


@router.get("/recipes/{name}")
async def get_recipe(name: str) -> dict:
    """Get recipe detail including meta and full config.

    Args:
        name: Recipe file stem (without .yaml).

    Returns:
        Dict with ``meta`` and ``config`` keys.
    """
    try:
        config = _registry.load_recipe(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Recipe '{name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    meta = _registry._read_meta(_registry._dir / f"{name}.yaml")
    return {"meta": meta, "config": config}


@router.post("/recipes/{name}/run")
async def run_recipe(name: str) -> dict:
    """Create an experiment from a recipe and start training.

    Args:
        name: Recipe file stem.

    Returns:
        Dict with job_id, experiment_id, and status.
    """
    try:
        config = _registry.load_recipe(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Recipe '{name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    task = config.get("task", "sft")
    experiment_id = _store.create(
        name=f"recipe-{name}",
        config=config,
        task=task,
    )
    job_id = submit_training_job(
        experiment_id=experiment_id,
        config=config,
        task=task,
    )
    return {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "status": "running",
    }
