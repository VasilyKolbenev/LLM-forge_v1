"""Background job manager for training jobs."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

from llm_forge.ui.experiment_store import ExperimentStore
from llm_forge.ui.progress import ProgressCallback, cleanup_queue

logger = logging.getLogger(__name__)

# Single worker — training is GPU-bound, one job at a time
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict[str, Any]] = {}
_store = ExperimentStore()


def submit_training_job(
    experiment_id: str,
    config: dict,
    task: str = "sft",
) -> str:
    """Submit a training job to run in background.

    Args:
        experiment_id: Experiment ID in the store.
        config: Full resolved training config.
        task: Training task (sft or dpo).

    Returns:
        Job ID for tracking progress.
    """
    job_id = str(uuid.uuid4())[:8]
    progress = ProgressCallback(job_id, experiment_id)

    future = _executor.submit(
        _run_training, job_id, experiment_id, config, task, progress
    )

    _jobs[job_id] = {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "status": "running",
        "future": future,
    }

    _store.update_status(experiment_id, "running")
    logger.info("Submitted training job %s for experiment %s", job_id, experiment_id)
    return job_id


def _run_training(
    job_id: str,
    experiment_id: str,
    config: dict,
    task: str,
    progress: ProgressCallback,
) -> dict:
    """Execute training in background thread.

    Args:
        job_id: Job ID.
        experiment_id: Experiment ID.
        config: Training config.
        task: Training task.
        progress: Progress callback for SSE.

    Returns:
        Training results dict.
    """
    store = ExperimentStore()

    try:
        if task == "sft":
            from llm_forge.training.sft import train_sft
            results = train_sft(config)
        elif task == "dpo":
            from llm_forge.training.dpo import train_dpo
            results = train_dpo(config)
        else:
            raise ValueError(f"Unknown task: {task}")

        store.update_status(experiment_id, "completed")
        store.set_artifacts(experiment_id, {
            k: v for k, v in results.items()
            if isinstance(v, str)
        })
        if "training_loss" in results:
            store.add_metrics(experiment_id, {
                "loss": results["training_loss"],
                "step": results.get("global_steps", 0),
            })

        progress.on_complete(results)
        _jobs[job_id]["status"] = "completed"

        logger.info("Training job %s completed", job_id)
        return results

    except Exception as e:
        logger.exception("Training job %s failed", job_id)
        store.update_status(experiment_id, "failed")
        progress.on_error(str(e))
        _jobs[job_id]["status"] = "failed"
        raise
    finally:
        cleanup_queue(job_id)


def get_job(job_id: str) -> dict | None:
    """Get job info by ID.

    Args:
        job_id: Job ID.

    Returns:
        Job info dict or None.
    """
    job = _jobs.get(job_id)
    if job:
        return {k: v for k, v in job.items() if k != "future"}
    return None


def list_jobs() -> list[dict]:
    """List all jobs.

    Returns:
        List of job info dicts.
    """
    return [
        {k: v for k, v in job.items() if k != "future"}
        for job in _jobs.values()
    ]


def cancel_job(job_id: str) -> bool:
    """Cancel a running job.

    Args:
        job_id: Job ID.

    Returns:
        True if cancelled, False otherwise.
    """
    job = _jobs.get(job_id)
    if job and isinstance(job.get("future"), Future):
        cancelled = job["future"].cancel()
        if cancelled:
            job["status"] = "cancelled"
            store = ExperimentStore()
            store.update_status(job["experiment_id"], "cancelled")
        return cancelled
    return False
