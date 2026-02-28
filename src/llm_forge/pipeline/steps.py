"""Step dispatchers for pipeline execution."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def dispatch_step(step_type: str, config: dict) -> dict[str, Any]:
    """Dispatch a pipeline step to the appropriate handler.

    Args:
        step_type: Step type (training, evaluation, export).
        config: Resolved step config.

    Returns:
        Dict with step outputs (paths, metrics, etc.).

    Raises:
        ValueError: If step type is unknown.
    """
    if step_type == "training":
        return _run_training_step(config)
    elif step_type == "evaluation":
        return _run_evaluation_step(config)
    elif step_type == "export":
        return _run_export_step(config)
    else:
        raise ValueError(f"Unknown step type: {step_type}")


def _run_training_step(config: dict) -> dict[str, Any]:
    """Execute a training step.

    Args:
        config: Training config with task, model, dataset, etc.

    Returns:
        Training results dict.
    """
    task = config.get("task", "sft")
    logger.info("Training step: task=%s", task)

    if task == "sft":
        from llm_forge.training.sft import train_sft
        return train_sft(config)
    elif task == "dpo":
        from llm_forge.training.dpo import train_dpo
        return train_dpo(config)
    else:
        raise ValueError(f"Unknown training task: {task}")


def _run_evaluation_step(config: dict) -> dict[str, Any]:
    """Execute an evaluation step.

    Args:
        config: Evaluation config with model_path, test_data_path.

    Returns:
        Evaluation results dict.
    """
    logger.info("Evaluation step: model=%s", config.get("model_path"))

    from llm_forge.evaluation.runner import run_evaluation
    return run_evaluation(config)


def _run_export_step(config: dict) -> dict[str, Any]:
    """Execute an export step.

    Args:
        config: Export config with model_path, export format.

    Returns:
        Export results dict.
    """
    export_config = config.get("export", {})
    fmt = export_config.get("format", "gguf")
    logger.info("Export step: format=%s", fmt)

    if fmt == "gguf":
        from llm_forge.export.gguf import export_gguf
        return export_gguf(config)
    elif fmt == "merged":
        from llm_forge.export.merged import export_merged
        return export_merged(config)
    elif fmt == "hub":
        from llm_forge.export.hub import push_to_hub
        return push_to_hub(config)
    else:
        raise ValueError(f"Unknown export format: {fmt}")
