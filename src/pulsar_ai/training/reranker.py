"""Reranker (Cross-Encoder) trainer.

Uses sentence-transformers CrossEncoder for training reranking models
on query-document-relevance datasets.
Produces full model checkpoints (not LoRA adapters).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_reranker_dataset(df: pd.DataFrame) -> list[str]:
    """Validate that the DataFrame has the required columns for reranker training.

    Args:
        df: DataFrame to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    required = {"query", "document", "relevance_score"}
    missing = required - set(df.columns)
    if missing:
        errors.append(
            f"Missing columns: {sorted(missing)}. Got: {sorted(df.columns)}"
        )
    return errors


def _run_reranker_training(
    config: dict,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run reranker training using sentence-transformers CrossEncoder.

    Args:
        config: Full config dict.
        callbacks: Optional list of callbacks (unused by CrossEncoder.fit,
            kept for API consistency).

    Returns:
        Dict with training metrics.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranker training. "
            "Install with: pip install 'pulsar-ai[embedding]'"
        ) from exc

    from pulsar_ai.data.loader import load_dataset_from_config

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/reranker")

    model_name = model_config.get("name")
    logger.info("Loading cross-encoder model: %s", model_name)

    model = CrossEncoder(model_name, num_labels=1)

    # Load and validate dataset
    df = load_dataset_from_config(config)
    errors = validate_reranker_dataset(df)
    if errors:
        raise ValueError(
            f"Reranker dataset validation failed: {'; '.join(errors)}"
        )

    # Prepare training samples as list of (sentence_pair, score)
    train_samples = []
    for _, row in df.iterrows():
        train_samples.append(
            ([str(row["query"]), str(row["document"])], float(row["relevance_score"]))
        )

    logger.info("Training on %d samples", len(train_samples))

    model.fit(
        train_dataloader=train_samples,
        epochs=training_config.get("epochs", 3),
        warmup_steps=training_config.get("warmup_steps", 100),
        output_path=output_dir,
    )

    logger.info("Reranker model saved to %s", output_dir)

    return {
        "output_dir": output_dir,
        "artifact_type": "full_model",
        "num_samples": len(train_samples),
    }


def train_reranker(config: dict, progress: Any = None) -> dict:
    """Run reranker training based on config.

    Automatically tracks experiment if logging.tracker is set.

    Args:
        config: Fully resolved config dict with reranker settings.
        progress: Optional ProgressCallback for real-time metrics.

    Returns:
        Dict with training results (output_dir, artifact_type, num_samples).
    """
    from pulsar_ai.tracking import track_experiment

    with track_experiment(config, task="reranker") as tracker:
        results = _run_reranker_training(config)

        tracker.log_metrics(
            {
                "num_samples": results["num_samples"],
            }
        )

        if results.get("output_dir"):
            tracker.log_artifact("full_model", results["output_dir"])

    return results
