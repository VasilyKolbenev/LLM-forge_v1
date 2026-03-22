"""Embedding model trainer.

Uses sentence-transformers v3 API (SentenceTransformerTrainer) for
training embedding models on triplet or pair datasets.
Produces full model checkpoints (not LoRA adapters).
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_embedding_dataset(
    df: pd.DataFrame,
    schema: str = "triplet",
) -> list[str]:
    """Validate that the DataFrame has the required columns for embedding training.

    Args:
        df: DataFrame to validate.
        schema: Dataset schema type — "triplet" or "pair".

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    if schema == "triplet":
        required = {"anchor", "positive", "negative"}
    elif schema == "pair":
        required = {"sentence1", "sentence2", "label"}
    else:
        errors.append(f"Unknown schema '{schema}', expected 'triplet' or 'pair'")
        return errors

    missing = required - set(df.columns)
    if missing:
        errors.append(
            f"Missing columns for '{schema}' schema: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )
    return errors


def _run_embedding_training(
    config: dict,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run embedding training using sentence-transformers v3 API.

    Args:
        config: Full config dict.
        callbacks: Optional list of HF TrainerCallbacks.

    Returns:
        Dict with training metrics.
    """
    try:
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers import losses as st_losses
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding training. "
            "Install with: pip install 'pulsar-ai[embedding]'"
        ) from exc

    from pulsar_ai.data.loader import load_dataset_from_config
    from datasets import Dataset

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    embed_config = config.get("embedding", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/embedding")

    model_name = model_config.get("name")
    logger.info("Loading embedding model: %s", model_name)

    model = SentenceTransformer(model_name)

    # Load and validate dataset
    df = load_dataset_from_config(config)
    schema = embed_config.get("schema", "triplet")
    errors = validate_embedding_dataset(df, schema=schema)
    if errors:
        raise ValueError(
            f"Embedding dataset validation failed: {'; '.join(errors)}"
        )

    train_dataset = Dataset.from_pandas(df)

    # Resolve loss function
    loss_name = embed_config.get("loss", "MultipleNegativesRankingLoss")
    loss_cls = getattr(st_losses, loss_name, None)
    if loss_cls is None:
        raise ValueError(
            f"Unknown loss function '{loss_name}' in sentence_transformers.losses"
        )
    loss = loss_cls(model)
    logger.info("Using loss: %s", loss_name)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("batch_size", 16),
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 2),
        seed=training_config.get("seed", 42),
        report_to=config.get("logging", {}).get("report_to", "none"),
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=callbacks or None,
    )

    logger.info("Starting embedding training...")
    stats = trainer.train()

    # Save full model
    model.save(output_dir)
    logger.info("Embedding model saved to %s", output_dir)

    return {
        "training_loss": stats.training_loss,
        "global_steps": stats.global_step,
        "output_dir": output_dir,
        "artifact_type": "full_model",
    }


def train_embedding(config: dict, progress: Any = None) -> dict:
    """Run embedding model training based on config.

    Automatically tracks experiment if logging.tracker is set.

    Args:
        config: Fully resolved config dict with embedding settings.
        progress: Optional ProgressCallback for real-time metrics.

    Returns:
        Dict with training results (training_loss, global_steps,
        output_dir, artifact_type).
    """
    from pulsar_ai.tracking import track_experiment

    hf_callbacks: list[Any] = []
    if progress is not None:
        from pulsar_ai.ui.progress import make_hf_callback

        hf_callbacks.append(make_hf_callback(progress))

    with track_experiment(config, task="embedding") as tracker:
        results = _run_embedding_training(config, callbacks=hf_callbacks)

        tracker.log_metrics(
            {
                "training_loss": results["training_loss"],
                "global_steps": results["global_steps"],
            }
        )

        if results.get("output_dir"):
            tracker.log_artifact("full_model", results["output_dir"])

    return results
