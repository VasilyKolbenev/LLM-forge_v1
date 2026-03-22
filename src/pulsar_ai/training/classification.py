"""Text classification trainer.

Uses HuggingFace AutoModelForSequenceClassification for training
classification models. Produces full model checkpoints (not LoRA adapters).
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def detect_labels(
    df: pd.DataFrame,
    label_column: str = "label",
) -> dict[str, int]:
    """Auto-detect unique labels and create a label-to-id mapping.

    Args:
        df: DataFrame containing the label column.
        label_column: Name of the label column.

    Returns:
        Dict mapping label string to integer id.
    """
    unique_labels = sorted(df[label_column].astype(str).unique())
    return {label: idx for idx, label in enumerate(unique_labels)}


def validate_classification_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
) -> list[str]:
    """Validate that the DataFrame has the required columns for classification.

    Args:
        df: DataFrame to validate.
        text_column: Name of the text column.
        label_column: Name of the label column.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    required = {text_column, label_column}
    missing = required - set(df.columns)
    if missing:
        errors.append(
            f"Missing columns: {sorted(missing)}. Got: {sorted(df.columns)}"
        )
    return errors


def _run_classification_training(
    config: dict,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run classification training using HuggingFace Trainer.

    Args:
        config: Full config dict.
        callbacks: Optional list of HF TrainerCallbacks.

    Returns:
        Dict with training metrics.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset
    from pulsar_ai.data.loader import load_dataset_from_config

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    cls_config = config.get("classification", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/classification")

    text_column = cls_config.get("text_column", "text")
    label_column = cls_config.get("label_column", "label")

    # Load and validate dataset
    df = load_dataset_from_config(config)
    errors = validate_classification_dataset(df, text_column, label_column)
    if errors:
        raise ValueError(
            f"Classification dataset validation failed: {'; '.join(errors)}"
        )

    # Detect labels
    label_map = detect_labels(df, label_column)
    num_labels = len(label_map)
    logger.info("Detected %d labels: %s", num_labels, label_map)

    # Map string labels to integer ids
    df["_label_id"] = df[label_column].astype(str).map(label_map)

    model_name = model_config.get("name")
    logger.info("Loading classification model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True,
    )

    # Tokenize dataset
    train_dataset = Dataset.from_pandas(df)

    def tokenize_fn(examples: dict) -> dict:
        tokens = tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=training_config.get("max_seq_length", 512),
        )
        tokens["labels"] = examples["_label_id"]
        return tokens

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    # Training arguments
    lr = training_config.get("learning_rate", 2e-5)
    if isinstance(lr, str):
        lr = float(lr)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("batch_size", 16),
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=lr,
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 2),
        seed=training_config.get("seed", 42),
        report_to=config.get("logging", {}).get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    logger.info("Starting classification training...")
    stats = trainer.train()

    # Save model, tokenizer, and label map
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_map_path = Path(output_dir) / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Classification model saved to %s", output_dir)

    return {
        "training_loss": stats.training_loss,
        "global_steps": stats.global_step,
        "output_dir": output_dir,
        "artifact_type": "full_model",
        "num_labels": num_labels,
        "label_map": label_map,
    }


def train_classification(config: dict, progress: Any = None) -> dict:
    """Run text classification training based on config.

    Automatically tracks experiment if logging.tracker is set.

    Args:
        config: Fully resolved config dict with classification settings.
        progress: Optional ProgressCallback for real-time metrics.

    Returns:
        Dict with training results (training_loss, global_steps,
        output_dir, artifact_type, num_labels, label_map).
    """
    from pulsar_ai.tracking import track_experiment

    hf_callbacks: list[Any] = []
    if progress is not None:
        from pulsar_ai.ui.progress import make_hf_callback

        hf_callbacks.append(make_hf_callback(progress))

    with track_experiment(config, task="classification") as tracker:
        results = _run_classification_training(
            config, callbacks=hf_callbacks
        )

        tracker.log_metrics(
            {
                "training_loss": results["training_loss"],
                "global_steps": results["global_steps"],
                "num_labels": results["num_labels"],
            }
        )

        if results.get("output_dir"):
            tracker.log_artifact("full_model", results["output_dir"])

    return results
