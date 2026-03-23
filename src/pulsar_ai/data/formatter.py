"""Data formatting utilities for training dataset preparation.

Builds chat-style examples for SFT and preference pairs for DPO training.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_chat_examples(
    df: pd.DataFrame,
    system_prompt: str,
    text_column: str,
    label_columns: list[str],
    output_format: str = "json",
) -> list[dict]:
    """Build chat-style training examples from a DataFrame.

    Each example has system/user/assistant messages, ready for SFT training.

    Args:
        df: Source DataFrame with text and label columns.
        system_prompt: System message for all examples.
        text_column: Column name containing input text.
        label_columns: Column names to include in assistant response.
        output_format: "json" for JSON-formatted labels, "text" for plain text.

    Returns:
        List of dicts with ``messages`` key (system, user, assistant).
    """
    examples: list[dict] = []

    for _, row in df.iterrows():
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue

        # Build assistant response based on output format
        if output_format == "json":
            label_dict = {col: row[col] for col in label_columns}
            assistant_content = json.dumps(label_dict, ensure_ascii=False)
        else:
            # Plain text: join label values
            values = [str(row[col]) for col in label_columns]
            assistant_content = "\n".join(values) if len(values) > 1 else values[0]

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )

    logger.info(
        "Built %d chat examples from %d rows (format=%s)",
        len(examples),
        len(df),
        output_format,
    )
    return examples


def load_system_prompt(path: str) -> str:
    """Load a system prompt from a text file.

    Args:
        path: Path to the prompt file.

    Returns:
        Stripped prompt string.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def build_dpo_pairs(
    errors_df: pd.DataFrame,
    all_data: pd.DataFrame,
    label_columns: list[str],
    n_synthetic: int = 0,
    seed: Optional[int] = None,
) -> list[dict]:
    """Build DPO preference pairs from classification errors.

    Creates ``(prompt, chosen, rejected)`` triplets:
    - From errors: chosen=true label, rejected=predicted label
    - Synthetic: random samples from all_data with swapped labels

    Args:
        errors_df: DataFrame with columns ``phrase``, ``true_<col>``, ``pred_<col>``.
        all_data: Full dataset for synthetic pair generation.
        label_columns: Label column names (used to find true/pred columns).
        n_synthetic: Number of synthetic pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with ``prompt``, ``chosen``, ``rejected`` keys.
    """
    if seed is not None:
        random.seed(seed)

    pairs: list[dict] = []

    # Build pairs from actual errors
    for _, row in errors_df.iterrows():
        prompt = str(row.get("phrase", ""))

        chosen_parts = []
        rejected_parts = []
        for col in label_columns:
            true_col = f"true_{col}"
            pred_col = f"pred_{col}"
            chosen_parts.append(str(row.get(true_col, "")))
            rejected_parts.append(str(row.get(pred_col, "")))

        chosen = "\n".join(chosen_parts) if len(chosen_parts) > 1 else chosen_parts[0]
        rejected = (
            "\n".join(rejected_parts) if len(rejected_parts) > 1 else rejected_parts[0]
        )

        if chosen != rejected:
            pairs.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )

    # Generate synthetic pairs
    if n_synthetic > 0 and len(all_data) >= 2:
        unique_labels: dict[str, list[str]] = {}
        for col in label_columns:
            if col in all_data.columns:
                unique_labels[col] = list(all_data[col].dropna().unique())

        for _ in range(n_synthetic):
            idx = random.randint(0, len(all_data) - 1)
            sample = all_data.iloc[idx]
            prompt = str(sample.get("phrase", sample.get("text", "")))

            chosen_parts = []
            rejected_parts = []
            for col in label_columns:
                true_val = str(sample.get(col, ""))
                chosen_parts.append(true_val)
                # Pick a different label for rejected
                candidates = [
                    lbl for lbl in unique_labels.get(col, []) if str(lbl) != true_val
                ]
                if candidates:
                    rejected_parts.append(random.choice(candidates))
                else:
                    rejected_parts.append(true_val + "_wrong")

            chosen = (
                "\n".join(chosen_parts) if len(chosen_parts) > 1 else chosen_parts[0]
            )
            rejected = (
                "\n".join(rejected_parts)
                if len(rejected_parts) > 1
                else rejected_parts[0]
            )

            if chosen != rejected:
                pairs.append(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                )

    logger.info(
        "Built %d DPO pairs (%d from errors, %d synthetic)",
        len(pairs),
        len(errors_df),
        n_synthetic,
    )
    return pairs


def apply_chat_template(
    examples: list[dict],
    tokenizer: "Any",
) -> "Dataset":
    """Apply tokenizer chat template to message lists.

    Converts chat examples into formatted text strings using the
    tokenizer's built-in chat template.

    Args:
        examples: List of dicts with ``messages`` key.
        tokenizer: HuggingFace tokenizer with apply_chat_template method.

    Returns:
        HuggingFace Dataset with ``text`` field.
    """
    from datasets import Dataset as HFDataset

    texts = []
    for ex in examples:
        messages = ex.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)

    logger.info("Applied chat template to %d examples", len(texts))
    return HFDataset.from_dict({"text": texts})


def build_multimodal_chat_examples(
    df: pd.DataFrame,
    system_prompt: str,
    image_column: str = "image",
    conversations_column: Optional[str] = "conversations",
    text_column: Optional[str] = None,
    label_columns: Optional[list[str]] = None,
) -> list[dict]:
    """Build multimodal chat examples with image references.

    Supports two input modes:
    1. Pre-built conversations column with <image> placeholders.
    2. Text+label mode (like build_chat_examples but with image prepended).

    Args:
        df: Source DataFrame with image and text/conversation columns.
        system_prompt: System message.
        image_column: Column containing image paths/URLs.
        conversations_column: Column with pre-built conversation lists.
        text_column: Column with input text (fallback mode).
        label_columns: Label columns for assistant response (fallback mode).

    Returns:
        List of dicts with ``messages`` and ``images`` keys.
    """
    from pulsar_ai.data.image_loader import load_image

    examples: list[dict] = []

    for _, row in df.iterrows():
        image_ref = str(row.get(image_column, "")).strip()
        if not image_ref:
            continue

        # Mode 1: Pre-built conversations
        if conversations_column and conversations_column in df.columns:
            convs = row.get(conversations_column)
            if isinstance(convs, str):
                convs = json.loads(convs)
            if isinstance(convs, list):
                messages = [{"role": "system", "content": system_prompt}] + convs
                try:
                    image = load_image(image_ref)
                    examples.append({"messages": messages, "images": [image]})
                except Exception as exc:
                    logger.warning("Failed to load image %s: %s", image_ref, exc)
                continue

        # Mode 2: Text + labels with <image> prefix
        if text_column:
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue

            user_content = f"<image>\n{text}"

            if label_columns:
                label_dict = {col: row[col] for col in label_columns if col in df.columns}
                assistant_content = json.dumps(label_dict, ensure_ascii=False)
            else:
                assistant_content = ""

            try:
                image = load_image(image_ref)
                examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "images": [image],
                })
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", image_ref, exc)

    logger.info("Built %d multimodal examples from %d rows", len(examples), len(df))
    return examples


def apply_multimodal_chat_template(
    examples: list[dict],
    processor: "Any",
) -> "Dataset":
    """Apply processor chat template to multimodal examples.

    Args:
        examples: List of dicts with ``messages`` and ``images`` keys.
        processor: HuggingFace processor (tokenizer + image processor).

    Returns:
        HuggingFace Dataset with input_ids, attention_mask, pixel_values.
    """
    from datasets import Dataset as HFDataset

    texts = []
    all_images = []

    for ex in examples:
        messages = ex.get("messages", [])
        images = ex.get("images", [])

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
        all_images.append(images[0] if images else None)

    logger.info("Applied multimodal chat template to %d examples", len(texts))
    return HFDataset.from_dict({"text": texts, "images": all_images})
