"""GRPO (Group Relative Policy Optimization) trainer.

Supports configurable reward functions via a registry, including
built-in rewards and custom .py file loading.
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias for reward functions
RewardFn = Callable[..., list[float]]


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------


def reward_format_compliance(
    completions: list[str],
    prompts: list[str],
    expected_format: str = "json",
    **kwargs: Any,
) -> list[float]:
    """Score completions by format compliance.

    Args:
        completions: Model-generated texts.
        prompts: Corresponding prompts (unused, kept for API consistency).
        expected_format: Target format, currently supports "json".
        **kwargs: Extra args forwarded by the trainer.

    Returns:
        List of scores: 1.0 if valid, 0.0 otherwise.
    """
    scores: list[float] = []
    for text in completions:
        if expected_format == "json":
            try:
                json.loads(text)
                scores.append(1.0)
            except (json.JSONDecodeError, TypeError):
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores


def reward_length_penalty(
    completions: list[str],
    prompts: list[str],
    min_length: int = 10,
    max_length: int = 500,
    **kwargs: Any,
) -> list[float]:
    """Score completions by length within an acceptable range.

    Args:
        completions: Model-generated texts.
        prompts: Corresponding prompts (unused).
        min_length: Minimum acceptable character length.
        max_length: Maximum acceptable character length.
        **kwargs: Extra args forwarded by the trainer.

    Returns:
        List of scores between 0.0 and 1.0.
    """
    scores: list[float] = []
    for text in completions:
        length = len(text)
        if length < min_length:
            scores.append(length / min_length if min_length > 0 else 0.0)
        elif length > max_length:
            ratio = max_length / length if length > 0 else 0.0
            scores.append(max(ratio, 0.0))
        else:
            scores.append(1.0)
    return scores


def reward_keyword_match(
    completions: list[str],
    prompts: list[str],
    keywords: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score completions by fraction of keywords found.

    Args:
        completions: Model-generated texts.
        prompts: Corresponding prompts (unused).
        keywords: List of keywords to look for.
        **kwargs: Extra args forwarded by the trainer.

    Returns:
        List of scores (fraction of keywords present in each completion).
    """
    if not keywords:
        return [0.0] * len(completions)
    scores: list[float] = []
    for text in completions:
        lower_text = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in lower_text)
        scores.append(found / len(keywords))
    return scores


# ---------------------------------------------------------------------------
# Reward registry
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict[str, RewardFn] = {
    "format_compliance": reward_format_compliance,
    "length_penalty": reward_length_penalty,
    "keyword_match": reward_keyword_match,
}


def _load_custom_reward(path: Path) -> RewardFn:
    """Load a custom reward function from a .py file.

    The file must define a callable named ``reward_fn``.

    Args:
        path: Absolute or relative path to a Python file.

    Returns:
        The ``reward_fn`` callable from the file.

    Raises:
        ValueError: If the path contains ``..`` (directory traversal).
        FileNotFoundError: If the file does not exist.
        AttributeError: If the file has no ``reward_fn`` attribute.
    """
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"Directory traversal blocked in reward path: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Custom reward file not found: {path}")

    spec = importlib.util.spec_from_file_location("custom_reward", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "reward_fn"):
        raise AttributeError(f"Custom reward file {path} must define 'reward_fn'")
    return module.reward_fn  # type: ignore[return-value]


def get_reward_function(name: str) -> RewardFn:
    """Look up a reward function by registry name or .py file path.

    Args:
        name: Built-in reward name (e.g. "format_compliance") or path to
            a .py file containing a ``reward_fn`` callable.

    Returns:
        The resolved reward callable.

    Raises:
        ValueError: If the name is not in the registry and is not a
            valid file path.
    """
    if name in REWARD_REGISTRY:
        return REWARD_REGISTRY[name]

    path = Path(name)
    if path.suffix == ".py":
        return _load_custom_reward(path)

    raise ValueError(
        f"Unknown reward function '{name}'. "
        f"Available: {sorted(REWARD_REGISTRY.keys())}. "
        f"Or provide a .py file path."
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_grpo_dataset(config: dict) -> Any:
    """Load a prompt-only dataset for GRPO training.

    Tries columns in order: prompt, text, instruction, input.

    Args:
        config: Full config dict with dataset settings.

    Returns:
        HuggingFace Dataset with a ``prompt`` column.

    Raises:
        ValueError: If no recognised prompt column is found.
    """
    from datasets import Dataset

    ds_config = config.get("dataset", {})
    ds_path = ds_config.get("path")

    if not ds_path:
        raise ValueError("dataset.path is required for GRPO training")

    import pandas as pd

    df = pd.read_json(ds_path, lines=True)

    prompt_col = None
    for col_name in ("prompt", "text", "instruction", "input"):
        if col_name in df.columns:
            prompt_col = col_name
            break

    if prompt_col is None:
        raise ValueError(
            f"No prompt column found in {ds_path}. "
            f"Expected one of: prompt, text, instruction, input. "
            f"Got: {list(df.columns)}"
        )

    if prompt_col != "prompt":
        df = df.rename(columns={prompt_col: "prompt"})
        logger.info("Renamed column '%s' -> 'prompt'", prompt_col)

    ds = Dataset.from_pandas(df[["prompt"]])
    logger.info("Loaded %d GRPO prompts from %s", len(ds), ds_path)
    return ds


# ---------------------------------------------------------------------------
# TRL GRPO runner
# ---------------------------------------------------------------------------


def _run_grpo_trl(
    config: dict,
    reward_fn: RewardFn,
    callbacks: list[Any] | None = None,
) -> dict:
    """Run GRPO training using ``trl.GRPOTrainer``.

    Args:
        config: Full config dict.
        reward_fn: Callable used to score completions.
        callbacks: Optional list of HF TrainerCallbacks.

    Returns:
        Dict with training metrics.
    """
    import torch
    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    grpo_config = config.get("grpo", {})
    output_dir = config.get("output", {}).get("dir", "./outputs/grpo")

    model_name = model_config.get("name")
    logger.info("Loading model for GRPO: %s", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dataset = _load_grpo_dataset(config)

    grpo_args = GRPOConfig(
        per_device_train_batch_size=training_config.get("batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation", 8),
        warmup_steps=training_config.get("warmup_steps", 10),
        num_train_epochs=training_config.get("epochs", 1),
        learning_rate=float(training_config.get("learning_rate", 5e-6)),
        bf16=config.get("_hardware", {}).get("bf16_supported", True),
        logging_steps=training_config.get("logging_steps", 10),
        output_dir=output_dir,
        seed=training_config.get("seed", 42),
        report_to=config.get("logging", {}).get("report_to", "none"),
        num_generations=grpo_config.get("num_generations", 4),
        max_completion_length=grpo_config.get("max_completion_length", 256),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    logger.info("Starting GRPO training...")
    stats = trainer.train()

    adapter_dir = str(Path(output_dir) / "grpo_model")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("GRPO model saved to %s", adapter_dir)

    vram_peak = 0.0
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

    # TODO(#grpo-metrics): extract reward_mean/std from training logs;
    # GRPOTrainer does not expose a _reward_scores attribute.
    reward_mean = 0.0
    reward_std = 0.0

    return {
        "training_loss": stats.training_loss,
        "global_steps": stats.global_step,
        "reward_mean": round(reward_mean, 4),
        "reward_std": round(reward_std, 4),
        "vram_peak_gb": round(vram_peak, 2),
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_grpo(config: dict, progress: Any = None) -> dict:
    """Run GRPO training based on config.

    Validates the reward function before loading the model.
    Automatically tracks experiment if logging.tracker is set.

    Args:
        config: Fully resolved config dict with GRPO settings.
        progress: Optional ProgressCallback for real-time metrics.

    Returns:
        Dict with training results (training_loss, global_steps,
        reward_mean, reward_std, vram_peak_gb, output_dir).
    """
    from pulsar_ai.tracking import track_experiment

    grpo_config = config.get("grpo", {})
    reward_name = grpo_config.get("reward_function", "format_compliance")

    # Validate reward function early, before expensive model loading
    reward_fn = get_reward_function(reward_name)
    logger.info("Using reward function: %s", reward_name)

    hf_callbacks: list[Any] = []
    if progress is not None:
        from pulsar_ai.ui.progress import make_hf_callback

        hf_callbacks.append(make_hf_callback(progress))

    with track_experiment(config, task="grpo") as tracker:
        results = _run_grpo_trl(config, reward_fn, callbacks=hf_callbacks)

        tracker.log_metrics(
            {
                "training_loss": results["training_loss"],
                "global_steps": results["global_steps"],
                "reward_mean": results["reward_mean"],
                "reward_std": results["reward_std"],
                "vram_peak_gb": results["vram_peak_gb"],
            }
        )

        if results.get("adapter_dir"):
            tracker.log_artifact("grpo_model", results["adapter_dir"])

        tracker.finish(status="completed", results=results)

    return results
