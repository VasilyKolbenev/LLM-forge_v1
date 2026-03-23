"""Training resilience: checkpoint management, retry logic, OOM detection.

Provides utilities for recovering from training failures including
automatic checkpoint detection, retry decisions, and graceful shutdown.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Special exit code indicating OOM — retryable with checkpoint resume
OOM_EXIT_CODE = 42

# Patterns that indicate a retryable failure
RETRYABLE_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"NCCL timeout", re.IGNORECASE),
    re.compile(r"NCCL error", re.IGNORECASE),
    re.compile(r"Connection reset by peer", re.IGNORECASE),
    re.compile(r"SSH.*disconnect", re.IGNORECASE),
    re.compile(r"Socket timeout", re.IGNORECASE),
    re.compile(r"Broken pipe", re.IGNORECASE),
]

# Patterns that indicate a non-retryable failure
FATAL_PATTERNS = [
    re.compile(r"FileNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"KeyError", re.IGNORECASE),
    re.compile(r"Invalid config", re.IGNORECASE),
]


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving during training."""

    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: bool = True
    checkpoint_dir: str = ""


def get_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory in output_dir.

    Looks for directories named checkpoint-XXXX and returns
    the one with the highest step number.

    Args:
        output_dir: Training output directory.

    Returns:
        Path to latest checkpoint, or None if not found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = []
    for d in output_path.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, str(d)))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest = checkpoints[0][1]
    logger.info("Found latest checkpoint: %s", latest)
    return latest


def inject_checkpoint_args(
    config: dict,
    checkpoint_config: CheckpointConfig | None = None,
) -> dict:
    """Merge checkpoint settings into training config.

    Args:
        config: Training config dict.
        checkpoint_config: Checkpoint configuration. Uses defaults if None.

    Returns:
        Updated config with checkpoint settings.
    """
    ckpt = checkpoint_config or CheckpointConfig()
    training = config.setdefault("training", {})

    training["save_steps"] = ckpt.save_steps
    training["save_total_limit"] = ckpt.save_total_limit

    if ckpt.resume_from_checkpoint and ckpt.checkpoint_dir:
        latest = get_latest_checkpoint(ckpt.checkpoint_dir)
        if latest:
            training["resume_from_checkpoint"] = latest
            logger.info("Will resume from checkpoint: %s", latest)

    return config


def should_retry(
    error: str,
    retry_count: int,
    max_retries: int = 3,
    exit_code: int | None = None,
) -> bool:
    """Determine if a failed job should be retried.

    Retries on OOM, NCCL timeout, SSH disconnect.
    Does not retry on config errors, import errors, etc.

    Args:
        error: Error message string.
        retry_count: Current retry attempt count.
        max_retries: Maximum allowed retries.
        exit_code: Process exit code (42 = OOM).

    Returns:
        True if the job should be retried.
    """
    if retry_count >= max_retries:
        logger.info("Max retries (%d) reached, not retrying", max_retries)
        return False

    # OOM exit code is always retryable
    if exit_code == OOM_EXIT_CODE:
        logger.info("OOM exit code detected, will retry (attempt %d/%d)", retry_count + 1, max_retries)
        return True

    # Check for fatal patterns first (never retry)
    for pattern in FATAL_PATTERNS:
        if pattern.search(error):
            logger.info("Fatal error pattern matched: %s — not retrying", pattern.pattern)
            return False

    # Check for retryable patterns
    for pattern in RETRYABLE_PATTERNS:
        if pattern.search(error):
            logger.info(
                "Retryable error pattern: %s — retry %d/%d",
                pattern.pattern, retry_count + 1, max_retries,
            )
            return True

    logger.info("Unknown error type, not retrying: %.100s", error)
    return False


def get_retry_delay(retry_count: int) -> float:
    """Calculate retry delay with exponential backoff.

    Args:
        retry_count: Current retry attempt (0-indexed).

    Returns:
        Delay in seconds: 30, 60, 120, 240, ...
    """
    return 30.0 * (2 ** retry_count)


def reduce_batch_size(config: dict) -> dict:
    """Halve the batch size for OOM recovery.

    Args:
        config: Training config dict.

    Returns:
        Updated config with halved batch size and doubled gradient accumulation.
    """
    training = config.get("training", {})
    bs = training.get("batch_size", 1)
    ga = training.get("gradient_accumulation", 1)

    if bs > 1:
        training["batch_size"] = max(1, bs // 2)
        training["gradient_accumulation"] = ga * 2
        logger.info(
            "Reduced batch_size %d→%d, gradient_accumulation %d→%d",
            bs, training["batch_size"], ga, training["gradient_accumulation"],
        )
    else:
        logger.warning("batch_size already 1, cannot reduce further")

    return config
