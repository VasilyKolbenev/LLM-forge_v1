"""Entry point for distributed training via accelerate launch.

This script is invoked by distributed.py and should not be called directly.
It loads the training config from a YAML file and runs SFT, DPO, or GRPO.

Hardened with:
- OOM handler: saves checkpoint and exits with code 42 (retryable)
- SIGTERM handler: saves checkpoint for graceful shutdown
- GRPO task support
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

from pulsar_ai.training.resilience import OOM_EXIT_CODE

logger = logging.getLogger(__name__)

_trainer_ref = None  # Global ref for signal handler


def _save_checkpoint_if_possible(reason: str) -> None:
    """Attempt to save a training checkpoint."""
    global _trainer_ref
    if _trainer_ref is None:
        logger.warning("No trainer ref available, cannot save checkpoint on %s", reason)
        return

    try:
        output_dir = getattr(_trainer_ref, "args", None)
        if output_dir and hasattr(output_dir, "output_dir"):
            save_path = Path(output_dir.output_dir) / f"checkpoint-{reason}"
        else:
            save_path = Path("./checkpoint-emergency")

        logger.info("Saving emergency checkpoint to %s", save_path)
        _trainer_ref.save_model(str(save_path))
        logger.info("Emergency checkpoint saved successfully")
    except Exception as exc:
        logger.error("Failed to save emergency checkpoint: %s", exc)


def _sigterm_handler(signum: int, frame: object) -> None:
    """Handle SIGTERM: save checkpoint and exit gracefully."""
    logger.warning("SIGTERM received — saving checkpoint and shutting down")
    _save_checkpoint_if_possible("sigterm")
    sys.exit(0)


def main() -> None:
    """Parse args and run training with resilience."""
    global _trainer_ref

    parser = argparse.ArgumentParser(description="Distributed training entry point")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Install signal handlers
    signal.signal(signal.SIGTERM, _sigterm_handler)

    task = config.get("task", "sft")
    logger.info("Distributed training: task=%s", task)

    try:
        if task == "sft":
            from pulsar_ai.training.sft import train_sft
            results = train_sft(config)
        elif task == "dpo":
            from pulsar_ai.training.dpo import train_dpo
            results = train_dpo(config)
        elif task == "grpo":
            from pulsar_ai.training.grpo import train_grpo
            results = train_grpo(config)
        else:
            logger.error("Unknown task: %s", task)
            sys.exit(1)

        logger.info("Training complete: %s", results)

    except (RuntimeError, Exception) as exc:
        error_str = str(exc)

        # Check for CUDA OOM
        if "CUDA out of memory" in error_str or "OutOfMemoryError" in error_str:
            logger.error("CUDA OOM detected — saving checkpoint and exiting with code %d", OOM_EXIT_CODE)
            _save_checkpoint_if_possible("oom")
            sys.exit(OOM_EXIT_CODE)

        # Check for NCCL errors (retryable)
        if "NCCL" in error_str:
            logger.error("NCCL error — saving checkpoint and exiting with code %d", OOM_EXIT_CODE)
            _save_checkpoint_if_possible("nccl-error")
            sys.exit(OOM_EXIT_CODE)

        # Non-retryable error
        logger.exception("Training failed with non-retryable error")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
