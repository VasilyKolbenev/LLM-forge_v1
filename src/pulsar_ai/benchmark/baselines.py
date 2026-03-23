"""Pre-computed baseline benchmarks for common models.

These provide reference points so users can compare their
fine-tuned models against well-known baselines.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.benchmark.models import BenchmarkResult

logger = logging.getLogger(__name__)

BASELINES: dict[str, dict[str, Any]] = {
    "llama-3-8b": {
        "model_name": "Llama 3 8B",
        "model_path": "meta-llama/Meta-Llama-3-8B",
        "tokens_per_sec": 45.0,
        "time_to_first_token_ms": 85.0,
        "peak_vram_gb": 16.2,
        "model_size_params": 8_030_000_000,
        "model_size_disk_mb": 15400,
        "perplexity": 6.14,
        "eval_loss": 1.81,
        "estimated_cost_per_1m_tokens": 0.20,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 1},
    },
    "mistral-7b-v0.3": {
        "model_name": "Mistral 7B v0.3",
        "model_path": "mistralai/Mistral-7B-v0.3",
        "tokens_per_sec": 52.0,
        "time_to_first_token_ms": 72.0,
        "peak_vram_gb": 14.5,
        "model_size_params": 7_248_000_000,
        "model_size_disk_mb": 13800,
        "perplexity": 5.32,
        "eval_loss": 1.67,
        "estimated_cost_per_1m_tokens": 0.18,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 1},
    },
    "llama-3-70b": {
        "model_name": "Llama 3 70B",
        "model_path": "meta-llama/Meta-Llama-3-70B",
        "tokens_per_sec": 12.0,
        "time_to_first_token_ms": 320.0,
        "peak_vram_gb": 140.0,
        "model_size_params": 70_553_000_000,
        "model_size_disk_mb": 131000,
        "perplexity": 3.12,
        "eval_loss": 1.14,
        "estimated_cost_per_1m_tokens": 1.20,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 4},
    },
    "phi-3-mini": {
        "model_name": "Phi-3 Mini 3.8B",
        "model_path": "microsoft/Phi-3-mini-4k-instruct",
        "tokens_per_sec": 78.0,
        "time_to_first_token_ms": 45.0,
        "peak_vram_gb": 7.8,
        "model_size_params": 3_821_000_000,
        "model_size_disk_mb": 7600,
        "perplexity": 8.41,
        "eval_loss": 2.13,
        "estimated_cost_per_1m_tokens": 0.08,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 1},
    },
    "gemma-2-9b": {
        "model_name": "Gemma 2 9B",
        "model_path": "google/gemma-2-9b",
        "tokens_per_sec": 38.0,
        "time_to_first_token_ms": 95.0,
        "peak_vram_gb": 18.4,
        "model_size_params": 9_241_000_000,
        "model_size_disk_mb": 17800,
        "perplexity": 5.89,
        "eval_loss": 1.77,
        "estimated_cost_per_1m_tokens": 0.22,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 1},
    },
    "qwen-2.5-7b": {
        "model_name": "Qwen 2.5 7B",
        "model_path": "Qwen/Qwen2.5-7B",
        "tokens_per_sec": 55.0,
        "time_to_first_token_ms": 68.0,
        "peak_vram_gb": 14.8,
        "model_size_params": 7_615_000_000,
        "model_size_disk_mb": 14200,
        "perplexity": 5.01,
        "eval_loss": 1.61,
        "estimated_cost_per_1m_tokens": 0.17,
        "hardware_info": {"gpu_name": "A100-80GB", "vram_gb": 80, "num_gpus": 1},
    },
}


def seed_baselines(store: Any) -> int:
    """Insert baseline benchmarks into the store if not already present.

    Args:
        store: BenchmarkStore instance.

    Returns:
        Number of baselines seeded.
    """
    existing = store.list_all(is_baseline=True, limit=100)
    existing_names = {b["model_name"] for b in existing}
    count = 0

    now = datetime.now(timezone.utc).isoformat()

    for key, data in BASELINES.items():
        if data["model_name"] in existing_names:
            continue

        result = BenchmarkResult(
            id=f"baseline-{key}",
            model_path=data["model_path"],
            model_name=data["model_name"],
            timestamp=now,
            tokens_per_sec=data["tokens_per_sec"],
            time_to_first_token_ms=data["time_to_first_token_ms"],
            peak_vram_gb=data["peak_vram_gb"],
            model_size_params=data["model_size_params"],
            model_size_disk_mb=data["model_size_disk_mb"],
            perplexity=data.get("perplexity"),
            eval_loss=data.get("eval_loss"),
            estimated_cost_per_1m_tokens=data["estimated_cost_per_1m_tokens"],
            hardware_info=data["hardware_info"],
            is_baseline=True,
            status="completed",
        )
        store.save(result)
        count += 1
        logger.info("Seeded baseline: %s", data["model_name"])

    return count
