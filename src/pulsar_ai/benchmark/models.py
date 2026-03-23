"""Data models for the benchmark system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """A single benchmark run result."""

    id: str
    model_path: str
    model_name: str
    experiment_id: str = ""
    hardware_info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    # Speed metrics
    tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    training_samples_per_sec: float = 0.0

    # Memory metrics
    peak_vram_gb: float = 0.0
    model_size_params: int = 0
    model_size_disk_mb: float = 0.0

    # Quality metrics
    perplexity: float | None = None
    eval_loss: float | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)

    # Cost metrics
    estimated_cost_per_1m_tokens: float = 0.0

    # Metadata
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    tags: list[str] = field(default_factory=list)
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage."""
        return {
            "id": self.id,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "experiment_id": self.experiment_id,
            "hardware_info": json.dumps(self.hardware_info, ensure_ascii=False),
            "timestamp": self.timestamp,
            "tokens_per_sec": self.tokens_per_sec,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "training_samples_per_sec": self.training_samples_per_sec,
            "peak_vram_gb": self.peak_vram_gb,
            "model_size_params": self.model_size_params,
            "model_size_disk_mb": self.model_size_disk_mb,
            "perplexity": self.perplexity,
            "eval_loss": self.eval_loss,
            "task_metrics": json.dumps(self.task_metrics, ensure_ascii=False),
            "estimated_cost_per_1m_tokens": self.estimated_cost_per_1m_tokens,
            "config": json.dumps(self.config, ensure_ascii=False),
            "status": self.status,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "is_baseline": 1 if self.is_baseline else 0,
        }
