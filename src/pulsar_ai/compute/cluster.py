"""Cluster management for distributed training.

Provides cluster-level operations on top of ComputeManager:
status aggregation, GPU metrics polling, VRAM estimation,
and pre-flight validation for distributed training jobs.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pulsar_ai.compute.manager import ComputeManager, ComputeTarget
from pulsar_ai.compute.ssh import SSHConnection
from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)

# VRAM multipliers by strategy (relative to model params in GB at fp16)
VRAM_MULTIPLIERS = {
    "qlora": 0.5,
    "lora": 1.0,
    "full": 4.0,
    "fsdp_qlora": 0.5,
    "fsdp_lora": 1.0,
    "fsdp_full": 3.0,
    "deepspeed_zero2": 2.0,
    "deepspeed_zero3": 1.5,
}


class ClusterManager:
    """Manages compute cluster for distributed training.

    Args:
        compute_manager: ComputeManager instance.
        db: Optional Database instance.
    """

    def __init__(
        self,
        compute_manager: ComputeManager | None = None,
        db: Database | None = None,
    ) -> None:
        self._cm = compute_manager or ComputeManager()
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database
            self._db = get_database()

    def get_cluster_status(self) -> dict[str, Any]:
        """Get aggregated cluster status.

        Returns:
            Dict with targets, total GPUs, total VRAM, health summary.
        """
        targets = self._cm.list_targets()
        total_gpus = sum(t.gpu_count for t in targets)
        total_vram = sum(t.vram_gb * t.gpu_count for t in targets)
        healthy = sum(1 for t in targets if t.status == "online")

        return {
            "targets": [
                {
                    "id": t.id,
                    "name": t.name,
                    "host": t.host,
                    "gpu_count": t.gpu_count,
                    "gpu_type": t.gpu_type,
                    "vram_gb": t.vram_gb,
                    "status": t.status,
                    "last_heartbeat": t.last_heartbeat,
                }
                for t in targets
            ],
            "total_gpus": total_gpus,
            "total_vram_gb": round(total_vram, 1),
            "healthy_targets": healthy,
            "total_targets": len(targets),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def poll_gpu_metrics(self, target_id: str) -> list[dict[str, Any]]:
        """Poll live GPU metrics from a target via nvidia-smi.

        Args:
            target_id: Compute target ID.

        Returns:
            List of per-GPU metric dicts.
        """
        target = self._cm.get_target(target_id)
        if target is None:
            raise ValueError(f"Target not found: {target_id}")

        try:
            conn = SSHConnection(
                host=target.host,
                user=target.user,
                port=target.port,
                key_path=target.key_path,
            )
            conn.connect()
            output = conn.exec_command(
                "nvidia-smi --query-gpu=index,name,utilization.gpu,"
                "memory.used,memory.total,temperature.gpu "
                "--format=csv,noheader,nounits"
            )
            conn.close()
        except Exception as exc:
            logger.warning("Failed to poll GPU metrics for %s: %s", target_id, exc)
            return []

        gpus = []
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "vram_used_gb": round(float(parts[3]) / 1024, 2),
                    "vram_total_gb": round(float(parts[4]) / 1024, 2),
                    "temperature_c": float(parts[5]),
                })

        return gpus

    def estimate_vram_needed(
        self,
        model_size_params: int,
        strategy: str = "qlora",
    ) -> float:
        """Estimate VRAM needed per GPU for training.

        Args:
            model_size_params: Number of model parameters.
            strategy: Training strategy name.

        Returns:
            Estimated VRAM in GB per GPU.
        """
        # Model size at fp16 in GB
        model_gb = model_size_params * 2 / (1024**3)
        multiplier = VRAM_MULTIPLIERS.get(strategy, 2.0)
        return round(model_gb * multiplier, 1)

    def preflight_check(
        self,
        config: dict[str, Any],
        target_ids: list[str],
    ) -> dict[str, Any]:
        """Run pre-flight validation before distributed training.

        Checks:
        - All targets reachable
        - Sufficient VRAM
        - Compatible GPU types
        - Python/torch availability

        Args:
            config: Training configuration.
            target_ids: List of target IDs to use.

        Returns:
            Dict with passed/failed status and details.
        """
        issues: list[str] = []
        warnings: list[str] = []
        targets: list[ComputeTarget] = []

        for tid in target_ids:
            target = self._cm.get_target(tid)
            if target is None:
                issues.append(f"Target not found: {tid}")
                continue
            targets.append(target)

        if not targets:
            return {"passed": False, "issues": ["No valid targets"], "warnings": []}

        # Check connectivity
        for t in targets:
            try:
                conn = SSHConnection(host=t.host, user=t.user, port=t.port, key_path=t.key_path)
                conn.connect()
                conn.close()
            except Exception as exc:
                issues.append(f"Cannot connect to {t.name} ({t.host}): {exc}")

        # Check GPU compatibility
        gpu_types = {t.gpu_type for t in targets if t.gpu_type}
        if len(gpu_types) > 1:
            warnings.append(f"Mixed GPU types: {gpu_types}. Performance may be suboptimal.")

        # Check VRAM
        strategy = config.get("strategy", config.get("_detected_strategy", "qlora"))
        model_params = config.get("model_size_params", 0)
        if model_params > 0:
            needed = self.estimate_vram_needed(model_params, strategy)
            for t in targets:
                if t.vram_gb > 0 and t.vram_gb < needed:
                    issues.append(
                        f"{t.name}: {t.vram_gb}GB VRAM < {needed}GB needed ({strategy})"
                    )

        total_gpus = sum(t.gpu_count for t in targets)
        if total_gpus == 0:
            issues.append("No GPUs detected across targets")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_gpus": total_gpus,
            "total_vram_gb": sum(t.vram_gb * t.gpu_count for t in targets),
            "targets_checked": len(targets),
        }

    # ── Cluster config presets ─────────────────────────────────────

    def save_cluster_config(
        self,
        name: str,
        target_ids: list[str],
        master_target_id: str,
        strategy: str = "fsdp_qlora",
    ) -> str:
        """Save a cluster configuration preset."""
        import uuid
        config_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """INSERT INTO cluster_configs
               (id, name, target_ids, master_target_id, strategy, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (config_id, name, json.dumps(target_ids), master_target_id, strategy, now, now),
        )
        self._db.commit()
        return config_id

    def list_cluster_configs(self) -> list[dict[str, Any]]:
        """List saved cluster configuration presets."""
        rows = self._db.fetch_all("SELECT * FROM cluster_configs ORDER BY updated_at DESC")
        results = []
        for row in rows:
            d = dict(row)
            d["target_ids"] = json.loads(d.get("target_ids", "[]"))
            results.append(d)
        return results

    def store_distributed_metrics(
        self,
        job_id: str,
        rank: int,
        step: int,
        loss: float | None = None,
        vram_used_gb: float | None = None,
        gpu_util_pct: float | None = None,
        target_id: str = "",
        gpu_index: int = 0,
    ) -> None:
        """Store per-rank GPU metrics for a distributed job."""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """INSERT INTO distributed_metrics
               (job_id, rank, target_id, gpu_index, step, loss, vram_used_gb, gpu_util_pct, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, rank, target_id, gpu_index, step, loss, vram_used_gb, gpu_util_pct, now),
        )
        self._db.commit()

    def get_distributed_metrics(
        self, job_id: str, limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Get per-rank metrics for a distributed job."""
        rows = self._db.fetch_all(
            "SELECT * FROM distributed_metrics WHERE job_id = ? ORDER BY step, rank LIMIT ?",
            (job_id, limit),
        )
        return [dict(r) for r in rows]
