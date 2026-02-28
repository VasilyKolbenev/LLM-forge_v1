"""JSON-based experiment tracker for Web UI."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("./data/experiments.json")


class ExperimentStore:
    """CRUD operations on experiments stored in a JSON file.

    Args:
        store_path: Path to the JSON file.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path or DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._save([])

    def create(self, name: str, config: dict, task: str = "sft") -> str:
        """Create a new experiment entry.

        Args:
            name: Experiment name.
            config: Full training config dict.
            task: Training task type.

        Returns:
            Experiment ID.
        """
        exp_id = str(uuid.uuid4())[:8]
        experiments = self._load()

        experiments.append({
            "id": exp_id,
            "name": name,
            "status": "queued",
            "task": task,
            "model": config.get("model", {}).get("name", "unknown"),
            "dataset_id": config.get("_dataset_id", ""),
            "config": config,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "final_loss": None,
            "training_history": [],
            "eval_results": None,
            "artifacts": {},
        })

        self._save(experiments)
        logger.info("Created experiment %s: %s", exp_id, name)
        return exp_id

    def update_status(self, exp_id: str, status: str) -> None:
        """Update experiment status.

        Args:
            exp_id: Experiment ID.
            status: New status (queued/running/completed/failed).
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["status"] = status
                if status == "completed":
                    exp["completed_at"] = datetime.now().isoformat()
                break
        self._save(experiments)

    def add_metrics(self, exp_id: str, metrics: dict) -> None:
        """Append training metrics to history.

        Args:
            exp_id: Experiment ID.
            metrics: Dict with step, loss, epoch, etc.
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["training_history"].append(metrics)
                if "loss" in metrics:
                    exp["final_loss"] = metrics["loss"]
                break
        self._save(experiments)

    def set_artifacts(self, exp_id: str, artifacts: dict) -> None:
        """Store artifact paths for an experiment.

        Args:
            exp_id: Experiment ID.
            artifacts: Dict of artifact paths (adapter_dir, output_dir, etc.).
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["artifacts"] = artifacts
                break
        self._save(experiments)

    def set_eval_results(self, exp_id: str, results: dict) -> None:
        """Store evaluation results.

        Args:
            exp_id: Experiment ID.
            results: Eval results dict.
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["eval_results"] = results
                break
        self._save(experiments)

    def get(self, exp_id: str) -> dict | None:
        """Get a single experiment by ID.

        Args:
            exp_id: Experiment ID.

        Returns:
            Experiment dict or None.
        """
        for exp in self._load():
            if exp["id"] == exp_id:
                return exp
        return None

    def list_all(self, status: str | None = None) -> list[dict]:
        """List all experiments, optionally filtered by status.

        Args:
            status: Optional status filter.

        Returns:
            List of experiment dicts (newest first).
        """
        experiments = self._load()
        if status:
            experiments = [e for e in experiments if e["status"] == status]
        return sorted(experiments, key=lambda e: e["created_at"], reverse=True)

    def delete(self, exp_id: str) -> bool:
        """Delete an experiment.

        Args:
            exp_id: Experiment ID.

        Returns:
            True if deleted, False if not found.
        """
        experiments = self._load()
        original_len = len(experiments)
        experiments = [e for e in experiments if e["id"] != exp_id]
        if len(experiments) < original_len:
            self._save(experiments)
            return True
        return False

    def _load(self) -> list[dict]:
        with open(self.store_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, experiments: list[dict]) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, ensure_ascii=False, indent=2)
