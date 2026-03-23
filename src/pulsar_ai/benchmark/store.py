"""Persistent storage for benchmark results.

Stores BenchmarkResult data in SQLite via the shared Database layer,
enabling historical comparison and leaderboard ranking.
"""

import json
import logging
import uuid
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)

# Metrics that can be used in leaderboard queries (allowlist for SQL safety)
ALLOWED_METRICS = {
    "tokens_per_sec",
    "time_to_first_token_ms",
    "training_samples_per_sec",
    "peak_vram_gb",
    "model_size_params",
    "model_size_disk_mb",
    "perplexity",
    "eval_loss",
    "estimated_cost_per_1m_tokens",
}


class BenchmarkStore:
    """Store and retrieve benchmark results."""

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database
            self._db = get_database()

    def save(self, result: Any, user_id: str = "") -> str:
        """Save a BenchmarkResult to the database.

        Args:
            result: A BenchmarkResult instance.
            user_id: Owner user ID.

        Returns:
            Generated benchmark ID.
        """
        benchmark_id = result.id or uuid.uuid4().hex[:12]

        data = result.to_dict()
        data["id"] = benchmark_id

        self._db.execute(
            """
            INSERT OR REPLACE INTO benchmarks
                (id, model_path, model_name, experiment_id, hardware_info,
                 timestamp, tokens_per_sec, time_to_first_token_ms,
                 training_samples_per_sec, peak_vram_gb, model_size_params,
                 model_size_disk_mb, perplexity, eval_loss, task_metrics,
                 estimated_cost_per_1m_tokens, config, status, tags, is_baseline,
                 user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"], data["model_path"], data["model_name"],
                data["experiment_id"], data["hardware_info"],
                data["timestamp"], data["tokens_per_sec"],
                data["time_to_first_token_ms"], data["training_samples_per_sec"],
                data["peak_vram_gb"], data["model_size_params"],
                data["model_size_disk_mb"], data["perplexity"],
                data["eval_loss"], data["task_metrics"],
                data["estimated_cost_per_1m_tokens"], data["config"],
                data["status"], data["tags"], data["is_baseline"],
                user_id,
            ),
        )
        self._db.commit()
        logger.info("Saved benchmark %s for model %s", benchmark_id, data["model_name"])
        return benchmark_id

    def get(self, benchmark_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Get a single benchmark by ID."""
        sql = "SELECT * FROM benchmarks WHERE id = ?"
        params: list[Any] = [benchmark_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        row = self._db.fetch_one(sql, tuple(params))
        if row is None:
            return None
        return self._parse_row(row)

    def list_all(
        self,
        model_name: str | None = None,
        experiment_id: str | None = None,
        is_baseline: bool | None = None,
        limit: int = 50,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List benchmark results with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if is_baseline is not None:
            clauses.append("is_baseline = ?")
            params.append(1 if is_baseline else 0)

        sql = "SELECT * FROM benchmarks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._db.fetch_all(sql, tuple(params))
        return [self._parse_row(r) for r in rows]

    def compare(self, ids: list[str]) -> dict[str, Any]:
        """Compare multiple benchmark runs.

        Args:
            ids: List of benchmark IDs (first is treated as baseline).

        Returns:
            Dict with baseline, candidates, and deltas.

        Raises:
            ValueError: If any ID not found or less than 2 IDs.
        """
        if len(ids) < 2:
            raise ValueError("Need at least 2 benchmark IDs to compare")

        benchmarks = []
        for bid in ids:
            b = self.get(bid)
            if b is None:
                raise ValueError(f"Benchmark not found: {bid}")
            benchmarks.append(b)

        baseline = benchmarks[0]
        candidates = benchmarks[1:]
        deltas = []

        metric_keys = [
            "tokens_per_sec", "time_to_first_token_ms", "peak_vram_gb",
            "perplexity", "estimated_cost_per_1m_tokens",
        ]

        for cand in candidates:
            delta: dict[str, Any] = {"id": cand["id"], "model_name": cand["model_name"]}
            for key in metric_keys:
                bv = baseline.get(key) or 0
                cv = cand.get(key) or 0
                delta[key] = round(cv - bv, 4)
            deltas.append(delta)

        return {
            "baseline": baseline,
            "candidates": candidates,
            "deltas": deltas,
        }

    def leaderboard(
        self,
        metric: str = "tokens_per_sec",
        order: str = "desc",
        limit: int = 20,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get ranked list of benchmarks by a metric.

        Args:
            metric: Column name to sort by (validated against allowlist).
            order: 'asc' or 'desc'.
            limit: Max results.
            user_id: Optional user filter.

        Returns:
            Sorted list of benchmark dicts.
        """
        if metric not in ALLOWED_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Allowed: {ALLOWED_METRICS}")

        direction = "ASC" if order == "asc" else "DESC"
        where = f"{metric} IS NOT NULL"
        params: list[Any] = []
        if user_id is not None:
            where += " AND user_id = ?"
            params.append(user_id)
        sql = f"SELECT * FROM benchmarks WHERE {where} ORDER BY {metric} {direction} LIMIT ?"
        params.append(limit)

        rows = self._db.fetch_all(sql, tuple(params))
        return [self._parse_row(r) for r in rows]

    def delete(self, benchmark_id: str, user_id: str | None = None) -> bool:
        """Delete a benchmark result."""
        sql = "DELETE FROM benchmarks WHERE id = ?"
        params: list[Any] = [benchmark_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        self._db.execute(sql, tuple(params))
        self._db.commit()
        return True

    @staticmethod
    def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON fields from a raw DB row."""
        row = dict(row)
        for field in ("hardware_info", "task_metrics", "config"):
            raw = row.get(field, "{}")
            if isinstance(raw, str):
                try:
                    row[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    row[field] = {}
        raw_tags = row.get("tags", "[]")
        if isinstance(raw_tags, str):
            try:
                row["tags"] = json.loads(raw_tags)
            except (json.JSONDecodeError, TypeError):
                row["tags"] = []
        row["is_baseline"] = bool(row.get("is_baseline", 0))
        return row
