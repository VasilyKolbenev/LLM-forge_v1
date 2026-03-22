"""Lineage store — aggregates relationships across experiments, traces, datasets.

Reads from existing tables (experiments, traces, trace_feedback) to build
a graph of data lineage and a chronological timeline of lifecycle events.
No new tables are created.
"""

import hashlib
import json
import logging
from pathlib import PurePosixPath
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


class LineageStore:
    """Aggregates lineage relationships across experiments, traces, datasets.

    Args:
        db: Optional Database instance. Uses the singleton if omitted.
    """

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database

            self._db = get_database()

    # ── Graph ──────────────────────────────────────────────────────

    def get_lineage_graph(self, model_name: str | None = None) -> dict[str, Any]:
        """Build a lineage graph from existing data.

        Args:
            model_name: Optional filter — only include lineage touching
                this model.

        Returns:
            Dict with ``nodes`` and ``edges`` lists.
        """
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []

        # 1. Experiments → experiment nodes, dataset nodes, model nodes
        self._build_experiment_nodes(nodes, edges, model_name)

        # 2. Traces aggregated by model_name → trace collection nodes
        self._build_trace_nodes(nodes, edges, model_name)

        # 3. Feedback aggregated by model_name → feedback nodes
        self._build_feedback_nodes(nodes, edges, model_name)

        # 4. SFT→DPO chain edges
        self._build_chain_edges(nodes, edges)

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    # ── Timeline ───────────────────────────────────────────────────

    def get_timeline(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get chronological timeline of lifecycle events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dicts sorted newest-first.
        """
        events: list[dict[str, Any]] = []

        # Experiment events
        self._collect_experiment_events(events)

        # Trace events (daily aggregates per model)
        self._collect_trace_events(events)

        # Sort newest-first and limit
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        return events[:limit]

    # ── Private: graph builders ────────────────────────────────────

    def _build_experiment_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, str]],
        model_name: str | None,
    ) -> None:
        """Create experiment, dataset, and model nodes from the experiments table."""
        sql = (
            "SELECT id, name, task, status, model, dataset_id, config, "
            "created_at, final_loss FROM experiments ORDER BY created_at"
        )
        rows = self._db.fetch_all(sql)

        for row in rows:
            config = self._parse_json(row.get("config", "{}"))

            # If filtering by model_name, skip experiments that don't match
            if model_name and row.get("model", "") != model_name:
                output_dir = config.get("output_dir", "")
                if model_name not in output_dir:
                    continue

            exp_id = f"exp_{row['id']}"
            nodes[exp_id] = {
                "id": exp_id,
                "type": "experiment",
                "label": row["name"],
                "status": row["status"],
                "metadata": {
                    "task": row["task"],
                    "model": row.get("model", ""),
                    "loss": row.get("final_loss"),
                    "created_at": row["created_at"],
                },
            }

            # Dataset node from config or dataset_id
            ds_path = config.get("dataset_path", "") or row.get("dataset_id", "")
            if ds_path:
                ds_hash = hashlib.md5(ds_path.encode()).hexdigest()[:8]
                ds_id = f"ds_{ds_hash}"
                if ds_id not in nodes:
                    label = (
                        PurePosixPath(ds_path).name
                        if "/" in ds_path or "\\" in ds_path
                        else ds_path
                    )
                    nodes[ds_id] = {
                        "id": ds_id,
                        "type": "dataset",
                        "label": label,
                        "status": "ready",
                        "metadata": {"path": ds_path},
                    }
                edges.append({"source": ds_id, "target": exp_id, "label": "trains"})

            # Model node from output_dir
            output_dir = config.get("output_dir", "")
            if output_dir:
                model_id = f"model_{row['id']}"
                model_label = f"Model from {row['name']}"
                nodes[model_id] = {
                    "id": model_id,
                    "type": "model",
                    "label": model_label,
                    "status": "trained" if row["status"] == "completed" else row["status"],
                    "metadata": {"output_dir": output_dir},
                }
                edges.append({"source": exp_id, "target": model_id, "label": "produces"})

    def _build_trace_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, str]],
        model_name: str | None,
    ) -> None:
        """Create trace collection nodes aggregated by model_name."""
        sql = (
            "SELECT model_name, COUNT(*) as cnt, AVG(latency_ms) as avg_latency "
            "FROM traces GROUP BY model_name"
        )
        rows = self._db.fetch_all(sql)

        for row in rows:
            mname = row["model_name"] or "unknown"
            if model_name and mname != model_name:
                continue

            trace_id = f"traces_{mname}"
            nodes[trace_id] = {
                "id": trace_id,
                "type": "traces",
                "label": f"{row['cnt']} traces",
                "status": "active",
                "metadata": {
                    "count": row["cnt"],
                    "avg_latency": round(row["avg_latency"] or 0, 2),
                    "model_name": mname,
                },
            }

            # Connect model → traces (find matching model node)
            model_node_id = self._find_model_node_for(nodes, mname)
            if model_node_id:
                edges.append({"source": model_node_id, "target": trace_id, "label": "serves"})

    def _build_feedback_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, str]],
        model_name: str | None,
    ) -> None:
        """Create feedback summary nodes aggregated by model_name."""
        sql = (
            "SELECT t.model_name, "
            "SUM(CASE WHEN tf.rating > 0.5 THEN 1 ELSE 0 END) as thumbs_up, "
            "SUM(CASE WHEN tf.rating <= 0.5 THEN 1 ELSE 0 END) as thumbs_down "
            "FROM traces t "
            "JOIN trace_feedback tf ON t.trace_id = tf.trace_id "
            "GROUP BY t.model_name"
        )
        rows = self._db.fetch_all(sql)

        for row in rows:
            mname = row["model_name"] or "unknown"
            if model_name and mname != model_name:
                continue

            fb_id = f"fb_{mname}"
            thumbs_up = row["thumbs_up"] or 0
            thumbs_down = row["thumbs_down"] or 0
            nodes[fb_id] = {
                "id": fb_id,
                "type": "feedback",
                "label": f"+{thumbs_up} / -{thumbs_down}",
                "status": "active",
                "metadata": {
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "model_name": mname,
                },
            }

            # Connect traces → feedback
            trace_node_id = f"traces_{mname}"
            if trace_node_id in nodes:
                edges.append({"source": trace_node_id, "target": fb_id, "label": "feedback"})

    def _build_chain_edges(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, str]],
    ) -> None:
        """Connect SFT model → DPO experiment if sft_adapter_path is in config."""
        sql = "SELECT id, config FROM experiments WHERE task = 'dpo'"
        rows = self._db.fetch_all(sql)

        for row in rows:
            config = self._parse_json(row.get("config", "{}"))
            sft_path = config.get("sft_adapter_path", "")
            if not sft_path:
                continue

            dpo_exp_id = f"exp_{row['id']}"
            if dpo_exp_id not in nodes:
                continue

            # Find the SFT model node whose output_dir matches
            for node in nodes.values():
                if node["type"] == "model" and node["metadata"].get("output_dir") == sft_path:
                    edges.append(
                        {
                            "source": node["id"],
                            "target": dpo_exp_id,
                            "label": "sft_base",
                        }
                    )
                    break

    # ── Private: timeline collectors ───────────────────────────────

    def _collect_experiment_events(self, events: list[dict[str, Any]]) -> None:
        """Add experiment lifecycle events to the events list."""
        rows = self._db.fetch_all(
            "SELECT id, name, status, created_at, completed_at FROM experiments"
        )
        for row in rows:
            # Created event
            events.append(
                {
                    "timestamp": row["created_at"],
                    "event_type": "experiment_created",
                    "title": row["name"],
                    "description": f"Experiment '{row['name']}' created",
                    "entity_id": row["id"],
                    "entity_type": "experiment",
                }
            )

            # Completed/failed event
            if row["status"] == "completed" and row.get("completed_at"):
                events.append(
                    {
                        "timestamp": row["completed_at"],
                        "event_type": "experiment_completed",
                        "title": row["name"],
                        "description": f"Experiment '{row['name']}' completed",
                        "entity_id": row["id"],
                        "entity_type": "experiment",
                    }
                )
            elif row["status"] == "failed" and row.get("completed_at"):
                events.append(
                    {
                        "timestamp": row["completed_at"],
                        "event_type": "experiment_failed",
                        "title": row["name"],
                        "description": f"Experiment '{row['name']}' failed",
                        "entity_id": row["id"],
                        "entity_type": "experiment",
                    }
                )

    def _collect_trace_events(self, events: list[dict[str, Any]]) -> None:
        """Add daily trace aggregation events to the events list."""
        rows = self._db.fetch_all(
            "SELECT DATE(created_at) as day, model_name, COUNT(*) as cnt "
            "FROM traces GROUP BY day, model_name ORDER BY day"
        )
        for row in rows:
            mname = row["model_name"] or "unknown"
            events.append(
                {
                    "timestamp": row["day"],
                    "event_type": "traces_recorded",
                    "title": f"{row['cnt']} traces for {mname}",
                    "description": f"{row['cnt']} traces recorded for model '{mname}'",
                    "entity_id": mname,
                    "entity_type": "traces",
                }
            )

    # ── Private: helpers ───────────────────────────────────────────

    @staticmethod
    def _find_model_node_for(
        nodes: dict[str, dict[str, Any]],
        model_name: str,
    ) -> str | None:
        """Find a model node that matches the given model name."""
        for node in nodes.values():
            if node["type"] != "model":
                continue
            output_dir = node["metadata"].get("output_dir", "")
            if model_name in output_dir:
                return node["id"]
        return None

    @staticmethod
    def _parse_json(raw: str | dict[str, Any]) -> dict[str, Any]:
        """Safely parse a JSON string; return empty dict on failure."""
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            return {}
