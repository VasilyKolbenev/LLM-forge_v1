"""Persistent storage for agent evaluation reports.

Stores EvalReport data in SQLite via the shared Database layer,
enabling historical comparison of agent performance across models,
suites, and time.
"""

import json
import logging
import uuid
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


class AgentEvalStore:
    """Store and retrieve agent evaluation reports.

    Args:
        db: Optional Database instance. Uses the singleton if omitted.
    """

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database

            self._db = get_database()

    def save_report(self, report: Any) -> str:
        """Save an EvalReport to the database.

        Args:
            report: An EvalReport instance with results populated.

        Returns:
            Generated report ID.
        """
        report_id = uuid.uuid4().hex[:12]

        results_json = json.dumps(
            [r.to_dict() for r in report.results],
            ensure_ascii=False,
        )
        by_tag_json = json.dumps(report.by_tag, ensure_ascii=False)

        self._db.execute(
            """
            INSERT INTO agent_eval_reports
                (id, suite_name, model_name, timestamp,
                 success_rate, avg_score, avg_latency_ms,
                 total_tokens, total_cost, tools_accuracy,
                 results_json, by_tag_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                report.suite_name,
                report.model_name,
                report.timestamp,
                report.success_rate,
                report.avg_score,
                report.avg_latency_ms,
                report.total_tokens,
                report.total_cost,
                report.tools_accuracy,
                results_json,
                by_tag_json,
            ),
        )
        self._db.commit()
        logger.info("Saved eval report %s", report_id)
        return report_id

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Get a single report by ID.

        Args:
            report_id: Primary key of the report.

        Returns:
            Report as dict with parsed JSON fields, or None.
        """
        row = self._db.fetch_one(
            "SELECT * FROM agent_eval_reports WHERE id = ?",
            (report_id,),
        )
        if row is None:
            return None
        return self._parse_report_row(row)

    def list_reports(
        self,
        model_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent evaluation reports.

        Args:
            model_name: Optional filter by model name.
            limit: Maximum number of reports to return.

        Returns:
            List of report dicts ordered by timestamp desc.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)

        sql = "SELECT * FROM agent_eval_reports"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._db.fetch_all(sql, tuple(params))
        return [self._parse_report_row(r) for r in rows]

    def get_comparison(
        self,
        report_id_a: str,
        report_id_b: str,
    ) -> dict[str, Any]:
        """Compare two stored reports.

        Args:
            report_id_a: ID of the baseline report.
            report_id_b: ID of the candidate report.

        Returns:
            Comparison dict with deltas and winner.

        Raises:
            ValueError: If either report is not found.
        """
        a = self.get_report(report_id_a)
        b = self.get_report(report_id_b)

        if a is None:
            raise ValueError(f"Report not found: {report_id_a}")
        if b is None:
            raise ValueError(f"Report not found: {report_id_b}")

        success_delta = b["success_rate"] - a["success_rate"]
        score_delta = b["avg_score"] - a["avg_score"]
        latency_delta = b["avg_latency_ms"] - a["avg_latency_ms"]
        cost_delta = b["total_cost"] - a["total_cost"]

        if score_delta > 0.01:
            winner = "B"
        elif score_delta < -0.01:
            winner = "A"
        elif success_delta > 0:
            winner = "B"
        elif success_delta < 0:
            winner = "A"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "report_a": report_id_a,
            "report_b": report_id_b,
            "model_a": a["model_name"],
            "model_b": b["model_name"],
            "success_delta": round(success_delta, 4),
            "score_delta": round(score_delta, 4),
            "latency_delta": round(latency_delta, 2),
            "cost_delta": round(cost_delta, 6),
        }

    @staticmethod
    def _parse_report_row(
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """Deserialize JSON fields from a raw DB row."""
        row = dict(row)
        for field in ("results_json", "by_tag_json"):
            raw = row.get(field, "")
            if isinstance(raw, str):
                try:
                    row[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    row[field] = [] if field == "results_json" else {}
        return row
