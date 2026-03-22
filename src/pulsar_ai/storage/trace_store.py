"""Persistent storage for agent execution traces and feedback.

Bridges the gap between live agent runs and training data:
traces are saved to SQLite, users attach feedback, and the
store exports SFT / DPO examples via ``data_gen`` helpers.
"""

import json
import logging
import uuid
from itertools import product
from typing import Any

from pulsar_ai.storage.database import Database

logger = logging.getLogger(__name__)


class TraceStore:
    """Persistent storage for agent execution traces and feedback.

    Args:
        db: Optional Database instance. Uses a new one if omitted.
    """

    def __init__(self, db: Database | None = None) -> None:
        if db is not None:
            self._db = db
        else:
            from pulsar_ai.storage.database import get_database

            self._db = get_database()

    # ── Traces ────────────────────────────────────────────────────

    def save_trace(self, trace_data: dict[str, Any]) -> str:
        """Save an agent trace.

        Args:
            trace_data: Dict with keys matching the ``traces`` table
                columns.  ``user_query`` is required; everything else
                has sensible defaults.

        Returns:
            The generated ``trace_id``.
        """
        trace_id = uuid.uuid4().hex[:12]
        trace_json = trace_data.get("trace_json", [])
        if isinstance(trace_json, (list, dict)):
            trace_json = json.dumps(trace_json)

        self._db.execute(
            """
            INSERT INTO traces
                (trace_id, agent_id, model_name, model_version,
                 user_query, response, trace_json, status,
                 tokens_used, cost, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                trace_data.get("agent_id", ""),
                trace_data.get("model_name", ""),
                trace_data.get("model_version", ""),
                trace_data["user_query"],
                trace_data.get("response", ""),
                trace_json,
                trace_data.get("status", "success"),
                trace_data.get("tokens_used", 0),
                trace_data.get("cost", 0.0),
                trace_data.get("latency_ms", 0),
            ),
        )
        self._db.commit()
        logger.info("Saved trace %s", trace_id)
        return trace_id

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get a single trace by ID.

        Args:
            trace_id: Primary key of the trace.

        Returns:
            Row as dict (with ``trace_json`` parsed) or ``None``.
        """
        row = self._db.fetch_one(
            "SELECT * FROM traces WHERE trace_id = ?",
            (trace_id,),
        )
        if row is None:
            return None
        return self._parse_trace_row(row)

    def list_traces(
        self,
        *,
        date_from: str = "",
        date_to: str = "",
        model_name: str = "",
        status: str = "",
        min_rating: float | None = None,
        has_feedback: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List traces with optional filters.

        Args:
            date_from: ISO date lower bound (inclusive).
            date_to: ISO date upper bound (inclusive).
            model_name: Filter by model name.
            status: Filter by status string.
            min_rating: Only traces whose average feedback rating
                is at least this value.
            has_feedback: If ``True``, only traces with feedback;
                if ``False``, only without.
            limit: Max rows to return.
            offset: Pagination offset.

        Returns:
            List of trace dicts.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if date_from:
            clauses.append("t.created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("t.created_at <= ?")
            params.append(date_to)
        if model_name:
            clauses.append("t.model_name = ?")
            params.append(model_name)
        if status:
            clauses.append("t.status = ?")
            params.append(status)

        need_join = min_rating is not None or has_feedback is not None

        if need_join:
            base = (
                "SELECT t.*, AVG(f.rating) AS avg_rating, "
                "COUNT(f.id) AS feedback_count "
                "FROM traces t "
                "LEFT JOIN trace_feedback f "
                "ON t.trace_id = f.trace_id"
            )
        else:
            base = "SELECT t.* FROM traces t"

        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        sql = base + where

        if need_join:
            sql += " GROUP BY t.trace_id"
            having: list[str] = []
            if min_rating is not None:
                having.append("AVG(f.rating) >= ?")
                params.append(min_rating)
            if has_feedback is True:
                having.append("COUNT(f.id) > 0")
            elif has_feedback is False:
                having.append("COUNT(f.id) = 0")
            if having:
                sql += " HAVING " + " AND ".join(having)

        sql += " ORDER BY t.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._db.fetch_all(sql, tuple(params))
        return [self._parse_trace_row(r) for r in rows]

    # ── Feedback ──────────────────────────────────────────────────

    def add_feedback(
        self,
        trace_id: str,
        feedback_type: str,
        rating: float,
        reason: str = "",
        user_id: str = "",
    ) -> str:
        """Add feedback to a trace.

        Args:
            trace_id: The trace to annotate.
            feedback_type: E.g. ``'thumbs'``, ``'rating'``.
            rating: Numeric score.
            reason: Free-text explanation.
            user_id: Who gave the feedback.

        Returns:
            The generated feedback ``id``.
        """
        feedback_id = uuid.uuid4().hex[:12]
        self._db.execute(
            """
            INSERT INTO trace_feedback
                (id, trace_id, feedback_type, rating,
                 reason, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                trace_id,
                feedback_type,
                rating,
                reason,
                user_id,
            ),
        )
        self._db.commit()
        logger.info(
            "Added feedback %s to trace %s",
            feedback_id,
            trace_id,
        )
        return feedback_id

    def get_feedback(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all feedback for a trace.

        Args:
            trace_id: The trace to look up.

        Returns:
            List of feedback row dicts.
        """
        return self._db.fetch_all(
            "SELECT * FROM trace_feedback WHERE trace_id = ? " "ORDER BY created_at",
            (trace_id,),
        )

    # ── Export helpers ─────────────────────────────────────────────

    def export_as_sft(self, trace_ids: list[str]) -> list[dict[str, Any]]:
        """Export traces as SFT training examples.

        Uses ``data_gen.trace_to_sft`` for conversion.

        Args:
            trace_ids: IDs of traces to export.

        Returns:
            List of SFT example dicts.
        """
        from pulsar_ai.agent.data_gen import trace_to_sft

        results: list[dict[str, Any]] = []
        for tid in trace_ids:
            row = self.get_trace(tid)
            if row is None:
                continue
            example = trace_to_sft(
                row["trace_json"],
                row["user_query"],
            )
            if example is not None:
                results.append(example)
        return results

    def export_as_dpo(
        self,
        good_ids: list[str],
        bad_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Export trace pairs as DPO training examples.

        Each good trace is paired with each bad trace that
        shares the same ``user_query``.

        Args:
            good_ids: IDs of preferred traces.
            bad_ids: IDs of rejected traces.

        Returns:
            List of DPO pair dicts.
        """
        from pulsar_ai.agent.data_gen import trace_to_dpo_pair

        good_traces = [self.get_trace(tid) for tid in good_ids]
        bad_traces = [self.get_trace(tid) for tid in bad_ids]
        good_traces = [t for t in good_traces if t]
        bad_traces = [t for t in bad_traces if t]

        results: list[dict[str, Any]] = []
        for good, bad in product(good_traces, bad_traces):
            pair = trace_to_dpo_pair(
                good["trace_json"],
                bad["trace_json"],
                good["user_query"],
            )
            if pair is not None:
                results.append(pair)
        return results

    def auto_pair_dpo(self, trace_ids: list[str]) -> list[dict[str, Any]]:
        """Auto-generate DPO pairs by grouping similar queries.

        Traces with high rating (> 0.5) become ``chosen``;
        low rating (<= 0.5) become ``rejected``.  Groups are
        formed by normalising ``user_query`` (lowercase, stripped).

        Args:
            trace_ids: IDs of traces to consider.

        Returns:
            List of DPO pair dicts.
        """
        from pulsar_ai.agent.data_gen import trace_to_dpo_pair

        # Fetch traces with their average rating.
        traces_with_rating: list[tuple[dict[str, Any], float]] = []
        for tid in trace_ids:
            row = self.get_trace(tid)
            if row is None:
                continue
            fb = self.get_feedback(tid)
            avg_rating = 0.0
            if fb:
                avg_rating = sum(f["rating"] for f in fb) / len(fb)
            traces_with_rating.append((row, avg_rating))

        # Group by normalised query.
        groups: dict[
            str,
            dict[str, list[dict[str, Any]]],
        ] = {}
        for row, rating in traces_with_rating:
            key = row["user_query"].lower().strip()
            if key not in groups:
                groups[key] = {"good": [], "bad": []}
            if rating > 0.5:
                groups[key]["good"].append(row)
            else:
                groups[key]["bad"].append(row)

        # Generate pairs.
        results: list[dict[str, Any]] = []
        for _key, bucket in groups.items():
            for good, bad in product(bucket["good"], bucket["bad"]):
                pair = trace_to_dpo_pair(
                    good["trace_json"],
                    bad["trace_json"],
                    good["user_query"],
                )
                if pair is not None:
                    results.append(pair)
        return results

    # ── Statistics ─────────────────────────────────────────────────

    def get_stats(self, days: int = 30) -> dict[str, Any]:
        """Get trace statistics for the last *days* days.

        Args:
            days: How far back to look.

        Returns:
            Dict with ``total``, ``with_feedback``,
            ``avg_rating``, ``status_counts``,
            and ``traces_per_day`` keys.
        """
        total_row = self._db.fetch_one(
            "SELECT COUNT(*) AS cnt FROM traces " "WHERE created_at >= datetime('now', ?)",
            (f"-{days} days",),
        )
        total = total_row["cnt"] if total_row else 0

        fb_row = self._db.fetch_one(
            "SELECT COUNT(DISTINCT t.trace_id) AS cnt "
            "FROM traces t "
            "JOIN trace_feedback f ON t.trace_id = f.trace_id "
            "WHERE t.created_at >= datetime('now', ?)",
            (f"-{days} days",),
        )
        with_feedback = fb_row["cnt"] if fb_row else 0

        avg_row = self._db.fetch_one(
            "SELECT AVG(f.rating) AS avg_r "
            "FROM trace_feedback f "
            "JOIN traces t ON t.trace_id = f.trace_id "
            "WHERE t.created_at >= datetime('now', ?)",
            (f"-{days} days",),
        )
        avg_rating = round(avg_row["avg_r"], 4) if avg_row and avg_row["avg_r"] is not None else 0.0

        status_rows = self._db.fetch_all(
            "SELECT status, COUNT(*) AS cnt FROM traces "
            "WHERE created_at >= datetime('now', ?) "
            "GROUP BY status",
            (f"-{days} days",),
        )
        status_counts = {r["status"]: r["cnt"] for r in status_rows}

        per_day_rows = self._db.fetch_all(
            "SELECT DATE(created_at) AS day, COUNT(*) AS cnt "
            "FROM traces "
            "WHERE created_at >= datetime('now', ?) "
            "GROUP BY DATE(created_at) "
            "ORDER BY day",
            (f"-{days} days",),
        )
        traces_per_day = {r["day"]: r["cnt"] for r in per_day_rows}

        return {
            "total": total,
            "with_feedback": with_feedback,
            "avg_rating": avg_rating,
            "status_counts": status_counts,
            "traces_per_day": traces_per_day,
        }

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _parse_trace_row(
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """Deserialise ``trace_json`` from a raw DB row."""
        row = dict(row)
        raw = row.get("trace_json", "[]")
        if isinstance(raw, str):
            try:
                row["trace_json"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                row["trace_json"] = []
        return row
