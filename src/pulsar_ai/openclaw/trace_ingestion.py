"""OpenClaw trace ingestion into Pulsar AI TraceStore.

Converts OpenClaw execution traces into the Pulsar AI trace format
and persists them for feedback, dataset building, and analysis.
"""

import logging
from typing import Any

from pulsar_ai.storage.trace_store import TraceStore

logger = logging.getLogger(__name__)


class OpenClawTraceIngester:
    """Ingests execution traces from OpenClaw sessions into Pulsar AI TraceStore.

    Args:
        adapter: Backend providing get_session / get_trace / list_sessions.
            Accepts OpenClawAdapter (HTTP) or OpenClawRuntime (built-in).
        trace_store: TraceStore instance for persisting traces.
    """

    def __init__(self, adapter: Any, trace_store: TraceStore) -> None:
        self._adapter = adapter
        self._store = trace_store

    def ingest_session(self, session_id: str) -> list[str]:
        """Ingest all traces from an OpenClaw session.

        Fetches session details and execution trace from OpenClaw,
        converts each trace step to Pulsar AI format, and saves
        to TraceStore.

        Args:
            session_id: OpenClaw session ID to ingest.

        Returns:
            List of trace_ids saved to TraceStore.
        """
        session = self._adapter.get_session(session_id)
        if not session:
            logger.warning("Session %s not found, skipping ingestion", session_id)
            return []

        openclaw_trace = self._adapter.get_trace(session_id)
        if not openclaw_trace:
            logger.info("No trace data for session %s", session_id)
            return []

        trace_ids: list[str] = []
        for step in openclaw_trace:
            pulsar_trace = self._convert_trace(step, session)
            trace_id = self._store.save_trace(pulsar_trace)
            trace_ids.append(trace_id)
            logger.debug(
                "Ingested trace step %s from session %s",
                trace_id,
                session_id,
            )

        logger.info(
            "Ingested %d traces from session %s",
            len(trace_ids),
            session_id,
        )
        return trace_ids

    def ingest_all_sessions(self, status: str = "completed") -> dict[str, int]:
        """Ingest traces from all sessions with given status.

        Args:
            status: Filter sessions by this status before ingesting.

        Returns:
            Dict with sessions_processed and traces_ingested counts.
        """
        sessions = self._adapter.list_sessions(status=status)
        sessions_processed = 0
        traces_ingested = 0

        for session in sessions:
            trace_ids = self.ingest_session(session.session_id)
            if trace_ids:
                sessions_processed += 1
                traces_ingested += len(trace_ids)

        logger.info(
            "Bulk ingestion complete: %d sessions, %d traces",
            sessions_processed,
            traces_ingested,
        )
        return {
            "sessions_processed": sessions_processed,
            "traces_ingested": traces_ingested,
        }

    def _convert_trace(
        self,
        openclaw_trace: dict[str, Any],
        session: Any,
    ) -> dict[str, Any]:
        """Convert OpenClaw trace format to Pulsar AI trace format.

        Maps OpenClaw trace step fields to the TraceStore schema:
        agent_id, model_name, user_query, response, trace_json,
        status, tokens_used, latency_ms.

        Args:
            openclaw_trace: Single trace step from OpenClaw.
            session: Parent session for metadata.

        Returns:
            Dict ready to pass to TraceStore.save_trace().
        """
        step_type = openclaw_trace.get("type", "unknown")
        content = openclaw_trace.get("content", "")
        result = openclaw_trace.get("result", "")

        user_query = content if step_type == "llm_response" else ""
        response = result if result else content

        return {
            "agent_id": f"openclaw:{session.agent_name}",
            "model_name": session.model,
            "model_version": "",
            "user_query": user_query or f"[{step_type}]",
            "response": str(response),
            "trace_json": [openclaw_trace],
            "status": "success" if step_type != "error" else "error",
            "tokens_used": openclaw_trace.get("tokens", 0),
            "cost": 0.0,
            "latency_ms": openclaw_trace.get("latency_ms", 0),
        }
