"""Tests for /api/v1/traces API routes."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pulsar_ai.storage.database import Database, reset_database
from pulsar_ai.storage.trace_store import TraceStore


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level DB singleton is reset between tests."""
    reset_database()
    yield
    reset_database()


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Create a fresh Database in a temp directory."""
    return Database(db_path=tmp_path / "test.db")


@pytest.fixture
def store(db: Database) -> TraceStore:
    """Create a TraceStore backed by a temp SQLite DB."""
    return TraceStore(db=db)


@pytest.fixture
def client(store: TraceStore):
    """Create a test client with the traces router wired to a temp store."""
    with patch("pulsar_ai.ui.routes.traces._store", store):
        from pulsar_ai.ui.app import create_app

        app = create_app()
        yield TestClient(app)


def _sample_trace() -> list[dict]:
    """Return a minimal agent trace."""
    return [
        {"type": "tool_call", "tool": "search", "arguments": {"q": "hi"}},
        {"type": "observation", "result": "hello"},
        {"type": "answer", "content": "hello world"},
    ]


# ── list traces ──────────────────────────────────────────────────


class TestListTraces:
    """Tests for GET /api/v1/traces."""

    def test_list_traces_empty(self, client: TestClient) -> None:
        """Empty store returns empty list."""
        resp = client.get("/api/v1/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["traces"] == []
        assert data["total"] == 0

    def test_list_traces_returns_data(self, client: TestClient, store: TraceStore) -> None:
        """Saved traces appear in the list."""
        store.save_trace({"user_query": "q1", "model_name": "gpt-4"})
        store.save_trace({"user_query": "q2", "model_name": "llama"})

        resp = client.get("/api/v1/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["traces"]) == 2

    def test_list_traces_filter_model(self, client: TestClient, store: TraceStore) -> None:
        """Model name filter works."""
        store.save_trace({"user_query": "q1", "model_name": "gpt-4"})
        store.save_trace({"user_query": "q2", "model_name": "llama"})

        resp = client.get("/api/v1/traces?model_name=gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["traces"][0]["model_name"] == "gpt-4"

    def test_list_traces_filter_status(self, client: TestClient, store: TraceStore) -> None:
        """Status filter works."""
        store.save_trace({"user_query": "q1", "status": "success"})
        store.save_trace({"user_query": "q2", "status": "error"})

        resp = client.get("/api/v1/traces?status=error")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["traces"][0]["status"] == "error"


# ── trace detail ─────────────────────────────────────────────────


class TestTraceDetail:
    """Tests for GET /api/v1/traces/{trace_id}."""

    def test_get_trace_detail(self, client: TestClient, store: TraceStore) -> None:
        """Get a trace with its feedback."""
        tid = store.save_trace(
            {
                "user_query": "What is AI?",
                "response": "Artificial Intelligence",
                "trace_json": _sample_trace(),
            }
        )
        store.add_feedback(tid, "thumbs", 1.0, reason="great")

        resp = client.get(f"/api/v1/traces/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_query"] == "What is AI?"
        assert data["response"] == "Artificial Intelligence"
        assert isinstance(data["trace_json"], list)
        assert len(data["feedback"]) == 1
        assert data["feedback"][0]["reason"] == "great"

    def test_get_trace_not_found(self, client: TestClient) -> None:
        """Missing trace returns 404."""
        resp = client.get("/api/v1/traces/nonexistent")
        assert resp.status_code == 404


# ── feedback ─────────────────────────────────────────────────────


class TestAddFeedback:
    """Tests for POST /api/v1/traces/{trace_id}/feedback."""

    def test_add_feedback(self, client: TestClient, store: TraceStore) -> None:
        """Add feedback and verify it was saved."""
        tid = store.save_trace({"user_query": "test"})

        resp = client.post(
            f"/api/v1/traces/{tid}/feedback",
            json={
                "feedback_type": "thumbs",
                "rating": 1.0,
                "reason": "good answer",
            },
        )
        assert resp.status_code == 200
        assert "feedback_id" in resp.json()

        # Verify feedback is persisted
        fb = store.get_feedback(tid)
        assert len(fb) == 1
        assert fb[0]["reason"] == "good answer"

    def test_add_feedback_not_found(self, client: TestClient) -> None:
        """Feedback on missing trace returns 404."""
        resp = client.post(
            "/api/v1/traces/nonexistent/feedback",
            json={"rating": 1.0},
        )
        assert resp.status_code == 404


# ── stats ────────────────────────────────────────────────────────


class TestStats:
    """Tests for GET /api/v1/traces/stats."""

    def test_get_stats(self, client: TestClient, store: TraceStore) -> None:
        """Stats endpoint returns expected structure."""
        t1 = store.save_trace({"user_query": "q1", "status": "success"})
        store.save_trace({"user_query": "q2", "status": "error"})
        store.add_feedback(t1, "rating", 0.9)

        resp = client.get("/api/v1/traces/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["with_feedback"] == 1
        assert data["avg_rating"] == pytest.approx(0.9)
        assert data["status_counts"]["success"] == 1
        assert data["status_counts"]["error"] == 1


# ── build dataset ────────────────────────────────────────────────


class TestBuildDataset:
    """Tests for POST /api/v1/traces/build-dataset."""

    def test_build_sft_dataset(self, client: TestClient, store: TraceStore, tmp_path: Path) -> None:
        """Build SFT dataset from traces."""
        tid = store.save_trace(
            {
                "user_query": "What is AI?",
                "trace_json": _sample_trace(),
            }
        )

        resp = client.post(
            "/api/v1/traces/build-dataset",
            json={
                "trace_ids": [tid],
                "format": "sft",
                "name": "test-sft",
                "output_dir": str(tmp_path / "output"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "sft"
        assert data["num_examples"] >= 1
        assert Path(data["path"]).exists()

    def test_build_dpo_dataset(self, client: TestClient, store: TraceStore, tmp_path: Path) -> None:
        """Build DPO dataset from traces with mixed ratings."""
        t_good = store.save_trace(
            {
                "user_query": "What is AI?",
                "trace_json": [{"type": "answer", "content": "AI is ..."}],
            }
        )
        t_bad = store.save_trace(
            {
                "user_query": "what is ai?",
                "trace_json": [{"type": "answer", "content": "dunno"}],
            }
        )
        store.add_feedback(t_good, "rating", 0.9)
        store.add_feedback(t_bad, "rating", 0.1)

        resp = client.post(
            "/api/v1/traces/build-dataset",
            json={
                "trace_ids": [t_good, t_bad],
                "format": "dpo",
                "name": "test-dpo",
                "output_dir": str(tmp_path / "output"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "dpo"
        assert data["num_examples"] >= 1

    def test_build_dataset_invalid_format(self, client: TestClient, store: TraceStore) -> None:
        """Invalid format returns 400."""
        tid = store.save_trace({"user_query": "q"})
        resp = client.post(
            "/api/v1/traces/build-dataset",
            json={"trace_ids": [tid], "format": "invalid"},
        )
        assert resp.status_code == 400

    def test_build_dataset_empty_result(self, client: TestClient, tmp_path: Path) -> None:
        """No valid traces produces 400."""
        resp = client.post(
            "/api/v1/traces/build-dataset",
            json={
                "trace_ids": ["nonexistent"],
                "format": "sft",
                "output_dir": str(tmp_path / "output"),
            },
        )
        assert resp.status_code == 400
