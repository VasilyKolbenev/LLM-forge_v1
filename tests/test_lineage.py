"""Tests for LineageStore and /api/v1/lineage API routes."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pulsar_ai.storage.database import Database, reset_database
from pulsar_ai.storage.lineage_store import LineageStore
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
def store(db: Database) -> LineageStore:
    """Create a LineageStore backed by a temp SQLite DB."""
    return LineageStore(db=db)


@pytest.fixture
def trace_store(db: Database) -> TraceStore:
    """Create a TraceStore backed by the same temp DB."""
    return TraceStore(db=db)


def _insert_experiment(
    db: Database,
    exp_id: str = "exp1",
    name: str = "SFT Run",
    task: str = "sft",
    status: str = "completed",
    model: str = "llama-7b",
    dataset_id: str = "ds-001",
    config: dict | None = None,
    final_loss: float | None = 0.42,
) -> None:
    """Insert a test experiment into the database."""
    cfg = config or {
        "dataset_path": "/data/train.jsonl",
        "output_dir": "/models/sft-llama",
    }
    now = datetime.utcnow().isoformat()
    db.execute(
        """
        INSERT INTO experiments
            (id, name, task, status, model, dataset_id, config,
             created_at, last_update_at, completed_at, final_loss)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            exp_id,
            name,
            task,
            status,
            model,
            dataset_id,
            json.dumps(cfg),
            now,
            now,
            now if status in ("completed", "failed") else None,
            final_loss,
        ),
    )
    db.commit()


# ── LineageStore: get_lineage_graph ───────────────────────────────


class TestLineageGraph:
    """Tests for LineageStore.get_lineage_graph."""

    def test_empty_graph(self, store: LineageStore) -> None:
        """No data returns empty nodes and edges."""
        graph = store.get_lineage_graph()
        assert graph == {"nodes": [], "edges": []}

    def test_graph_with_experiment(self, db: Database, store: LineageStore) -> None:
        """A single experiment produces dataset, experiment, and model nodes."""
        _insert_experiment(db)
        graph = store.get_lineage_graph()

        node_types = {n["type"] for n in graph["nodes"]}
        assert "experiment" in node_types
        assert "dataset" in node_types
        assert "model" in node_types

        # Should have at least 3 nodes: dataset, experiment, model
        assert len(graph["nodes"]) >= 3

    def test_graph_with_traces(
        self,
        db: Database,
        store: LineageStore,
        trace_store: TraceStore,
    ) -> None:
        """Experiment + traces produce trace and feedback nodes when feedback exists."""
        _insert_experiment(
            db,
            config={
                "dataset_path": "/data/train.jsonl",
                "output_dir": "/models/sft-llama",
            },
        )

        # Add traces matching the model output dir
        tid = trace_store.save_trace(
            {
                "user_query": "Hello?",
                "model_name": "sft-llama",
                "response": "Hi!",
                "latency_ms": 120,
            }
        )

        # Add feedback
        trace_store.add_feedback(tid, "thumbs", 1.0)

        graph = store.get_lineage_graph()

        node_types = {n["type"] for n in graph["nodes"]}
        assert "traces" in node_types
        assert "feedback" in node_types

    def test_graph_edges(self, db: Database, store: LineageStore) -> None:
        """Edges connect dataset→experiment and experiment→model."""
        _insert_experiment(db)
        graph = store.get_lineage_graph()

        edge_labels = [e["label"] for e in graph["edges"]]
        assert "trains" in edge_labels
        assert "produces" in edge_labels

    def test_graph_sft_dpo_chain(self, db: Database, store: LineageStore) -> None:
        """DPO experiment referencing SFT output creates sft_base edge."""
        _insert_experiment(
            db,
            exp_id="sft1",
            name="SFT Base",
            task="sft",
            config={
                "dataset_path": "/data/sft.jsonl",
                "output_dir": "/models/sft-out",
            },
        )
        _insert_experiment(
            db,
            exp_id="dpo1",
            name="DPO Refine",
            task="dpo",
            config={
                "dataset_path": "/data/dpo.jsonl",
                "output_dir": "/models/dpo-out",
                "sft_adapter_path": "/models/sft-out",
            },
        )
        graph = store.get_lineage_graph()

        edge_labels = [e["label"] for e in graph["edges"]]
        assert "sft_base" in edge_labels

    def test_graph_filter_by_model_name(
        self,
        db: Database,
        store: LineageStore,
        trace_store: TraceStore,
    ) -> None:
        """Filtering by model_name scopes the graph."""
        _insert_experiment(
            db,
            exp_id="e1",
            name="Run A",
            model="modelA",
            config={"dataset_path": "/data/a.jsonl", "output_dir": "/models/modelA"},
        )
        _insert_experiment(
            db,
            exp_id="e2",
            name="Run B",
            model="modelB",
            config={"dataset_path": "/data/b.jsonl", "output_dir": "/models/modelB"},
        )

        graph = store.get_lineage_graph(model_name="modelA")
        exp_nodes = [n for n in graph["nodes"] if n["type"] == "experiment"]
        assert len(exp_nodes) == 1
        assert exp_nodes[0]["label"] == "Run A"


# ── LineageStore: get_timeline ────────────────────────────────────


class TestTimeline:
    """Tests for LineageStore.get_timeline."""

    def test_timeline_empty(self, store: LineageStore) -> None:
        """No data returns empty list."""
        events = store.get_timeline()
        assert events == []

    def test_timeline_with_experiments(self, db: Database, store: LineageStore) -> None:
        """Created experiments produce timeline events."""
        _insert_experiment(db, exp_id="e1", name="First Run")
        _insert_experiment(db, exp_id="e2", name="Second Run", status="failed")

        events = store.get_timeline()
        assert len(events) >= 2

        event_types = {e["event_type"] for e in events}
        assert "experiment_created" in event_types

    def test_timeline_with_traces(
        self,
        db: Database,
        store: LineageStore,
        trace_store: TraceStore,
    ) -> None:
        """Traces produce traces_recorded events."""
        trace_store.save_trace(
            {
                "user_query": "test",
                "model_name": "gpt-4",
            }
        )

        events = store.get_timeline()
        trace_events = [e for e in events if e["event_type"] == "traces_recorded"]
        assert len(trace_events) >= 1

    def test_timeline_respects_limit(self, db: Database, store: LineageStore) -> None:
        """Timeline respects the limit parameter."""
        for i in range(5):
            _insert_experiment(db, exp_id=f"e{i}", name=f"Run {i}")

        events = store.get_timeline(limit=3)
        assert len(events) <= 3


# ── API endpoints ─────────────────────────────────────────────────


@pytest.fixture
def client(store: LineageStore):
    """Create a test client with the lineage router wired to a temp store."""
    with patch("pulsar_ai.ui.routes.lineage._store", store):
        from pulsar_ai.ui.app import create_app

        app = create_app()
        yield TestClient(app)


class TestLineageAPI:
    """Tests for /api/v1/lineage/* endpoints."""

    def test_api_graph_endpoint(self, client: TestClient) -> None:
        """GET /lineage/graph returns valid structure."""
        resp = client.get("/api/v1/lineage/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_api_timeline_endpoint(self, client: TestClient) -> None:
        """GET /lineage/timeline returns events list."""
        resp = client.get("/api/v1/lineage/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_api_graph_with_data(
        self,
        db: Database,
        client: TestClient,
    ) -> None:
        """GET /lineage/graph returns nodes when experiments exist."""
        _insert_experiment(db)
        resp = client.get("/api/v1/lineage/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) >= 3

    def test_api_timeline_with_data(
        self,
        db: Database,
        client: TestClient,
    ) -> None:
        """GET /lineage/timeline returns events when experiments exist."""
        _insert_experiment(db)
        resp = client.get("/api/v1/lineage/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) >= 1
