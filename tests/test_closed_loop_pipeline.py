"""Tests for closed-loop pipeline steps: collect, build, regression eval."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pulsar_ai.storage.database import Database, reset_database
from pulsar_ai.storage.trace_store import TraceStore
from pulsar_ai.pipeline.closed_loop_steps import (
    PipelineError,
    step_collect_traces,
    step_build_dataset,
    step_regression_eval,
    _filter_min_length,
    _deduplicate,
)


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


def _sample_trace(query: str = "What is 2+2?", response: str = "4") -> dict:
    """Build a minimal trace dict."""
    return {
        "user_query": query,
        "response": response,
        "model_name": "test-model",
        "model_version": "v1",
        "status": "success",
        "trace_json": [
            {"type": "answer", "content": response},
        ],
    }


def _seed_traces(store: TraceStore, count: int = 5) -> list[str]:
    """Insert N traces and return their IDs."""
    ids = []
    for i in range(count):
        tid = store.save_trace(_sample_trace(query=f"Question {i}", response=f"Answer {i}"))
        ids.append(tid)
    return ids


# ── TestCollectTracesStep ──────────────────────────────────────────


class TestCollectTracesStep:
    """Tests for the collect_traces pipeline step."""

    def test_collect_traces_returns_ids(self, store: TraceStore) -> None:
        """Collected trace IDs match those in the store."""
        inserted = _seed_traces(store, 3)

        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_collect_traces({"days": 30, "limit": 100})

        assert result["count"] == 3
        assert set(result["trace_ids"]) == set(inserted)

    def test_collect_traces_with_filters(self, store: TraceStore) -> None:
        """Filters narrow down the returned traces."""
        store.save_trace(
            {
                "user_query": "Good question",
                "response": "Great answer",
                "model_name": "alpha",
                "status": "success",
                "trace_json": [],
            }
        )
        store.save_trace(
            {
                "user_query": "Bad question",
                "response": "Err",
                "model_name": "beta",
                "status": "error",
                "trace_json": [],
            }
        )

        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_collect_traces({"days": 30, "status": "error", "limit": 100})

        assert result["count"] == 1

    def test_collect_traces_empty_returns_empty(self, store: TraceStore) -> None:
        """Empty store yields empty result."""
        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_collect_traces({"days": 7})

        assert result["count"] == 0
        assert result["trace_ids"] == []


# ── TestBuildDatasetStep ───────────────────────────────────────────


class TestBuildDatasetStep:
    """Tests for the build_dataset pipeline step."""

    def test_build_sft_dataset_writes_jsonl(self, store: TraceStore, tmp_path: Path) -> None:
        """SFT format produces a valid JSONL file."""
        ids = _seed_traces(store, 3)

        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_build_dataset(
                {
                    "trace_ids": ids,
                    "format": "sft",
                    "output_dir": str(tmp_path / "out"),
                    "name": "test-sft",
                    "quality_filter": {"dedup": True, "min_length": 1},
                }
            )

        assert result["format"] == "sft"
        assert result["num_examples"] >= 0
        assert Path(result["path"]).exists()

        # Verify it is valid JSONL
        with open(result["path"], encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            json.loads(line)

    def test_build_dpo_dataset_auto_pairs(self, store: TraceStore, tmp_path: Path) -> None:
        """DPO format calls auto_pair_dpo and writes JSONL."""
        # Create traces with ratings for DPO pairing
        tid_good = store.save_trace(_sample_trace(query="test query", response="Good answer here"))
        store.add_feedback(tid_good, "rating", 0.9)

        tid_bad = store.save_trace(_sample_trace(query="test query", response="Bad answer here"))
        store.add_feedback(tid_bad, "rating", 0.1)

        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_build_dataset(
                {
                    "trace_ids": [tid_good, tid_bad],
                    "format": "dpo",
                    "output_dir": str(tmp_path / "dpo-out"),
                    "name": "test-dpo",
                    "quality_filter": {"dedup": True, "min_length": 1},
                }
            )

        assert result["format"] == "dpo"
        assert Path(result["path"]).exists()

    def test_quality_filter_dedup(self) -> None:
        """Dedup removes exact-duplicate examples."""
        examples = [
            {"instruction": "hello", "output": "world"},
            {"instruction": "hello", "output": "world"},
            {"instruction": "foo", "output": "bar"},
        ]
        deduped = _deduplicate(examples, "sft")
        assert len(deduped) == 2

    def test_quality_filter_min_length(self) -> None:
        """Min-length filter removes short outputs."""
        examples = [
            {"instruction": "q1", "output": "short"},
            {"instruction": "q2", "output": "this is a sufficiently long response"},
        ]
        filtered = _filter_min_length(examples, 10, "sft")
        assert len(filtered) == 1
        assert filtered[0]["output"] == "this is a sufficiently long response"

    def test_build_dataset_empty_traces(self, store: TraceStore, tmp_path: Path) -> None:
        """Empty trace list produces zero-example dataset."""
        with patch(
            "pulsar_ai.pipeline.closed_loop_steps.TraceStore",
            return_value=store,
        ):
            result = step_build_dataset(
                {
                    "trace_ids": [],
                    "format": "sft",
                    "output_dir": str(tmp_path / "empty-out"),
                    "name": "empty-ds",
                }
            )

        assert result["num_examples"] == 0


# ── TestRegressionEvalStep ─────────────────────────────────────────


class TestRegressionEvalStep:
    """Tests for the regression_eval pipeline step."""

    def _make_eval_dataset(self, tmp_path: Path, n: int = 5) -> str:
        """Write a small eval JSONL and return the path."""
        path = tmp_path / "eval.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                json.dump({"instruction": f"Eval question {i}"}, f)
                f.write("\n")
        return str(path)

    def test_regression_eval_no_regression(self, tmp_path: Path) -> None:
        """When new model matches baseline, no regression is detected."""
        eval_path = self._make_eval_dataset(tmp_path, 5)

        result = step_regression_eval(
            {
                "new_model": "/models/new",
                "baseline_model": "/models/baseline",
                "eval_dataset": eval_path,
                "num_samples": 5,
                "block_on_regression": False,
            }
        )

        # Simulated judge gives both models score 4, so win_rate >= 0.5
        assert result["win_rate"] >= 0.5
        assert result["regression_detected"] is False
        assert "samples" in result["report"]

    def test_regression_eval_detected(self, tmp_path: Path) -> None:
        """Mocked regression is detected when new model scores lower."""
        eval_path = self._make_eval_dataset(tmp_path, 3)

        with patch("pulsar_ai.pipeline.closed_loop_steps._run_judge") as mock_judge:
            # New model gets low scores, baseline gets high scores
            call_count = [0]

            def side_effect(judge, instruction, response):
                call_count[0] += 1
                if "new" in response.lower() or call_count[0] % 2 == 1:
                    return "\n".join(f"{c.name}: 1 | Poor" for c in judge.criteria)
                return "\n".join(f"{c.name}: 5 | Excellent" for c in judge.criteria)

            mock_judge.side_effect = side_effect

            result = step_regression_eval(
                {
                    "new_model": "/models/new",
                    "baseline_model": "/models/baseline",
                    "eval_dataset": eval_path,
                    "num_samples": 3,
                    "block_on_regression": False,
                }
            )

        assert result["regression_detected"] is True
        assert result["win_rate"] < 0.5

    def test_regression_eval_blocks_pipeline(self, tmp_path: Path) -> None:
        """PipelineError is raised when block_on_regression is True."""
        eval_path = self._make_eval_dataset(tmp_path, 3)

        with patch("pulsar_ai.pipeline.closed_loop_steps._run_judge") as mock_judge:
            call_count = [0]

            def side_effect(judge, instruction, response):
                call_count[0] += 1
                if "new" in response.lower() or call_count[0] % 2 == 1:
                    return "\n".join(f"{c.name}: 1 | Poor" for c in judge.criteria)
                return "\n".join(f"{c.name}: 5 | Excellent" for c in judge.criteria)

            mock_judge.side_effect = side_effect

            with pytest.raises(PipelineError, match="Regression detected"):
                step_regression_eval(
                    {
                        "new_model": "/models/new",
                        "baseline_model": "/models/baseline",
                        "eval_dataset": eval_path,
                        "num_samples": 3,
                        "block_on_regression": True,
                    }
                )

    def test_regression_eval_empty_dataset(self, tmp_path: Path) -> None:
        """Non-existent eval dataset returns safe defaults."""
        result = step_regression_eval(
            {
                "new_model": "/models/new",
                "baseline_model": "/models/baseline",
                "eval_dataset": str(tmp_path / "nonexistent.jsonl"),
                "num_samples": 5,
            }
        )

        assert result["win_rate"] == 0.0
        assert result["regression_detected"] is False
