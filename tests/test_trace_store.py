"""Tests for TraceStore — trace persistence and feedback."""

from pathlib import Path

import pytest

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


def _sample_trace() -> list[dict]:
    """Return a minimal agent trace with a tool call and answer."""
    return [
        {
            "type": "tool_call",
            "tool": "search",
            "arguments": {"q": "hello"},
        },
        {
            "type": "observation",
            "result": "world",
        },
        {
            "type": "answer",
            "content": "The answer is world.",
        },
    ]


# ── save / get ────────────────────────────────────────────────────


def test_save_and_get_trace(store: TraceStore) -> None:
    """Save a trace and retrieve it; all fields should round-trip."""
    tid = store.save_trace(
        {
            "user_query": "What is 2+2?",
            "response": "4",
            "model_name": "gpt-4",
            "model_version": "0613",
            "agent_id": "a1",
            "trace_json": _sample_trace(),
            "status": "success",
            "tokens_used": 100,
            "cost": 0.002,
            "latency_ms": 350,
        }
    )
    assert len(tid) == 12

    row = store.get_trace(tid)
    assert row is not None
    assert row["user_query"] == "What is 2+2?"
    assert row["response"] == "4"
    assert row["model_name"] == "gpt-4"
    assert row["model_version"] == "0613"
    assert row["agent_id"] == "a1"
    assert row["status"] == "success"
    assert row["tokens_used"] == 100
    assert row["cost"] == pytest.approx(0.002)
    assert row["latency_ms"] == 350
    assert isinstance(row["trace_json"], list)
    assert len(row["trace_json"]) == 3


def test_save_trace_minimal(store: TraceStore) -> None:
    """Only user_query is required; defaults fill the rest."""
    tid = store.save_trace({"user_query": "hi"})
    row = store.get_trace(tid)
    assert row is not None
    assert row["user_query"] == "hi"
    assert row["model_name"] == ""
    assert row["status"] == "success"
    assert row["tokens_used"] == 0


def test_get_trace_missing(store: TraceStore) -> None:
    """Non-existent trace returns None."""
    assert store.get_trace("nonexistent") is None


# ── list_traces ───────────────────────────────────────────────────


def test_list_traces_empty(store: TraceStore) -> None:
    """Empty database returns an empty list."""
    assert store.list_traces() == []


def test_list_traces_with_filters(store: TraceStore) -> None:
    """Filters by model_name and status work."""
    store.save_trace(
        {
            "user_query": "q1",
            "model_name": "gpt-4",
            "status": "success",
        }
    )
    store.save_trace(
        {
            "user_query": "q2",
            "model_name": "llama",
            "status": "error",
        }
    )

    gpt_only = store.list_traces(model_name="gpt-4")
    assert len(gpt_only) == 1
    assert gpt_only[0]["model_name"] == "gpt-4"

    errors = store.list_traces(status="error")
    assert len(errors) == 1
    assert errors[0]["status"] == "error"


def test_list_traces_pagination(store: TraceStore) -> None:
    """Limit and offset paginate correctly."""
    for i in range(5):
        store.save_trace({"user_query": f"q{i}"})

    page1 = store.list_traces(limit=2, offset=0)
    page2 = store.list_traces(limit=2, offset=2)
    page3 = store.list_traces(limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1


# ── feedback ──────────────────────────────────────────────────────


def test_add_and_get_feedback(store: TraceStore) -> None:
    """Save feedback and retrieve it for a trace."""
    tid = store.save_trace({"user_query": "test"})
    fid = store.add_feedback(tid, "thumbs", 1.0, reason="good", user_id="u1")
    assert len(fid) == 12

    fb = store.get_feedback(tid)
    assert len(fb) == 1
    assert fb[0]["rating"] == pytest.approx(1.0)
    assert fb[0]["reason"] == "good"
    assert fb[0]["user_id"] == "u1"


def test_feedback_rating_filter(store: TraceStore) -> None:
    """list_traces with min_rating filters correctly."""
    t1 = store.save_trace({"user_query": "good"})
    t2 = store.save_trace({"user_query": "bad"})

    store.add_feedback(t1, "rating", 0.9)
    store.add_feedback(t2, "rating", 0.2)

    high = store.list_traces(min_rating=0.5)
    assert len(high) == 1
    assert high[0]["trace_id"] == t1


def test_list_traces_has_feedback_filter(
    store: TraceStore,
) -> None:
    """has_feedback=True/False filter works."""
    t1 = store.save_trace({"user_query": "with fb"})
    store.save_trace({"user_query": "without fb"})
    store.add_feedback(t1, "thumbs", 1.0)

    with_fb = store.list_traces(has_feedback=True)
    assert len(with_fb) == 1
    assert with_fb[0]["trace_id"] == t1

    without_fb = store.list_traces(has_feedback=False)
    assert len(without_fb) == 1
    assert without_fb[0]["user_query"] == "without fb"


# ── export SFT / DPO ─────────────────────────────────────────────


def test_export_as_sft(store: TraceStore) -> None:
    """Exported SFT example has messages in chat format."""
    tid = store.save_trace(
        {
            "user_query": "What is AI?",
            "trace_json": _sample_trace(),
        }
    )
    results = store.export_as_sft([tid])
    assert len(results) == 1
    msgs = results[0]["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is AI?"
    assert msgs[-1]["role"] == "assistant"


def test_export_as_dpo(store: TraceStore) -> None:
    """Exported DPO pair has prompt/chosen/rejected."""
    good_id = store.save_trace(
        {
            "user_query": "Explain X",
            "trace_json": [
                {"type": "answer", "content": "Good answer"},
            ],
        }
    )
    bad_id = store.save_trace(
        {
            "user_query": "Explain X",
            "trace_json": [
                {"type": "answer", "content": "Bad answer"},
            ],
        }
    )
    results = store.export_as_dpo([good_id], [bad_id])
    assert len(results) == 1
    assert "chosen" in results[0]
    assert "rejected" in results[0]


# ── auto_pair_dpo ─────────────────────────────────────────────────


def test_auto_pair_dpo(store: TraceStore) -> None:
    """Groups by query and pairs good vs bad by rating."""
    t_good = store.save_trace(
        {
            "user_query": "What is AI?",
            "trace_json": [
                {"type": "answer", "content": "AI is ..."},
            ],
        }
    )
    t_bad = store.save_trace(
        {
            "user_query": "what is ai?",  # same normalised
            "trace_json": [
                {"type": "answer", "content": "dunno"},
            ],
        }
    )

    store.add_feedback(t_good, "rating", 0.9)
    store.add_feedback(t_bad, "rating", 0.1)

    pairs = store.auto_pair_dpo([t_good, t_bad])
    assert len(pairs) == 1
    assert pairs[0]["chosen"] != pairs[0]["rejected"]


def test_auto_pair_dpo_no_pairs(store: TraceStore) -> None:
    """All traces with same polarity produce no pairs."""
    t1 = store.save_trace(
        {
            "user_query": "q",
            "trace_json": [
                {"type": "answer", "content": "a1"},
            ],
        }
    )
    t2 = store.save_trace(
        {
            "user_query": "q",
            "trace_json": [
                {"type": "answer", "content": "a2"},
            ],
        }
    )

    store.add_feedback(t1, "rating", 0.9)
    store.add_feedback(t2, "rating", 0.8)

    pairs = store.auto_pair_dpo([t1, t2])
    assert pairs == []


# ── stats ─────────────────────────────────────────────────────────


def test_get_stats(store: TraceStore) -> None:
    """Stats return counts and averages."""
    t1 = store.save_trace(
        {
            "user_query": "q1",
            "status": "success",
        }
    )
    store.save_trace(
        {
            "user_query": "q2",
            "status": "error",
        }
    )
    store.add_feedback(t1, "rating", 0.8)

    stats = store.get_stats(days=30)
    assert stats["total"] == 2
    assert stats["with_feedback"] == 1
    assert stats["avg_rating"] == pytest.approx(0.8)
    assert stats["status_counts"]["success"] == 1
    assert stats["status_counts"]["error"] == 1
