"""Tests for ExperimentStore CRUD operations."""

import json
import pytest
from pathlib import Path

from llm_forge.ui.experiment_store import ExperimentStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh ExperimentStore with temp file."""
    return ExperimentStore(store_path=tmp_path / "experiments.json")


def test_create_experiment(store):
    """Test creating a new experiment."""
    exp_id = store.create("test-exp", {"model": {"name": "test"}}, task="sft")
    assert len(exp_id) == 8
    exp = store.get(exp_id)
    assert exp is not None
    assert exp["name"] == "test-exp"
    assert exp["status"] == "queued"
    assert exp["task"] == "sft"


def test_update_status(store):
    """Test updating experiment status."""
    exp_id = store.create("test", {})
    store.update_status(exp_id, "running")
    assert store.get(exp_id)["status"] == "running"

    store.update_status(exp_id, "completed")
    exp = store.get(exp_id)
    assert exp["status"] == "completed"
    assert exp["completed_at"] is not None


def test_add_metrics(store):
    """Test appending training metrics."""
    exp_id = store.create("test", {})
    store.add_metrics(exp_id, {"step": 10, "loss": 0.5})
    store.add_metrics(exp_id, {"step": 20, "loss": 0.3})

    exp = store.get(exp_id)
    assert len(exp["training_history"]) == 2
    assert exp["final_loss"] == 0.3


def test_set_artifacts(store):
    """Test storing artifact paths."""
    exp_id = store.create("test", {})
    store.set_artifacts(exp_id, {"adapter_dir": "/path/to/lora"})
    assert store.get(exp_id)["artifacts"]["adapter_dir"] == "/path/to/lora"


def test_set_eval_results(store):
    """Test storing eval results."""
    exp_id = store.create("test", {})
    store.set_eval_results(exp_id, {"accuracy": 0.95})
    assert store.get(exp_id)["eval_results"]["accuracy"] == 0.95


def test_list_all(store):
    """Test listing all experiments."""
    store.create("exp1", {})
    store.create("exp2", {})
    store.create("exp3", {})

    all_exps = store.list_all()
    assert len(all_exps) == 3


def test_list_all_filtered(store):
    """Test filtering experiments by status."""
    id1 = store.create("exp1", {})
    id2 = store.create("exp2", {})
    store.update_status(id1, "completed")

    completed = store.list_all(status="completed")
    assert len(completed) == 1
    assert completed[0]["id"] == id1


def test_delete_experiment(store):
    """Test deleting an experiment."""
    exp_id = store.create("test", {})
    assert store.delete(exp_id) is True
    assert store.get(exp_id) is None
    assert store.delete(exp_id) is False


def test_get_nonexistent(store):
    """Test getting a nonexistent experiment returns None."""
    assert store.get("nonexistent") is None


def test_list_empty(store):
    """Test listing returns empty list when no experiments."""
    assert store.list_all() == []


def test_persistence(tmp_path):
    """Test that data persists across store instances."""
    path = tmp_path / "experiments.json"
    store1 = ExperimentStore(store_path=path)
    exp_id = store1.create("persistent", {"key": "value"})

    store2 = ExperimentStore(store_path=path)
    exp = store2.get(exp_id)
    assert exp is not None
    assert exp["name"] == "persistent"
