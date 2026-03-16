"""Tests for WorkflowStore CRUD operations."""
import pytest
from pulsar_ai.ui.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(store_path=tmp_path / "workflows.json")


def test_create_and_get_workflow(store):
    result = store.save(name="test-wf", nodes=[{"id": "n1"}], edges=[])
    wf_id = result["id"]
    wf = store.get(wf_id)
    assert wf is not None
    assert wf["name"] == "test-wf"
    assert len(wf["nodes"]) == 1


def test_list_workflows(store):
    store.save(name="wf1", nodes=[], edges=[])
    store.save(name="wf2", nodes=[], edges=[])
    workflows = store.list_all()
    assert len(workflows) == 2


def test_delete_workflow(store):
    result = store.save(name="to-delete", nodes=[], edges=[])
    wf_id = result["id"]
    assert store.delete(wf_id) is True
    assert store.get(wf_id) is None


def test_update_workflow(store):
    result = store.save(name="original", nodes=[], edges=[])
    wf_id = result["id"]
    store.save(name="updated", nodes=[{"id": "n1"}], edges=[], workflow_id=wf_id)
    wf = store.get(wf_id)
    assert wf["name"] == "updated"
    assert len(wf["nodes"]) == 1
