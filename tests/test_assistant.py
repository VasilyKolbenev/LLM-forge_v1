"""Tests for Forge Co-pilot assistant: tools, command parser, and API."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from llm_forge.ui.assistant import (
    _get_forge_tools,
    parse_command,
    HELP_TEXT,
)
from llm_forge.ui.app import create_app


# ──────────────────────────────────────────────────────────
# Forge Tools
# ──────────────────────────────────────────────────────────

class TestForgeTools:
    """Test that forge tools are properly registered."""

    def test_registry_has_all_tools(self):
        """Test all 10 forge tools are registered."""
        tools = _get_forge_tools()
        expected = {
            "list_experiments", "get_experiment", "start_training",
            "check_training", "cancel_training", "list_datasets",
            "preview_dataset", "recommend_params", "get_hardware",
            "run_evaluation",
        }
        assert set(tools.list_tools()) == expected

    def test_list_experiments_empty(self):
        """Test list_experiments with no experiments."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant._store") as mock_store:
            mock_store.list_all.return_value = []
            result = tools.get("list_experiments").execute()
        assert "No experiments found" in result

    def test_list_experiments_with_data(self):
        """Test list_experiments returns formatted list."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant._store") as mock_store:
            mock_store.list_all.return_value = [
                {"id": "abc", "name": "test-exp", "status": "completed", "final_loss": 0.5},
            ]
            result = tools.get("list_experiments").execute()
        assert "abc" in result
        assert "test-exp" in result
        assert "completed" in result

    def test_get_experiment_found(self):
        """Test get_experiment returns experiment details."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant._store") as mock_store:
            mock_store.get.return_value = {
                "id": "abc", "name": "test", "status": "completed",
                "task": "sft", "model": "qwen", "final_loss": 0.3,
                "created_at": "2024-01-01", "artifacts": {},
            }
            result = tools.get("get_experiment").execute(experiment_id="abc")
        parsed = json.loads(result)
        assert parsed["id"] == "abc"
        assert parsed["final_loss"] == 0.3

    def test_get_experiment_not_found(self):
        """Test get_experiment returns error for missing experiment."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant._store") as mock_store:
            mock_store.get.return_value = None
            result = tools.get("get_experiment").execute(experiment_id="missing")
        assert "not found" in result

    def test_check_training_no_jobs(self):
        """Test check_training with no jobs."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant.list_jobs") as mock_list:
            mock_list.return_value = []
            result = tools.get("check_training").execute()
        assert "No training jobs" in result

    def test_check_training_with_jobs(self):
        """Test check_training shows job info."""
        tools = _get_forge_tools()
        with patch("llm_forge.ui.assistant.list_jobs") as mock_list:
            mock_list.return_value = [
                {"job_id": "j1", "status": "running", "experiment_id": "e1"},
            ]
            result = tools.get("check_training").execute()
        assert "j1" in result
        assert "running" in result

    def test_recommend_params_default(self):
        """Test recommend_params returns recommendations."""
        tools = _get_forge_tools()
        result = tools.get("recommend_params").execute(
            model="Qwen/Qwen2.5-3B-Instruct"
        )
        assert "Learning rate" in result
        assert "Batch size" in result

    def test_recommend_params_small_model(self):
        """Test recommend_params for small model."""
        tools = _get_forge_tools()
        result = tools.get("recommend_params").execute(
            model="llama-1B", dataset_rows=50
        )
        assert "Epochs: 10" in result  # Small dataset → more epochs

    def test_get_hardware(self):
        """Test get_hardware returns info."""
        tools = _get_forge_tools()
        result = tools.get("get_hardware").execute()
        # Should return something (CPU or GPU info)
        assert "VRAM" in result or "CPU" in result or "detection failed" in result

    def test_start_training_no_dataset(self):
        """Test start_training without dataset returns error."""
        tools = _get_forge_tools()
        result = tools.get("start_training").execute(name="test")
        assert "dataset_path is required" in result


# ──────────────────────────────────────────────────────────
# Command Parser
# ──────────────────────────────────────────────────────────

class TestCommandParser:
    """Test slash command parsing."""

    def test_help_command(self):
        """Test /help returns help text."""
        result = parse_command("/help")
        assert result is not None
        assert HELP_TEXT in result["results"][0]

    def test_status_command(self):
        """Test /status calls check_training + list_experiments."""
        with patch("llm_forge.ui.assistant._store") as mock_store, \
             patch("llm_forge.ui.assistant.list_jobs") as mock_jobs:
            mock_store.list_all.return_value = []
            mock_jobs.return_value = []
            result = parse_command("/status")
        assert result is not None
        assert len(result["results"]) == 2

    def test_datasets_command(self):
        """Test /datasets calls list_datasets."""
        result = parse_command("/datasets")
        assert result is not None
        assert len(result["results"]) == 1

    def test_recommend_command(self):
        """Test /recommend with model parameter."""
        result = parse_command("/recommend model=qwen")
        assert result is not None
        assert "Learning rate" in result["results"][0] or "Recommended" in result["results"][0]

    def test_hardware_command(self):
        """Test /hardware command."""
        result = parse_command("/hardware")
        assert result is not None

    def test_experiments_command(self):
        """Test /experiments command."""
        with patch("llm_forge.ui.assistant._store") as mock_store:
            mock_store.list_all.return_value = []
            result = parse_command("/experiments")
        assert result is not None

    def test_train_without_dataset(self):
        """Test /train without dataset returns error."""
        result = parse_command("/train name=test model=qwen")
        assert result is not None
        assert "dataset is required" in result["results"][0]

    def test_unknown_command(self):
        """Test unknown command returns help."""
        result = parse_command("/foobar")
        assert result is not None
        assert "Unknown command" in result["results"][0]

    def test_non_command_returns_none(self):
        """Test non-slash message returns None."""
        result = parse_command("hello how are you")
        assert result is None

    def test_cancel_command(self):
        """Test /cancel command."""
        with patch("llm_forge.ui.assistant.cancel_job") as mock_cancel:
            mock_cancel.return_value = True
            result = parse_command("/cancel job_id=j1")
        assert result is not None
        assert "cancelled" in result["results"][0]


# ──────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create test client for assistant endpoints."""
    with patch("llm_forge.ui.routes.training._store"), \
         patch("llm_forge.ui.routes.experiments._store"), \
         patch("llm_forge.ui.routes.evaluation._store"), \
         patch("llm_forge.ui.routes.export_routes._store"):
        app = create_app()
        yield TestClient(app)


class TestAssistantAPI:
    """Test assistant API endpoints."""

    def test_chat_help_command(self, client):
        """Test POST /api/v1/assistant/chat with /help."""
        resp = client.post("/api/v1/assistant/chat", json={"message": "/help"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "command"
        assert "Available commands" in data["answer"]

    def test_chat_empty_message(self, client):
        """Test POST /api/v1/assistant/chat with empty message."""
        resp = client.post("/api/v1/assistant/chat", json={"message": ""})
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    @patch("llm_forge.ui.assistant._check_llm_available")
    def test_chat_no_llm_no_command(self, mock_llm, client):
        """Test chat without LLM and without command shows help."""
        mock_llm.return_value = False
        resp = client.post("/api/v1/assistant/chat", json={
            "message": "how do I train a model?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "command"
        assert "slash commands" in data["answer"].lower() or "/help" in data["answer"]

    def test_status_endpoint(self, client):
        """Test GET /api/v1/assistant/status."""
        with patch("llm_forge.ui.assistant.list_jobs") as mock_jobs, \
             patch("llm_forge.ui.assistant._store") as mock_store, \
             patch("llm_forge.ui.assistant._check_llm_available") as mock_llm:
            mock_jobs.return_value = []
            mock_store.list_all.return_value = []
            mock_llm.return_value = False

            resp = client.get("/api/v1/assistant/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_jobs" in data
        assert "llm_available" in data

    def test_chat_preserves_session(self, client):
        """Test that session_id is returned and can be reused."""
        resp1 = client.post("/api/v1/assistant/chat", json={"message": "/help"})
        sid = resp1.json()["session_id"]

        resp2 = client.post("/api/v1/assistant/chat", json={
            "message": "/help",
            "session_id": sid,
        })
        assert resp2.json()["session_id"] == sid

    def test_delete_session(self, client):
        """Test DELETE /api/v1/assistant/session/{id}."""
        resp = client.delete("/api/v1/assistant/session/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_chat_with_context(self, client):
        """Test chat passes context through."""
        resp = client.post("/api/v1/assistant/chat", json={
            "message": "/status",
            "context": {"page": "/new", "active_jobs": []},
        })
        assert resp.status_code == 200
