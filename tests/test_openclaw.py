"""Tests for OpenClaw adapter and NemoClaw sandbox deployment."""

from unittest.mock import MagicMock, patch

from pulsar_ai.openclaw.adapter import OpenClawAdapter, OpenClawSession
from pulsar_ai.openclaw.nemoclaw import NemoClawManager, SandboxPolicy

# ── OpenClaw Adapter Tests ────────────────────────────────────────


class TestOpenClawAdapter:
    """Tests for OpenClawAdapter with mocked HTTP."""

    def setup_method(self) -> None:
        self.adapter = OpenClawAdapter(
            base_url="http://localhost:8100",
            api_key="test-key",
        )

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_create_session(self, mock_request: MagicMock) -> None:
        """Test session creation with mocked API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "abc123",
            "status": "created",
        }
        mock_request.return_value = mock_resp

        session = self.adapter.create_session(
            {
                "name": "test-agent",
                "model": "llama-3",
                "tools": ["search"],
            }
        )

        assert session.session_id == "abc123"
        assert session.agent_name == "test-agent"
        assert session.model == "llama-3"
        assert session.status == "created"
        assert "search" in session.tools

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_run_session(self, mock_request: MagicMock) -> None:
        """Test running an agent in a session."""
        # First create a session
        mock_resp_create = MagicMock()
        mock_resp_create.status_code = 200
        mock_resp_create.json.return_value = {
            "session_id": "sess1",
            "status": "created",
        }

        mock_resp_run = MagicMock()
        mock_resp_run.status_code = 200
        mock_resp_run.json.return_value = {
            "response": "Hello! How can I help?",
            "trace": [{"type": "llm_response"}],
            "tools_used": [],
            "tokens": 42,
            "latency_ms": 150,
        }

        mock_request.side_effect = [mock_resp_create, mock_resp_run]

        session = self.adapter.create_session({"name": "test"})
        result = self.adapter.run(session.session_id, "Hello")

        assert result["response"] == "Hello! How can I help?"
        assert result["tokens"] == 42
        assert result["latency_ms"] == 150

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_list_sessions(self, mock_request: MagicMock) -> None:
        """Test listing sessions from API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "sessions": [
                {
                    "session_id": "s1",
                    "agent_name": "agent-a",
                    "model": "llama",
                    "status": "running",
                    "created_at": "2026-01-01T00:00:00Z",
                    "tools": [],
                    "metadata": {},
                },
                {
                    "session_id": "s2",
                    "agent_name": "agent-b",
                    "model": "mistral",
                    "status": "completed",
                    "created_at": "2026-01-02T00:00:00Z",
                    "tools": [],
                    "metadata": {},
                },
            ]
        }
        mock_request.return_value = mock_resp

        sessions = self.adapter.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[1].agent_name == "agent-b"

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_stop_session(self, mock_request: MagicMock) -> None:
        """Test stopping a session."""
        mock_resp_create = MagicMock()
        mock_resp_create.status_code = 200
        mock_resp_create.json.return_value = {
            "session_id": "stop-me",
            "status": "created",
        }

        mock_resp_delete = MagicMock()
        mock_resp_delete.status_code = 204

        mock_request.side_effect = [mock_resp_create, mock_resp_delete]

        session = self.adapter.create_session({"name": "test"})
        success = self.adapter.stop_session(session.session_id)
        assert success is True

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_get_trace(self, mock_request: MagicMock) -> None:
        """Test retrieving execution trace."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "trace": [
                {"type": "tool_call", "tool": "search"},
                {"type": "observation", "result": "found 3 items"},
            ]
        }
        mock_request.return_value = mock_resp

        trace = self.adapter.get_trace("sess1")
        assert len(trace) == 2
        assert trace[0]["type"] == "tool_call"

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_health_check(self, mock_request: MagicMock) -> None:
        """Test health check returns healthy status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"version": "1.0.0"}
        mock_request.return_value = mock_resp

        result = self.adapter.health_check()
        assert result["status"] == "healthy"
        assert result["version"] == "1.0.0"

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_connection_error_handled(self, mock_request: MagicMock) -> None:
        """Test graceful handling of connection errors."""
        import requests as req_lib

        mock_request.side_effect = req_lib.ConnectionError("refused")

        result = self.adapter.health_check()
        assert result["status"] == "unavailable"
        assert "Cannot connect" in result["error"]

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_connection_error_on_run(self, mock_request: MagicMock) -> None:
        """Test that run handles connection errors gracefully."""
        import requests as req_lib

        # Create session locally (connection error on create falls back)
        mock_request.side_effect = req_lib.ConnectionError("refused")
        session = self.adapter.create_session({"name": "test"})

        # Run also fails gracefully
        result = self.adapter.run(session.session_id, "hello")
        assert result["error"] == "OpenClaw runtime unavailable"
        assert result["response"] == ""


# ── NemoClaw Tests ────────────────────────────────────────────────


class TestSandboxPolicy:
    """Tests for SandboxPolicy dataclass."""

    def test_sandbox_policy_creation(self) -> None:
        """Test creating a sandbox policy with defaults."""
        policy = SandboxPolicy()
        assert policy.allow_network is False
        assert policy.max_memory_mb == 512
        assert policy.max_tokens == 4096

    def test_sandbox_policy_from_dict(self) -> None:
        """Test creating policy from dict."""
        policy = SandboxPolicy.from_dict(
            {
                "allow_network": True,
                "allowed_domains": ["api.example.com"],
                "max_memory_mb": 1024,
            }
        )
        assert policy.allow_network is True
        assert policy.allowed_domains == ["api.example.com"]
        assert policy.max_memory_mb == 1024

    def test_sandbox_policy_to_dict(self) -> None:
        """Test serializing policy to dict."""
        policy = SandboxPolicy(allow_network=True, max_tokens=8192)
        data = policy.to_dict()
        assert data["allow_network"] is True
        assert data["max_tokens"] == 8192


class TestNemoClawManager:
    """Tests for NemoClawManager with mocked OpenClaw."""

    def setup_method(self) -> None:
        self.mock_adapter = MagicMock(spec=OpenClawAdapter)
        self.manager = NemoClawManager(self.mock_adapter)

    def test_nemoclaw_deploy(self) -> None:
        """Test deploying an agent with sandbox policy."""
        mock_session = OpenClawSession(
            session_id="sess-abc",
            agent_name="test-agent",
            model="llama-3",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session

        policy = SandboxPolicy(allow_network=True, max_tokens=2048)
        deployment = self.manager.deploy(
            {"name": "test-agent", "model": "llama-3"},
            policy,
        )

        assert deployment.session_id == "sess-abc"
        assert deployment.status == "running"
        assert deployment.policy.allow_network is True
        assert deployment.policy.max_tokens == 2048
        self.mock_adapter.create_session.assert_called_once()

    def test_nemoclaw_run_sandboxed(self) -> None:
        """Test running in sandboxed environment."""
        mock_session = OpenClawSession(
            session_id="sess-run",
            agent_name="test",
            model="llama",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session
        self.mock_adapter.run.return_value = {
            "response": "sandboxed result",
            "trace": [],
            "tools_used": [],
            "tokens": 100,
            "latency_ms": 50,
        }

        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)
        result = self.manager.run_sandboxed(deployment.deployment_id, "hello")

        assert result["response"] == "sandboxed result"
        assert result["deployment_id"] == deployment.deployment_id
        assert "policy" in result

    def test_nemoclaw_run_not_found(self) -> None:
        """Test running with invalid deployment ID."""
        result = self.manager.run_sandboxed("nonexistent", "hello")
        assert "error" in result

    def test_nemoclaw_policy_update(self) -> None:
        """Test updating sandbox policy for a deployment."""
        mock_session = OpenClawSession(
            session_id="sess-upd",
            agent_name="test",
            model="llama",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session

        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)

        new_policy = SandboxPolicy(allow_network=True, max_tokens=8192)
        success = self.manager.update_policy(deployment.deployment_id, new_policy)
        assert success is True

        updated = self.manager.get_deployment(deployment.deployment_id)
        assert updated is not None
        assert updated.policy.allow_network is True
        assert updated.policy.max_tokens == 8192

    def test_nemoclaw_stop(self) -> None:
        """Test stopping a deployment."""
        mock_session = OpenClawSession(
            session_id="sess-stop",
            agent_name="test",
            model="llama",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session
        self.mock_adapter.stop_session.return_value = True

        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)
        success = self.manager.stop_deployment(deployment.deployment_id)
        assert success is True

    def test_nemoclaw_list_deployments(self) -> None:
        """Test listing deployments."""
        mock_session = OpenClawSession(
            session_id="sess-list",
            agent_name="test",
            model="llama",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session

        policy = SandboxPolicy()
        self.manager.deploy({"name": "agent1"}, policy)
        self.manager.deploy({"name": "agent2"}, policy)

        deployments = self.manager.list_deployments()
        assert len(deployments) == 2

    def test_nemoclaw_health(self) -> None:
        """Test getting deployment health."""
        mock_session = OpenClawSession(
            session_id="sess-health",
            agent_name="test",
            model="llama",
            status="created",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.create_session.return_value = mock_session
        self.mock_adapter.health_check.return_value = {"status": "healthy"}

        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)
        health = self.manager.get_health(deployment.deployment_id)

        assert health["deployment_status"] == "running"
        assert health["openclaw"]["status"] == "healthy"


# ── API Route Tests ───────────────────────────────────────────────


class TestOpenClawAPI:
    """Tests for OpenClaw API endpoints with FastAPI TestClient."""

    def setup_method(self) -> None:
        # Reset module-level state before each test
        from pulsar_ai.ui.routes import openclaw as routes_mod

        routes_mod._adapter = None
        routes_mod._manager = None

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_api_health(self, mock_request: MagicMock) -> None:
        """Test GET /openclaw/health endpoint."""
        from fastapi.testclient import TestClient

        from pulsar_ai.ui.routes.openclaw import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"version": "1.0.0"}
        mock_request.return_value = mock_resp

        response = client.get("/openclaw/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_api_create_session(self, mock_request: MagicMock) -> None:
        """Test POST /openclaw/sessions endpoint."""
        from fastapi.testclient import TestClient

        from pulsar_ai.ui.routes.openclaw import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "api-sess-1",
            "status": "created",
        }
        mock_request.return_value = mock_resp

        response = client.post(
            "/openclaw/sessions",
            json={"name": "test-agent", "model": "llama-3"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "api-sess-1"
        assert data["agent_name"] == "test-agent"

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_api_list_sessions(self, mock_request: MagicMock) -> None:
        """Test GET /openclaw/sessions endpoint."""
        from fastapi.testclient import TestClient

        from pulsar_ai.ui.routes.openclaw import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"sessions": []}
        mock_request.return_value = mock_resp

        response = client.get("/openclaw/sessions")
        assert response.status_code == 200
        assert "sessions" in response.json()

    @patch("pulsar_ai.openclaw.adapter.requests.request")
    def test_api_create_deployment(self, mock_request: MagicMock) -> None:
        """Test POST /openclaw/deployments endpoint."""
        from fastapi.testclient import TestClient

        from pulsar_ai.ui.routes.openclaw import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "session_id": "dep-sess-1",
            "status": "created",
        }
        mock_request.return_value = mock_resp

        response = client.post(
            "/openclaw/deployments",
            json={
                "name": "sandbox-agent",
                "model": "llama-3",
                "policy": {
                    "allow_network": True,
                    "max_tokens": 2048,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "deployment_id" in data
        assert data["status"] == "running"
        assert data["policy"]["allow_network"] is True
