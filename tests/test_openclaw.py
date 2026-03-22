"""Tests for OpenClaw runtime, adapter, NemoClaw sandbox, and trace ingestion."""

from unittest.mock import MagicMock, patch

from pulsar_ai.openclaw.adapter import OpenClawAdapter, OpenClawSession
from pulsar_ai.openclaw.nemoclaw import NemoClawManager, SandboxPolicy
from pulsar_ai.openclaw.runtime import OpenClawRuntime, get_runtime
from pulsar_ai.openclaw.trace_ingestion import OpenClawTraceIngester

# ── OpenClaw Runtime Tests ────────────────────────────────────────


class TestOpenClawRuntime:
    """Tests for the built-in OpenClaw runtime."""

    def setup_method(self) -> None:
        self.runtime = OpenClawRuntime()

    def test_runtime_create_session(self) -> None:
        """Test creating a session returns a valid RuntimeSession."""
        session = self.runtime.create_session(
            {
                "name": "test-agent",
                "model": "llama3",
                "tools": ["search"],
                "system_prompt": "You are helpful.",
            }
        )

        assert session.session_id
        assert len(session.session_id) == 12
        assert session.agent_name == "test-agent"
        assert session.model == "llama3"
        assert session.tools == ["search"]
        assert session.status == "created"
        assert session.system_prompt == "You are helpful."

    def test_runtime_create_session_defaults(self) -> None:
        """Test creating a session with minimal config uses defaults."""
        session = self.runtime.create_session({})

        assert session.agent_name == "default-agent"
        assert session.model == ""
        assert session.tools == []
        assert session.system_prompt == "You are a helpful AI assistant."

    @patch("pulsar_ai.agent.base.BaseAgent.from_config")
    def test_runtime_run_session(self, mock_from_config: MagicMock) -> None:
        """Test running a session delegates to BaseAgent."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Hello! I can help."
        mock_agent.trace = [{"type": "answer", "content": "Hello! I can help."}]
        mock_agent._total_tokens = 42
        mock_from_config.return_value = mock_agent

        session = self.runtime.create_session({"name": "test", "model": "llama3"})
        result = self.runtime.run(session.session_id, "Hi there")

        assert result["response"] == "Hello! I can help."
        assert result["tokens"] == 42
        assert result["session_id"] == session.session_id
        assert result["latency_ms"] >= 0
        assert isinstance(result["trace"], list)
        assert isinstance(result["tools_used"], list)
        mock_from_config.assert_called_once()

    @patch("pulsar_ai.agent.base.BaseAgent.from_config")
    def test_runtime_run_tracks_tools_used(self, mock_from_config: MagicMock) -> None:
        """Test that tools_used is extracted from trace."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Result"
        mock_agent.trace = [
            {"type": "tool_call", "tool": "search"},
            {"type": "observation", "tool": "search", "result": "found"},
            {"type": "tool_call", "tool": "calc"},
            {"type": "answer", "content": "Result"},
        ]
        mock_agent._total_tokens = 100
        mock_from_config.return_value = mock_agent

        session = self.runtime.create_session({"name": "test"})
        result = self.runtime.run(session.session_id, "query")

        assert result["tools_used"] == ["search", "calc"]

    def test_runtime_run_session_not_found(self) -> None:
        """Test running with invalid session raises ValueError."""
        try:
            self.runtime.run("nonexistent", "hello")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "not found" in str(exc).lower()

    @patch("pulsar_ai.agent.base.BaseAgent.from_config")
    def test_runtime_run_handles_agent_failure(
        self,
        mock_from_config: MagicMock,
    ) -> None:
        """Test that agent failures are handled gracefully."""
        mock_from_config.side_effect = ConnectionError("No LLM")

        session = self.runtime.create_session({"name": "test"})
        result = self.runtime.run(session.session_id, "hello")

        assert result["response"] == ""
        assert "error" in result
        assert session.status == "failed"

    def test_runtime_list_sessions(self) -> None:
        """Test listing sessions with optional status filter."""
        self.runtime.create_session({"name": "a"})
        s2 = self.runtime.create_session({"name": "b"})
        s2.status = "running"

        all_sessions = self.runtime.list_sessions()
        assert len(all_sessions) == 2

        running = self.runtime.list_sessions(status="running")
        assert len(running) == 1
        assert running[0].agent_name == "b"

    def test_runtime_stop_session(self) -> None:
        """Test stopping a session marks it completed."""
        session = self.runtime.create_session({"name": "test"})
        assert self.runtime.stop_session(session.session_id) is True
        assert session.status == "completed"

    def test_runtime_stop_session_not_found(self) -> None:
        """Test stopping nonexistent session returns False."""
        assert self.runtime.stop_session("nonexistent") is False

    def test_runtime_get_session(self) -> None:
        """Test getting a session by ID."""
        session = self.runtime.create_session({"name": "test"})
        found = self.runtime.get_session(session.session_id)
        assert found is session

    def test_runtime_get_session_not_found(self) -> None:
        """Test getting nonexistent session returns None."""
        assert self.runtime.get_session("nope") is None

    def test_runtime_get_trace_empty(self) -> None:
        """Test getting trace for new session returns empty list."""
        session = self.runtime.create_session({"name": "test"})
        assert self.runtime.get_trace(session.session_id) == []

    def test_runtime_get_trace_not_found(self) -> None:
        """Test getting trace for nonexistent session returns empty list."""
        assert self.runtime.get_trace("nope") == []

    def test_runtime_health(self) -> None:
        """Test health check returns correct structure."""
        health = self.runtime.health()
        assert health["status"] == "ok"
        assert health["runtime"] == "pulsar-builtin"
        assert health["version"] == "0.1.0"
        assert health["active_sessions"] == 0
        assert health["total_sessions"] == 0

    def test_runtime_health_counts_sessions(self) -> None:
        """Test health reports correct session counts."""
        s1 = self.runtime.create_session({"name": "a"})
        s1.status = "running"
        self.runtime.create_session({"name": "b"})

        health = self.runtime.health()
        assert health["active_sessions"] == 1
        assert health["total_sessions"] == 2


class TestGetRuntime:
    """Tests for the global runtime singleton."""

    def test_get_runtime_returns_singleton(self) -> None:
        """Test that get_runtime returns the same instance."""
        import pulsar_ai.openclaw.runtime as mod

        mod._runtime = None
        r1 = get_runtime()
        r2 = get_runtime()
        assert r1 is r2
        mod._runtime = None  # cleanup


# ── OpenClaw Adapter Tests ────────────────────────────────────────


class TestOpenClawAdapter:
    """Tests for OpenClawAdapter with mocked HTTP."""

    def setup_method(self) -> None:
        self.adapter = OpenClawAdapter(
            base_url="http://localhost:8888/api/v1",
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

        mock_request.side_effect = req_lib.ConnectionError("refused")
        session = self.adapter.create_session({"name": "test"})

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


class TestNemoClawManagerWithRuntime:
    """Tests for NemoClawManager backed by the built-in runtime."""

    def setup_method(self) -> None:
        self.runtime = OpenClawRuntime()
        self.manager = NemoClawManager(self.runtime)

    def test_nemoclaw_deploy(self) -> None:
        """Test deploying an agent with sandbox policy."""
        policy = SandboxPolicy(allow_network=True, max_tokens=2048)
        deployment = self.manager.deploy(
            {"name": "test-agent", "model": "llama-3"},
            policy,
        )

        assert deployment.session_id
        assert deployment.status == "running"
        assert deployment.policy.allow_network is True
        assert deployment.policy.max_tokens == 2048

    def test_nemoclaw_stop(self) -> None:
        """Test stopping a deployment."""
        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)
        success = self.manager.stop_deployment(deployment.deployment_id)
        assert success is True

    def test_nemoclaw_list_deployments(self) -> None:
        """Test listing deployments."""
        policy = SandboxPolicy()
        self.manager.deploy({"name": "agent1"}, policy)
        self.manager.deploy({"name": "agent2"}, policy)

        deployments = self.manager.list_deployments()
        assert len(deployments) == 2

    def test_nemoclaw_policy_update(self) -> None:
        """Test updating sandbox policy for a deployment."""
        policy = SandboxPolicy()
        deployment = self.manager.deploy({"name": "test"}, policy)

        new_policy = SandboxPolicy(allow_network=True, max_tokens=8192)
        success = self.manager.update_policy(deployment.deployment_id, new_policy)
        assert success is True

        updated = self.manager.get_deployment(deployment.deployment_id)
        assert updated is not None
        assert updated.policy.allow_network is True
        assert updated.policy.max_tokens == 8192


class TestNemoClawManagerWithAdapter:
    """Tests for NemoClawManager with mocked OpenClaw adapter."""

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
    """Tests for OpenClaw API endpoints via built-in runtime."""

    def setup_method(self) -> None:
        from pulsar_ai.ui.routes import openclaw as routes_mod

        routes_mod._manager = None
        routes_mod._ingester = None
        # Reset global runtime so each test starts fresh
        import pulsar_ai.openclaw.runtime as rt_mod

        rt_mod._runtime = OpenClawRuntime()

    def _make_client(self):  # noqa: ANN202
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from pulsar_ai.ui.routes.openclaw import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_api_health_returns_ok(self) -> None:
        """Test GET /openclaw/health returns ok status."""
        client = self._make_client()
        response = client.get("/openclaw/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["runtime"] == "pulsar-builtin"

    def test_api_create_and_run_session(self) -> None:
        """Test creating and running a session via API."""
        client = self._make_client()

        # Create session
        resp = client.post(
            "/openclaw/sessions",
            json={"name": "api-agent", "model": "llama3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_name"] == "api-agent"
        session_id = data["session_id"]

        # Session should be retrievable
        resp = client.get(f"/openclaw/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    def test_api_list_sessions(self) -> None:
        """Test GET /openclaw/sessions returns session list."""
        client = self._make_client()

        client.post("/openclaw/sessions", json={"name": "a"})
        client.post("/openclaw/sessions", json={"name": "b"})

        resp = client.get("/openclaw/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2

    def test_api_stop_session(self) -> None:
        """Test DELETE /openclaw/sessions/{id} stops session."""
        client = self._make_client()

        resp = client.post("/openclaw/sessions", json={"name": "stop-me"})
        session_id = resp.json()["session_id"]

        resp = client.delete(f"/openclaw/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_api_session_not_found(self) -> None:
        """Test 404 for nonexistent session."""
        client = self._make_client()
        resp = client.get("/openclaw/sessions/nonexistent")
        assert resp.status_code == 404

    def test_api_get_trace(self) -> None:
        """Test GET /openclaw/sessions/{id}/trace returns trace."""
        client = self._make_client()

        resp = client.post("/openclaw/sessions", json={"name": "trace-test"})
        session_id = resp.json()["session_id"]

        resp = client.get(f"/openclaw/sessions/{session_id}/trace")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id
        assert resp.json()["trace"] == []

    def test_api_create_deployment(self) -> None:
        """Test POST /openclaw/deployments creates a deployment."""
        client = self._make_client()

        resp = client.post(
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
        assert resp.status_code == 200
        data = resp.json()
        assert "deployment_id" in data
        assert data["status"] == "running"
        assert data["policy"]["allow_network"] is True


# ── Trace Ingestion Tests ────────────────────────────────────────


class TestOpenClawTraceIngestion:
    """Tests for OpenClawTraceIngester with mocked adapter and store."""

    def setup_method(self) -> None:
        self.mock_adapter = MagicMock(spec=OpenClawAdapter)
        self.mock_store = MagicMock()
        self.mock_store.save_trace.side_effect = lambda data: f"trace-{data['agent_id'][:8]}"
        self.ingester = OpenClawTraceIngester(self.mock_adapter, self.mock_store)

    def test_trace_ingestion_single_session(self) -> None:
        """Test ingesting traces from a single OpenClaw session."""
        mock_session = OpenClawSession(
            session_id="sess-ingest",
            agent_name="test-agent",
            model="llama-3",
            status="completed",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.get_session.return_value = mock_session
        self.mock_adapter.get_trace.return_value = [
            {"type": "llm_response", "content": "Thinking about the query..."},
            {"type": "tool_call", "tool": "search", "content": "search query"},
            {"type": "observation", "result": "found 3 items"},
        ]

        trace_ids = self.ingester.ingest_session("sess-ingest")

        assert len(trace_ids) == 3
        assert self.mock_store.save_trace.call_count == 3
        self.mock_adapter.get_session.assert_called_once_with("sess-ingest")
        self.mock_adapter.get_trace.assert_called_once_with("sess-ingest")

    def test_trace_ingestion_all_sessions(self) -> None:
        """Test ingesting traces from all completed sessions."""
        sessions = [
            OpenClawSession(
                session_id="s1",
                agent_name="agent-a",
                model="llama",
                status="completed",
                created_at="2026-01-01T00:00:00Z",
            ),
            OpenClawSession(
                session_id="s2",
                agent_name="agent-b",
                model="mistral",
                status="completed",
                created_at="2026-01-02T00:00:00Z",
            ),
        ]
        self.mock_adapter.list_sessions.return_value = sessions
        self.mock_adapter.get_session.side_effect = lambda sid: (
            sessions[0] if sid == "s1" else sessions[1]
        )
        self.mock_adapter.get_trace.side_effect = lambda sid: (
            [{"type": "llm_response", "content": "hello"}]
            if sid == "s1"
            else [
                {"type": "tool_call", "tool": "calc", "content": "2+2"},
                {"type": "observation", "result": "4"},
            ]
        )

        result = self.ingester.ingest_all_sessions(status="completed")

        assert result["sessions_processed"] == 2
        assert result["traces_ingested"] == 3
        self.mock_adapter.list_sessions.assert_called_once_with(status="completed")

    def test_convert_trace_format(self) -> None:
        """Test converting OpenClaw trace step to Pulsar AI format."""
        session = OpenClawSession(
            session_id="sess-conv",
            agent_name="converter",
            model="gpt-4",
            status="completed",
            created_at="2026-01-01T00:00:00Z",
        )
        openclaw_step = {
            "type": "tool_call",
            "tool": "search",
            "content": "query about AI",
            "tokens": 50,
            "latency_ms": 120,
        }

        result = self.ingester._convert_trace(openclaw_step, session)

        assert result["agent_id"] == "openclaw:converter"
        assert result["model_name"] == "gpt-4"
        assert result["status"] == "success"
        assert result["tokens_used"] == 50
        assert result["latency_ms"] == 120
        assert result["trace_json"] == [openclaw_step]

    def test_trace_ingestion_session_not_found(self) -> None:
        """Test ingestion when session does not exist."""
        self.mock_adapter.get_session.return_value = None

        trace_ids = self.ingester.ingest_session("nonexistent")

        assert trace_ids == []
        self.mock_store.save_trace.assert_not_called()

    def test_trace_ingestion_empty_trace(self) -> None:
        """Test ingestion when session has no trace data."""
        mock_session = OpenClawSession(
            session_id="sess-empty",
            agent_name="empty-agent",
            model="llama",
            status="completed",
            created_at="2026-01-01T00:00:00Z",
        )
        self.mock_adapter.get_session.return_value = mock_session
        self.mock_adapter.get_trace.return_value = []

        trace_ids = self.ingester.ingest_session("sess-empty")

        assert trace_ids == []
        self.mock_store.save_trace.assert_not_called()

    def test_convert_trace_error_status(self) -> None:
        """Test that error trace steps get error status in conversion."""
        session = OpenClawSession(
            session_id="sess-err",
            agent_name="err-agent",
            model="llama",
            status="failed",
            created_at="2026-01-01T00:00:00Z",
        )
        error_step = {"type": "error", "content": "Tool execution failed"}

        result = self.ingester._convert_trace(error_step, session)

        assert result["status"] == "error"
