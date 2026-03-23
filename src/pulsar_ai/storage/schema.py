"""SQLite schema definitions for Pulsar AI persistence layer.

Each table mirrors a JSON store entity.  JSON blobs are stored as TEXT
columns and parsed on read — this keeps the schema simple while still
giving us transactions, concurrent access, and crash recovery for free.

Schema version is tracked in ``_schema_meta`` so future migrations
can inspect the current version before applying ALTER TABLE / new tables.
"""

SCHEMA_VERSION = 11

BOOTSTRAP_SQL = """
-- ── Meta ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS _schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── Experiments ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    task            TEXT NOT NULL DEFAULT 'sft',
    model           TEXT DEFAULT '',
    dataset_id      TEXT DEFAULT '',
    config          TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL,
    last_update_at  TEXT NOT NULL,
    completed_at    TEXT,
    final_loss      REAL,
    eval_results    TEXT,
    artifacts       TEXT DEFAULT '{}',
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_user
    ON experiments(user_id);

-- Append-only metrics history (one row per log_metrics call).
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    data            TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_metrics_exp
    ON experiment_metrics(experiment_id);

-- ── Prompts ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prompts (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    current_version INTEGER NOT NULL DEFAULT 1,
    tags            TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_prompts_user
    ON prompts(user_id);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id       TEXT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    system_prompt   TEXT NOT NULL,
    variables       TEXT DEFAULT '[]',
    model           TEXT DEFAULT '',
    parameters      TEXT DEFAULT '{}',
    metrics         TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(prompt_id, version)
);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_pid
    ON prompt_versions(prompt_id);

-- ── Workflows ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workflows (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    nodes           TEXT DEFAULT '[]',
    edges           TEXT DEFAULT '[]',
    schema_version  INTEGER DEFAULT 2,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_run        TEXT,
    run_count       INTEGER DEFAULT 0,
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_workflows_user
    ON workflows(user_id);

-- ── Runs (tracking.py) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    project         TEXT DEFAULT 'pulsar-ai',
    backend         TEXT DEFAULT 'local',
    status          TEXT NOT NULL,
    config          TEXT DEFAULT '{}',
    tags            TEXT DEFAULT '[]',
    metrics_history TEXT DEFAULT '[]',
    artifacts       TEXT DEFAULT '{}',
    results         TEXT DEFAULT '{}',
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    duration_s      REAL,
    environment     TEXT DEFAULT '{}',
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project);
CREATE INDEX IF NOT EXISTS idx_runs_user    ON runs(user_id);

-- ── API Keys ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    key_hash    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at  TEXT,
    revoked     INTEGER DEFAULT 0,
    user_id     TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user
    ON api_keys(user_id);

-- ── Compute Targets ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS compute_targets (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    host            TEXT NOT NULL,
    user            TEXT NOT NULL,
    port            INTEGER DEFAULT 22,
    key_path        TEXT DEFAULT '',
    gpu_count       INTEGER DEFAULT 0,
    gpu_type        TEXT DEFAULT '',
    vram_gb         REAL DEFAULT 0,
    status          TEXT DEFAULT 'unknown',
    created_at      TEXT NOT NULL,
    last_heartbeat  TEXT
);

-- ── Jobs ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    experiment_id   TEXT,
    status          TEXT NOT NULL DEFAULT 'queued',
    job_type        TEXT NOT NULL DEFAULT 'sft',
    config          TEXT DEFAULT '{}',
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    error_message   TEXT,
    pid             INTEGER,
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user   ON jobs(user_id);

-- ── Assistant Sessions ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS assistant_sessions (
    id              TEXT PRIMARY KEY,
    session_type    TEXT NOT NULL DEFAULT 'assistant',
    messages        TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    ttl_hours       INTEGER DEFAULT 24,
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_assistant_sessions_user
    ON assistant_sessions(user_id);

-- ── Users ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    name            TEXT DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'user',
    is_active       INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ── API Key Events (audit trail) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS api_key_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    ip_address  TEXT DEFAULT ''
);

-- ── Traces (agent execution traces) ─────────────────────────────
CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    agent_id      TEXT DEFAULT '',
    model_name    TEXT DEFAULT '',
    model_version TEXT DEFAULT '',
    user_query    TEXT NOT NULL,
    response      TEXT DEFAULT '',
    trace_json    TEXT DEFAULT '[]',
    status        TEXT DEFAULT 'success',
    tokens_used   INTEGER DEFAULT 0,
    cost          REAL DEFAULT 0.0,
    latency_ms    INTEGER DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now')),
    user_id       TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_traces_created
    ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_user
    ON traces(user_id);
CREATE INDEX IF NOT EXISTS idx_traces_model
    ON traces(model_name);
CREATE INDEX IF NOT EXISTS idx_traces_status
    ON traces(status);

-- ── Trace Feedback ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trace_feedback (
    id            TEXT PRIMARY KEY,
    trace_id      TEXT NOT NULL REFERENCES traces(trace_id),
    feedback_type TEXT NOT NULL DEFAULT 'thumbs',
    rating        REAL DEFAULT 0,
    reason        TEXT DEFAULT '',
    chosen        TEXT DEFAULT '',
    rejected      TEXT DEFAULT '',
    user_id       TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trace_feedback_trace
    ON trace_feedback(trace_id);

-- ── Agent Eval Reports ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_eval_reports (
    id              TEXT PRIMARY KEY,
    suite_name      TEXT NOT NULL,
    model_name      TEXT DEFAULT '',
    timestamp       TEXT DEFAULT (datetime('now')),
    success_rate    REAL DEFAULT 0,
    avg_score       REAL DEFAULT 0,
    avg_latency_ms  REAL DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    total_cost      REAL DEFAULT 0,
    tools_accuracy  REAL DEFAULT 0,
    results_json    TEXT DEFAULT '[]',
    by_tag_json     TEXT DEFAULT '{}',
    user_id         TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_agent_eval_reports_user
    ON agent_eval_reports(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_eval_reports_suite
    ON agent_eval_reports(suite_name);
CREATE INDEX IF NOT EXISTS idx_agent_eval_reports_model
    ON agent_eval_reports(model_name);

-- ── Benchmarks ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS benchmarks (
    id                          TEXT PRIMARY KEY,
    model_path                  TEXT NOT NULL,
    model_name                  TEXT NOT NULL DEFAULT '',
    experiment_id               TEXT DEFAULT '',
    hardware_info               TEXT DEFAULT '{}',
    timestamp                   TEXT DEFAULT (datetime('now')),
    tokens_per_sec              REAL DEFAULT 0,
    time_to_first_token_ms      REAL DEFAULT 0,
    training_samples_per_sec    REAL DEFAULT 0,
    peak_vram_gb                REAL DEFAULT 0,
    model_size_params           INTEGER DEFAULT 0,
    model_size_disk_mb          REAL DEFAULT 0,
    perplexity                  REAL,
    eval_loss                   REAL,
    task_metrics                TEXT DEFAULT '{}',
    estimated_cost_per_1m_tokens REAL DEFAULT 0,
    config                      TEXT DEFAULT '{}',
    status                      TEXT DEFAULT 'completed',
    tags                        TEXT DEFAULT '[]',
    is_baseline                 INTEGER DEFAULT 0,
    user_id                     TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_user
    ON benchmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_model
    ON benchmarks(model_name);
CREATE INDEX IF NOT EXISTS idx_benchmarks_experiment
    ON benchmarks(experiment_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_timestamp
    ON benchmarks(timestamp);

-- ── Cluster Configs (saved multi-node presets) ──────────────────
CREATE TABLE IF NOT EXISTS cluster_configs (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    target_ids        TEXT NOT NULL DEFAULT '[]',
    master_target_id  TEXT NOT NULL,
    strategy          TEXT DEFAULT 'fsdp_qlora',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- ── Distributed Training Metrics (per-rank GPU stats) ───────────
CREATE TABLE IF NOT EXISTS distributed_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    rank            INTEGER NOT NULL,
    target_id       TEXT DEFAULT '',
    gpu_index       INTEGER DEFAULT 0,
    step            INTEGER NOT NULL,
    loss            REAL,
    vram_used_gb    REAL,
    gpu_util_pct    REAL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_distributed_metrics_job
    ON distributed_metrics(job_id);

-- ── Workspaces ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workspaces (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,
    owner_id        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL DEFAULT 'member',
    joined_at       TEXT NOT NULL,
    UNIQUE(workspace_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_members_ws
    ON workspace_members(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workspace_members_user
    ON workspace_members(user_id);

-- ── Approval Requests ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS approval_requests (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT DEFAULT '',
    resource_type   TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    action          TEXT NOT NULL DEFAULT 'execute',
    requester_id    TEXT NOT NULL,
    approver_id     TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    reason          TEXT DEFAULT '',
    review_note     TEXT DEFAULT '',
    decided_at      TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_approval_requests_ws_status
    ON approval_requests(workspace_id, status);

-- ── Audit Logs ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT DEFAULT '',
    workspace_id    TEXT DEFAULT '',
    action          TEXT NOT NULL,
    resource_type   TEXT NOT NULL,
    resource_id     TEXT DEFAULT '',
    details         TEXT DEFAULT '{}',
    ip_address      TEXT DEFAULT '',
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_ws_ts
    ON audit_logs(workspace_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_ts
    ON audit_logs(user_id, timestamp);

-- ── Token Blacklist (revoked JWTs) ──────────────────────────────
CREATE TABLE IF NOT EXISTS token_blacklist (
    jti         TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    revoked_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_blacklist_user
    ON token_blacklist(user_id);
CREATE INDEX IF NOT EXISTS idx_token_blacklist_expires
    ON token_blacklist(expires_at);

-- ── Login Attempts (brute-force protection) ──────────────────────
CREATE TABLE IF NOT EXISTS login_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL,
    ip_address  TEXT DEFAULT '',
    success     INTEGER NOT NULL DEFAULT 0,
    attempted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_attempts_email ON login_attempts(email, attempted_at);
CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address, attempted_at);

-- ── User MFA (TOTP / backup codes) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS user_mfa (
    user_id     TEXT PRIMARY KEY REFERENCES users(id),
    totp_secret TEXT NOT NULL,
    backup_codes TEXT DEFAULT '[]',
    enabled     INTEGER DEFAULT 0,
    verified_at TEXT,
    created_at  TEXT NOT NULL
);

-- ── OIDC State (CSRF protection for SSO flows) ────────────────────
CREATE TABLE IF NOT EXISTS oidc_states (
    state       TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
"""
