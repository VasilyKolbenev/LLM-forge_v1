const BASE = "/api/v1"

// Benchmark types used by api methods and Benchmarks page
export interface BenchmarkHardwareInfo {
  gpu_name: string
  vram_gb: number
  num_gpus: number
}

export interface Benchmark {
  id: string
  model_path: string
  model_name: string
  experiment_id: string
  hardware_info: BenchmarkHardwareInfo
  timestamp: string
  tokens_per_sec: number
  time_to_first_token_ms: number
  training_samples_per_sec: number
  peak_vram_gb: number
  model_size_params: number
  model_size_disk_mb: number
  perplexity: number | null
  eval_loss: number | null
  task_metrics: Record<string, number>
  estimated_cost_per_1m_tokens: number
  config: Record<string, unknown>
  status: string
  tags: string[]
  is_baseline: boolean
}

export interface CompareResult {
  baseline: Benchmark
  candidates: Benchmark[]
  deltas: Record<string, Record<string, number>>
}

export interface AdminUser {
  id: string
  email: string
  name: string
  role: string
  is_active: boolean
  last_login: string | null
  created_at: string
}

export interface SystemHealth {
  database: { connected: boolean; type: string }
  redis: { connected: boolean; latency_ms: number }
  s3: { configured: boolean; bucket: string }
  disk: { total_gb: number; used_gb: number; free_gb: number }
  memory: { total_gb: number; used_gb: number; free_gb: number }
}

export interface SystemStats {
  total_users: number
  active_users: number
  total_experiments: number
  total_datasets: number
  uptime_seconds: number
}

let _apiKey: string | null = localStorage.getItem("pulsar_api_key")

export function setApiKey(key: string | null) {
  _apiKey = key
  if (key) localStorage.setItem("pulsar_api_key", key)
  else localStorage.removeItem("pulsar_api_key")
}

export function getApiKey(): string | null {
  return _apiKey
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  }
  if (_apiKey) {
    headers["Authorization"] = `Bearer ${_apiKey}`
  }
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...headers,
      ...options?.headers,
    },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `HTTP ${res.status}`)
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

export const api = {
  // Training
  startTraining: (data: { name: string; config: Record<string, unknown>; task: string }) =>
    request<{ job_id: string; experiment_id: string; status: string }>(
      "/training/start", { method: "POST", body: JSON.stringify(data) }
    ),
  getJobs: () => request<Array<Record<string, unknown>>>("/training/jobs"),
  cancelJob: (id: string) => request<Record<string, unknown>>(`/training/jobs/${id}`, { method: "DELETE" }),

  // Datasets
  uploadDataset: async (file: File) => {
    const form = new FormData()
    form.append("file", file)
    const headers: Record<string, string> = {}
    if (_apiKey) {
      headers["Authorization"] = `Bearer ${_apiKey}`
    }
    const res = await fetch(`${BASE}/datasets/upload`, { method: "POST", body: form, headers })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || `HTTP ${res.status}`)
    }
    return res.json()
  },
  getDatasets: () => request<Array<Record<string, unknown>>>("/datasets"),
  previewDataset: (id: string, rows = 20) =>
    request<{ columns: string[]; rows: Array<Record<string, unknown>>; total_rows: number }>(
      `/datasets/${id}/preview?rows=${rows}`
    ),
  deleteDataset: (id: string) => request<Record<string, unknown>>(`/datasets/${id}`, { method: "DELETE" }),

  // Experiments
  getExperiments: (status?: string) =>
    request<Array<Record<string, unknown>>>(`/experiments${status ? `?status=${status}` : ""}`),
  getExperiment: (id: string) => request<Record<string, unknown>>(`/experiments/${id}`),
  compareExperiments: (ids: string[]) =>
    request<Record<string, unknown>>("/experiments/compare", {
      method: "POST", body: JSON.stringify({ experiment_ids: ids }),
    }),
  deleteExperiment: (id: string) => request<Record<string, unknown>>(`/experiments/${id}`, { method: "DELETE" }),

  // Eval & Export
  runEval: (data: { experiment_id: string; test_data_path: string; batch_size?: number }) =>
    request<Record<string, unknown>>("/evaluation/run", { method: "POST", body: JSON.stringify(data) }),
  exportModel: (data: { experiment_id: string; format?: string; quantization?: string }) =>
    request<Record<string, unknown>>("/export", { method: "POST", body: JSON.stringify(data) }),

  // Hardware
  getHardware: () => request<Record<string, unknown>>("/hardware"),

  // Metrics
  metricsSnapshot: () => request<Record<string, unknown>>("/metrics/snapshot"),

  // Compute
  computeTargets: () => request<Array<Record<string, unknown>>>("/compute/targets"),
  addComputeTarget: (data: { name: string; host: string; user: string; port?: number; key_path?: string }) =>
    request<Record<string, unknown>>("/compute/targets", { method: "POST", body: JSON.stringify(data) }),
  removeComputeTarget: (id: string) =>
    request<Record<string, unknown>>(`/compute/targets/${id}`, { method: "DELETE" }),
  testComputeTarget: (id: string) =>
    request<{ success: boolean; message: string; latency_ms: number }>(`/compute/targets/${id}/test`, { method: "POST" }),
  detectComputeHardware: (id: string) =>
    request<Record<string, unknown>>(`/compute/targets/${id}/detect`, { method: "POST" }),

  // Workflows
  listWorkflows: () => request<Array<Record<string, unknown>>>("/workflows"),
  saveWorkflow: (data: { name: string; nodes: Record<string, unknown>[]; edges: Record<string, unknown>[]; workflow_id?: string }) =>
    request<Record<string, unknown>>("/workflows", { method: "POST", body: JSON.stringify(data) }),
  getWorkflow: (id: string) => request<Record<string, unknown>>(`/workflows/${id}`),
  deleteWorkflow: (id: string) => request<Record<string, unknown>>(`/workflows/${id}`, { method: "DELETE" }),
  runWorkflow: (id: string) => request<Record<string, unknown>>(`/workflows/${id}/run`, { method: "POST" }),
  runPipelineSync: (pipelineConfig: Record<string, unknown>) =>
    request<Record<string, unknown>>("/pipeline/run/sync", {
      method: "POST",
      body: JSON.stringify({ pipeline_config: pipelineConfig }),
    }),
  getWorkflowConfig: (id: string) => request<Record<string, unknown>>(`/workflows/${id}/config`),
  getPipelineTrace: (workflowId: string) => request<Record<string, unknown>>(`/pipeline/trace/${workflowId}`),
  listWorkflowTemplates: () => request<Array<Record<string, unknown>>>("/workflows/templates"),
  createWorkflowFromTemplate: (templateId: string, data?: { name?: string }) =>
    request<Record<string, unknown>>(`/workflows/templates/${templateId}/create`, { method: "POST", body: JSON.stringify(data || {}) }),

  // Prompts
  listPrompts: (tag?: string) =>
    request<Array<Record<string, unknown>>>(`/prompts${tag ? `?tag=${tag}` : ""}`),
  createPrompt: (data: { name: string; system_prompt: string; description?: string; model?: string; parameters?: Record<string, unknown>; tags?: string[] }) =>
    request<Record<string, unknown>>("/prompts", { method: "POST", body: JSON.stringify(data) }),
  getPrompt: (id: string) => request<Record<string, unknown>>(`/prompts/${id}`),
  updatePrompt: (id: string, data: { name?: string; description?: string; tags?: string[] }) =>
    request<Record<string, unknown>>(`/prompts/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deletePrompt: (id: string) => request<Record<string, unknown>>(`/prompts/${id}`, { method: "DELETE" }),
  addPromptVersion: (id: string, data: { system_prompt: string; model?: string; parameters?: Record<string, unknown> }) =>
    request<Record<string, unknown>>(`/prompts/${id}/versions`, { method: "POST", body: JSON.stringify(data) }),
  getPromptVersion: (id: string, version: number) =>
    request<Record<string, unknown>>(`/prompts/${id}/versions/${version}`),
  diffPromptVersions: (id: string, v1: number, v2: number) =>
    request<Record<string, unknown>>(`/prompts/${id}/diff?v1=${v1}&v2=${v2}`),
  testPrompt: (id: string, data: { variables?: Record<string, string>; version?: number }) =>
    request<Record<string, unknown>>(`/prompts/${id}/test`, { method: "POST", body: JSON.stringify(data) }),

  // Settings
  getSettings: () =>
    request<{
      version: string
      auth_enabled: boolean
      stand_mode: string
      env_profile: string
      cors_origins: string[]
      data_dir: string
    }>("/settings"),
  listApiKeys: () => request<Array<{ name: string }>>("/settings/keys"),
  generateApiKey: (name: string) =>
    request<{ key: string; name: string }>("/settings/keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  revokeApiKey: (name: string) =>
    request<{ status: string }>(`/settings/keys/${name}`, { method: "DELETE" }),

  // Lineage
  getLineageGraph: (modelName?: string) =>
    request<{ nodes: Array<{ id: string; data: Record<string, unknown> }>; edges: Array<{ id: string; source: string; target: string }> }>(
      `/lineage/graph${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ""}`,
    ),
  getLineageTimeline: (limit?: number) =>
    request<{ events: Array<Record<string, unknown>> }>(
      `/lineage/timeline${limit ? `?limit=${limit}` : ""}`,
    ),

  // Agent Eval
  getAgentEvalReports: (modelName?: string) =>
    request<{ reports: Array<Record<string, unknown>>; total: number }>(
      `/agent-eval/reports${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ""}`,
    ),
  getAgentEvalReport: (id: string) =>
    request<Record<string, unknown>>(`/agent-eval/reports/${id}`),
  runAgentEval: (suiteFile: string, agentConfig: Record<string, unknown>, scoring: string) =>
    request<Record<string, unknown>>("/agent-eval/run", {
      method: "POST",
      body: JSON.stringify({ suite_path: suiteFile, agent_config: agentConfig, scoring }),
    }),
  compareAgentEvalReports: (idA: string, idB: string) =>
    request<Record<string, unknown>>(`/agent-eval/compare/${idA}/${idB}`),
  getAgentEvalSuites: () =>
    request<Array<{ path: string; name: string; description: string; num_cases: number; version: string }>>(
      "/agent-eval/suites",
    ),

  // OpenClaw
  getOpenClawHealth: () => request<Record<string, unknown>>("/openclaw/health"),
  getOpenClawSessions: (status?: string) =>
    request<Record<string, unknown>>(`/openclaw/sessions${status ? `?status=${status}` : ""}`),
  createOpenClawSession: (config: { name: string; model: string; tools: string[]; system_prompt: string }) =>
    request<Record<string, unknown>>("/openclaw/sessions", { method: "POST", body: JSON.stringify(config) }),
  runOpenClawSession: (id: string, input: string) =>
    request<Record<string, unknown>>(`/openclaw/sessions/${id}/run`, {
      method: "POST", body: JSON.stringify({ input }),
    }),
  stopOpenClawSession: (id: string) =>
    request<Record<string, unknown>>(`/openclaw/sessions/${id}`, { method: "DELETE" }),
  getOpenClawSessionTrace: (id: string) =>
    request<Record<string, unknown>>(`/openclaw/sessions/${id}/trace`),
  ingestOpenClawTraces: (sessionId: string) =>
    request<Record<string, unknown>>(`/openclaw/sessions/${sessionId}/ingest`, { method: "POST" }),
  getOpenClawDeployments: () =>
    request<Record<string, unknown>>("/openclaw/deployments"),
  createOpenClawDeployment: (
    config: { name: string; model: string; tools: string[]; system_prompt: string },
    policy: Record<string, unknown>,
  ) =>
    request<Record<string, unknown>>("/openclaw/deployments", {
      method: "POST",
      body: JSON.stringify({ ...config, policy }),
    }),
  stopOpenClawDeployment: (id: string) =>
    request<Record<string, unknown>>(`/openclaw/deployments/${id}`, { method: "DELETE" }),

  // Benchmarks
  getBenchmarks: () =>
    request<{ benchmarks: Benchmark[] }>("/benchmarks"),
  getBenchmarkLeaderboard: (metric: string, order: string) =>
    request<{ leaderboard: Benchmark[] }>(
      `/benchmarks/leaderboard?metric=${metric}&order=${order}`,
    ),
  compareBenchmarks: (ids: string[]) =>
    request<CompareResult>(`/benchmarks/compare?ids=${ids.join(",")}`),
  deleteBenchmark: (id: string) =>
    request<Record<string, unknown>>(`/benchmarks/${id}`, { method: "DELETE" }),
  runBenchmark: (data: {
    model_path: string
    model_name: string
    gpu_cost?: number
    tags?: string[]
  }) =>
    request<Record<string, unknown>>("/benchmarks/run", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Governance
  getGovernanceApprovals: (status?: string) =>
    request<{ approvals: Array<Record<string, unknown>> }>(
      `/governance/approvals${status ? `?status=${status}` : ""}`,
    ),
  reviewGovernanceApproval: (id: string, decision: { status: string; reason?: string }) =>
    request<Record<string, unknown>>(`/governance/approvals/${id}/review`, {
      method: "POST",
      body: JSON.stringify(decision),
    }),
  getGovernanceAudit: (resourceType?: string) =>
    request<{ events: Array<Record<string, unknown>> }>(
      `/governance/audit${resourceType ? `?resource_type=${resourceType}` : ""}`,
    ),

  // Admin - Users
  getAdminUsers: () => request<{ users: AdminUser[] }>("/admin/users"),
  getAdminUser: (id: string) => request<AdminUser>(`/admin/users/${id}`),
  updateAdminUser: (id: string, data: { name?: string; role?: string; is_active?: boolean }) =>
    request(`/admin/users/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  resetUserPassword: (id: string) =>
    request<{ temporary_password: string }>(`/admin/users/${id}/reset-password`, { method: "POST" }),
  deactivateUser: (id: string) =>
    request(`/admin/users/${id}/deactivate`, { method: "POST" }),
  activateUser: (id: string) =>
    request(`/admin/users/${id}/activate`, { method: "POST" }),
  forceLogoutUser: (id: string) =>
    request(`/admin/users/${id}/force-logout`, { method: "POST" }),
  // Admin - System
  getSystemHealth: () => request<SystemHealth>("/admin/system/health"),
  getSystemStats: () => request<SystemStats>("/admin/system/stats"),
  runSystemCleanup: () =>
    request<{ cleaned: Record<string, number> }>("/admin/system/cleanup", { method: "POST" }),
  getSystemConfig: () => request<Record<string, unknown>>("/admin/system/config"),

  // Health
  health: () => request<{ status: string }>("/health"),

  // Agent
  agentChat: (data: { message: string }) =>
    request<{ message?: string; answer?: string; trace_id?: string }>("/agent/chat", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Assistant
  assistantChat: (data: { message: string; session_id?: string; context?: Record<string, unknown> }) =>
    request<{
      answer: string
      session_id: string
      actions: Array<Record<string, unknown>>
      mode: string
    }>("/assistant/chat", { method: "POST", body: JSON.stringify(data) }),
  assistantStatus: () =>
    request<{
      active_jobs: Array<Record<string, unknown>>
      recent_experiments: Array<Record<string, unknown>>
      llm_available: boolean
    }>("/assistant/status"),
}

export function sseUrl(jobId: string) {
  return `${BASE}/training/progress/${jobId}`
}

export function metricsLiveUrl() {
  return `${BASE}/metrics/live`
}
