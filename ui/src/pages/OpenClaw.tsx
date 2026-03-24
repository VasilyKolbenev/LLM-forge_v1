import { useState, useEffect, useCallback, Fragment } from "react"
import {
  Cpu,
  RefreshCw,
  Play,
  Square,
  Download,
  Plus,
  ChevronDown,
  ChevronRight,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  Shield,
  Check,
  X,
  Clock,
  FileText,
  Lock,
  Unlock,
} from "lucide-react"
import { api } from "@/api/client"

type TabId = "sessions" | "deployments" | "governance"

interface Session {
  session_id: string
  agent_name: string
  model: string
  status: string
  created_at: string
  tools: string[]
  metadata: Record<string, unknown>
}

interface Deployment {
  deployment_id: string
  session_id: string
  policy: SandboxPolicy
  status: string
  created_at: string
  health: Record<string, unknown>
}

interface SandboxPolicy {
  allow_network: boolean
  allowed_domains: string[]
  allow_file_write: boolean
  allowed_paths: string[]
  max_memory_mb: number
  max_cpu_seconds: number
  max_tokens: number
}

interface HealthStatus {
  status: string
  version?: string
  error?: string
}

interface GovernanceApproval {
  id: string
  session_id: string
  agent_name: string
  status: "pending" | "approved" | "rejected"
  requested_at: string
  reviewed_at?: string
  reviewer?: string
  reason?: string
}

interface AuditEvent {
  id: string
  timestamp: string
  event_type: string
  session_id: string
  details: string
  actor?: string
}

const AGENT_TEMPLATES = [
  {
    id: "code-reviewer",
    name: "Code Reviewer",
    model: "qwen3.5:4b",
    tools: "read_file, search_files",
    system_prompt: "You are an expert Python code reviewer. When given a file path, read it and provide a detailed review covering:\\n1. Bugs and logic errors\\n2. Security vulnerabilities\\n3. Resource leaks\\n4. Best practice violations\\n5. Performance issues\\nFor each issue, specify line number, severity, and suggested fix.",
    demo_input: "Review data/demo/review_target.py for bugs and security issues",
  },
  {
    id: "data-analyst",
    name: "Data Analyst",
    model: "qwen3.5:4b",
    tools: "read_file, calculate",
    system_prompt: "You are a data analyst. Read CSV files and provide:\\n1. Dataset overview\\n2. Key statistics\\n3. Risk analysis\\n4. Actionable insights\\nUse read_file to load data and calculate for computations.",
    demo_input: "Analyze data/demo/loan_report.csv — approval rate, average amounts, risk patterns",
  },
  {
    id: "doc-summarizer",
    name: "Document Summarizer",
    model: "qwen3.5:4b",
    tools: "read_file",
    system_prompt: "You are a technical document summarizer. Provide:\\n1. Executive summary (2-3 sentences)\\n2. Key points\\n3. Technical concepts explained\\n4. Practical implications",
    demo_input: "Summarize data/demo/article.txt — focus on practical implications for ML teams",
  },
  {
    id: "math-tutor",
    name: "Math Tutor",
    model: "qwen3.5:4b",
    tools: "calculate, read_file",
    system_prompt: "You are a math tutor for ML/AI. Break down problems step by step, use the calculate tool for each computation, and explain WHY each step is needed.",
    demo_input: "Read data/demo/math_problems.json and solve problem #3 step by step",
  },
]

export function OpenClaw() {
  const [tab, setTab] = useState<TabId>("sessions")
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [expandedTrace, setExpandedTrace] = useState<Record<string, unknown>[] | null>(null)

  // Governance
  const [approvals, setApprovals] = useState<GovernanceApproval[]>([])
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
  const [governanceLoading, setGovernanceLoading] = useState(false)

  // Modals
  const [showCreateSession, setShowCreateSession] = useState(false)
  const [showCreateDeployment, setShowCreateDeployment] = useState(false)

  const [ollamaStatus, setOllamaStatus] = useState<{ connected: boolean; models: Array<{ name: string; size_gb: number }> } | null>(null)
  const [sessionInput, setSessionInput] = useState<Record<string, string>>({})
  const [sessionOutput, setSessionOutput] = useState<Record<string, string>>({})
  const [runningSession, setRunningSession] = useState<string | null>(null)

  // Create session form
  const [sessionForm, setSessionForm] = useState({
    name: "",
    model: "",
    tools: "",
    system_prompt: "",
  })

  // Create deployment form
  const [deployForm, setDeployForm] = useState({
    name: "",
    model: "",
    tools: "",
    system_prompt: "",
    allow_network: false,
    allow_file_write: false,
    max_memory_mb: 512,
    max_cpu_seconds: 60,
    max_tokens: 4096,
  })

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.getOpenClawHealth()
      setHealth(data as HealthStatus)
    } catch {
      setHealth({ status: "unavailable", error: "Cannot connect" })
    }
  }, [])

  const fetchSessions = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getOpenClawSessions()
      setSessions((data as { sessions: Session[] }).sessions || [])
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchDeployments = useCallback(async () => {
    try {
      const data = await api.getOpenClawDeployments()
      setDeployments((data as { deployments: Deployment[] }).deployments || [])
    } catch {
      /* ignore */
    }
  }, [])

  const fetchGovernance = useCallback(async () => {
    setGovernanceLoading(true)
    try {
      const [approvalsData, auditData] = await Promise.all([
        api.getGovernanceApprovals("pending"),
        api.getGovernanceAudit("openclaw-session"),
      ])
      setApprovals((approvalsData.approvals || []) as GovernanceApproval[])
      setAuditEvents((auditData.events || []) as AuditEvent[])
    } catch {
      /* ignore */
    } finally {
      setGovernanceLoading(false)
    }
  }, [])

  const refresh = useCallback(() => {
    fetchHealth()
    fetchSessions()
    fetchDeployments()
    fetchGovernance()
    api.ollamaModels().then(data => setOllamaStatus(data as any)).catch(() => setOllamaStatus({ connected: false, models: [] }))
  }, [fetchHealth, fetchSessions, fetchDeployments, fetchGovernance])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleRunSession = async (id: string) => {
    try {
      await api.runOpenClawSession(id, "")
      fetchSessions()
    } catch {
      /* ignore */
    }
  }

  const handleRunWithInput = async (sessionId: string, input: string) => {
    if (!input.trim()) return
    setRunningSession(sessionId)
    setSessionOutput(prev => ({ ...prev, [sessionId]: "Thinking..." }))
    try {
      const data = await api.runOpenClawSession(sessionId, input)
      const result = data as Record<string, unknown>
      const output = result.output || result.response || result.trace || JSON.stringify(result, null, 2)
      setSessionOutput(prev => ({ ...prev, [sessionId]: String(output) }))
      setSessionInput(prev => ({ ...prev, [sessionId]: "" }))
      loadSessionTrace(sessionId)
    } catch (e) {
      setSessionOutput(prev => ({ ...prev, [sessionId]: `Error: ${e}` }))
    } finally {
      setRunningSession(null)
    }
  }

  const handleStopSession = async (id: string) => {
    try {
      await api.stopOpenClawSession(id)
      fetchSessions()
    } catch {
      /* ignore */
    }
  }

  const handleIngestTraces = async (id: string) => {
    try {
      await api.ingestOpenClawTraces(id)
      fetchSessions()
    } catch {
      /* ignore */
    }
  }

  const handleStopDeployment = async (id: string) => {
    try {
      await api.stopOpenClawDeployment(id)
      fetchDeployments()
    } catch {
      /* ignore */
    }
  }

  const handleReviewApproval = async (id: string, status: "approved" | "rejected") => {
    try {
      await api.reviewGovernanceApproval(id, { status })
      fetchGovernance()
    } catch {
      /* ignore */
    }
  }

  const handleCreateSession = async () => {
    try {
      await api.createOpenClawSession({
        name: sessionForm.name,
        model: sessionForm.model,
        tools: sessionForm.tools
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
        system_prompt: sessionForm.system_prompt,
      })
      setShowCreateSession(false)
      setSessionForm({ name: "", model: "", tools: "", system_prompt: "" })
      fetchSessions()
    } catch {
      /* ignore */
    }
  }

  const handleCreateDeployment = async () => {
    try {
      await api.createOpenClawDeployment(
        {
          name: deployForm.name,
          model: deployForm.model,
          tools: deployForm.tools
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
          system_prompt: deployForm.system_prompt,
        },
        {
          allow_network: deployForm.allow_network,
          allowed_domains: [],
          allow_file_write: deployForm.allow_file_write,
          allowed_paths: [],
          max_memory_mb: deployForm.max_memory_mb,
          max_cpu_seconds: deployForm.max_cpu_seconds,
          max_tokens: deployForm.max_tokens,
        },
      )
      setShowCreateDeployment(false)
      setDeployForm({
        name: "",
        model: "",
        tools: "",
        system_prompt: "",
        allow_network: false,
        allow_file_write: false,
        max_memory_mb: 512,
        max_cpu_seconds: 60,
        max_tokens: 4096,
      })
      fetchDeployments()
    } catch {
      /* ignore */
    }
  }

  const loadSessionTrace = async (sessionId: string) => {
    if (expandedId === sessionId) {
      setExpandedId(null)
      setExpandedTrace(null)
      return
    }
    setExpandedId(sessionId)
    try {
      const data = await api.getOpenClawSessionTrace(sessionId)
      setExpandedTrace((data as { trace: Record<string, unknown>[] }).trace || [])
    } catch {
      setExpandedTrace([])
    }
  }

  const isHealthy = health?.status === "healthy"

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Cpu size={24} /> OpenClaw
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Agent runtime sessions and sandboxed deployments
          </p>
        </div>
        <button
          onClick={refresh}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Health status */}
      <div className="flex items-center gap-4">
        <HealthIndicator label="OpenClaw" healthy={isHealthy} detail={health?.version || health?.error || ""} />
        <HealthIndicator label="NemoClaw" healthy={isHealthy} detail="Sandbox runtime" />
        <HealthIndicator
          label="Ollama"
          healthy={ollamaStatus?.connected ?? false}
          detail={ollamaStatus?.connected ? `${ollamaStatus.models.length} models` : "Not connected"}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        <TabButton active={tab === "sessions"} onClick={() => setTab("sessions")}>
          Sessions ({sessions.length})
        </TabButton>
        <TabButton active={tab === "deployments"} onClick={() => setTab("deployments")}>
          Deployments ({deployments.length})
        </TabButton>
        <TabButton active={tab === "governance"} onClick={() => setTab("governance")}>
          <span className="flex items-center gap-1.5">
            <Shield size={14} />
            Governance
            {approvals.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 text-[10px] rounded-full bg-yellow-500/20 text-yellow-500 border border-yellow-500/30">
                {approvals.length}
              </span>
            )}
          </span>
        </TabButton>
      </div>

      {/* Sessions tab */}
      {tab === "sessions" && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            {AGENT_TEMPLATES.map((t) => (
              <button
                key={t.id}
                onClick={() => {
                  setSessionForm({
                    name: t.id,
                    model: t.model,
                    tools: t.tools,
                    system_prompt: t.system_prompt,
                  })
                  setShowCreateSession(true)
                }}
                className="px-2.5 py-1 text-xs rounded-md border border-primary/30 bg-primary/5 text-primary hover:bg-primary/10 transition-colors"
              >
                {t.name}
              </button>
            ))}
            <button
              onClick={() => setShowCreateSession(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
            >
              <Plus size={14} /> Custom Session
            </button>
          </div>

          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-card border-b border-border">
                <tr>
                  <th className="px-3 py-2 text-left">Session ID</th>
                  <th className="px-3 py-2 text-left">Agent</th>
                  <th className="px-3 py-2 text-left">Model</th>
                  <th className="px-3 py-2 text-left">Status</th>
                  <th className="px-3 py-2 text-left">Created</th>
                  <th className="px-3 py-2 text-right">Tools</th>
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-muted-foreground">
                      Loading...
                    </td>
                  </tr>
                )}
                {!loading && sessions.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-muted-foreground">
                      No sessions found
                    </td>
                  </tr>
                )}
                {sessions.map((s) => (
                  <Fragment key={s.session_id}>
                    <tr
                      onClick={() => loadSessionTrace(s.session_id)}
                      className="border-b border-border hover:bg-secondary/50 cursor-pointer transition-colors"
                    >
                      <td className="px-3 py-2 font-mono text-xs">
                        {expandedId === s.session_id ? (
                          <ChevronDown size={14} className="inline mr-1" />
                        ) : (
                          <ChevronRight size={14} className="inline mr-1" />
                        )}
                        {s.session_id}
                      </td>
                      <td className="px-3 py-2">{s.agent_name}</td>
                      <td className="px-3 py-2 text-muted-foreground">{s.model}</td>
                      <td className="px-3 py-2">
                        <StatusBadge status={s.status} />
                      </td>
                      <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                        {new Date(s.created_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right text-muted-foreground">
                        {s.tools.length}
                      </td>
                      <td className="px-3 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1">
                          {(s.status === "created" || s.status === "running") && (
                            <ActionButton
                              icon={<Play size={12} />}
                              title="Run"
                              onClick={() => handleRunSession(s.session_id)}
                            />
                          )}
                          {(s.status === "created" || s.status === "running") && (
                            <ActionButton
                              icon={<Square size={12} />}
                              title="Stop"
                              onClick={() => handleStopSession(s.session_id)}
                            />
                          )}
                          <ActionButton
                            icon={<Download size={12} />}
                            title="Ingest Traces"
                            onClick={() => handleIngestTraces(s.session_id)}
                          />
                        </div>
                      </td>
                    </tr>
                    {expandedId === s.session_id && (
                      <tr key={`${s.session_id}-detail`}>
                        <td colSpan={7} className="bg-card/50 px-6 py-4">
                          <SessionDetail
                            session={s}
                            trace={expandedTrace}
                            onIngest={() => handleIngestTraces(s.session_id)}
                          />
                        </td>
                      </tr>
                    )}
                    {s.status === "running" && expandedId === s.session_id && (
                      <tr className="bg-card/50">
                        <td colSpan={7} className="px-4 py-3">
                          <div className="space-y-2">
                            <div className="flex gap-2">
                              <input
                                value={sessionInput[s.session_id] || ""}
                                onChange={(e) => setSessionInput(prev => ({ ...prev, [s.session_id]: e.target.value }))}
                                placeholder="Send a message to the agent..."
                                className="flex-1 px-3 py-2 text-sm rounded-md bg-background border border-border"
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" && sessionInput[s.session_id]) {
                                    handleRunWithInput(s.session_id, sessionInput[s.session_id])
                                  }
                                }}
                              />
                              <button
                                onClick={() => handleRunWithInput(s.session_id, sessionInput[s.session_id] || "")}
                                disabled={!sessionInput[s.session_id] || runningSession === s.session_id}
                                className="px-3 py-2 text-sm rounded-md bg-primary text-primary-foreground disabled:opacity-50"
                              >
                                {runningSession === s.session_id ? "Running..." : "Send"}
                              </button>
                            </div>
                            {sessionOutput[s.session_id] && (
                              <div className="mt-2 p-3 rounded-md bg-background border border-border text-sm whitespace-pre-wrap font-mono text-xs max-h-96 overflow-auto">
                                {sessionOutput[s.session_id]}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Deployments tab */}
      {tab === "deployments" && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <button
              onClick={() => setShowCreateDeployment(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
            >
              <Plus size={14} /> New Deployment
            </button>
          </div>

          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-card border-b border-border">
                <tr>
                  <th className="px-3 py-2 text-left">Deployment ID</th>
                  <th className="px-3 py-2 text-left">Session</th>
                  <th className="px-3 py-2 text-left">Policy</th>
                  <th className="px-3 py-2 text-left">Status</th>
                  <th className="px-3 py-2 text-left">Health</th>
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {deployments.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-8 text-center text-muted-foreground">
                      No deployments found
                    </td>
                  </tr>
                )}
                {deployments.map((d) => (
                  <tr
                    key={d.deployment_id}
                    className="border-b border-border hover:bg-secondary/50 transition-colors"
                  >
                    <td className="px-3 py-2 font-mono text-xs">{d.deployment_id}</td>
                    <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                      {d.session_id}
                    </td>
                    <td className="px-3 py-2">
                      <PolicySummary policy={d.policy} />
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={d.status} />
                    </td>
                    <td className="px-3 py-2 text-muted-foreground text-xs">
                      {d.health?.sandbox ? String(d.health.sandbox) : "--"}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <ActionButton
                        icon={<Square size={12} />}
                        title="Stop"
                        onClick={() => handleStopDeployment(d.deployment_id)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Governance tab */}
      {tab === "governance" && (
        <div className="space-y-6">
          {/* Policy Summary Card */}
          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
              <ShieldCheck size={16} />
              Policy Summary
            </h3>
            <div className="flex items-center gap-4">
              {sessions.some(
                (s) =>
                  s.metadata?.policy === "restricted" ||
                  s.metadata?.restricted === true,
              ) ? (
                <div className="flex items-center gap-2 text-sm">
                  <Lock size={14} className="text-yellow-500" />
                  <span className="text-xs px-2 py-0.5 rounded-full border bg-yellow-500/10 text-yellow-500 border-yellow-500/30">
                    Restricted
                  </span>
                  <span className="text-muted-foreground text-xs">
                    Sessions require approval before execution
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm">
                  <Unlock size={14} className="text-green-500" />
                  <span className="text-xs px-2 py-0.5 rounded-full border bg-green-500/10 text-green-500 border-green-500/30">
                    Open
                  </span>
                  <span className="text-muted-foreground text-xs">
                    Sessions run without manual approval
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Pending Approvals */}
          <div>
            <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
              <Clock size={16} />
              Pending Approvals ({approvals.length})
            </h3>
            <div className="border border-border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-card border-b border-border">
                  <tr>
                    <th className="px-3 py-2 text-left">Session ID</th>
                    <th className="px-3 py-2 text-left">Agent</th>
                    <th className="px-3 py-2 text-left">Status</th>
                    <th className="px-3 py-2 text-left">Requested</th>
                    <th className="px-3 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {governanceLoading && (
                    <tr>
                      <td
                        colSpan={5}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        Loading...
                      </td>
                    </tr>
                  )}
                  {!governanceLoading && approvals.length === 0 && (
                    <tr>
                      <td
                        colSpan={5}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        No pending approvals
                      </td>
                    </tr>
                  )}
                  {approvals.map((a) => (
                    <tr
                      key={a.id}
                      className="border-b border-border hover:bg-secondary/50 transition-colors"
                    >
                      <td className="px-3 py-2 font-mono text-xs">
                        {a.session_id}
                      </td>
                      <td className="px-3 py-2">{a.agent_name}</td>
                      <td className="px-3 py-2">
                        <ApprovalBadge status={a.status} />
                      </td>
                      <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                        {new Date(a.requested_at).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <div className="flex items-center justify-end gap-1">
                          <button
                            onClick={() =>
                              handleReviewApproval(a.id, "approved")
                            }
                            title="Approve"
                            className="p-1.5 rounded hover:bg-green-500/20 transition-colors text-green-500"
                          >
                            <Check size={14} />
                          </button>
                          <button
                            onClick={() =>
                              handleReviewApproval(a.id, "rejected")
                            }
                            title="Reject"
                            className="p-1.5 rounded hover:bg-red-500/20 transition-colors text-red-500"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Audit Trail */}
          <div>
            <h3 className="text-sm font-medium flex items-center gap-2 mb-3">
              <FileText size={16} />
              Audit Trail
            </h3>
            <div className="border border-border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-card border-b border-border">
                  <tr>
                    <th className="px-3 py-2 text-left">Timestamp</th>
                    <th className="px-3 py-2 text-left">Event Type</th>
                    <th className="px-3 py-2 text-left">Session ID</th>
                    <th className="px-3 py-2 text-left">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {governanceLoading && (
                    <tr>
                      <td
                        colSpan={4}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        Loading...
                      </td>
                    </tr>
                  )}
                  {!governanceLoading && auditEvents.length === 0 && (
                    <tr>
                      <td
                        colSpan={4}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        No audit events found
                      </td>
                    </tr>
                  )}
                  {auditEvents.map((e) => (
                    <tr
                      key={e.id}
                      className="border-b border-border hover:bg-secondary/50 transition-colors"
                    >
                      <td className="px-3 py-2 text-muted-foreground whitespace-nowrap text-xs">
                        {new Date(e.timestamp).toLocaleString()}
                      </td>
                      <td className="px-3 py-2">
                        <AuditEventBadge type={e.event_type} />
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {e.session_id}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground text-xs max-w-[300px] truncate">
                        {e.details}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Create Session Modal */}
      {showCreateSession && (
        <Modal title="Create Session" onClose={() => setShowCreateSession(false)}>
          <div className="space-y-3">
            <FormInput
              label="Agent Name"
              value={sessionForm.name}
              onChange={(v) => setSessionForm({ ...sessionForm, name: v })}
              placeholder="my-agent"
            />
            <FormInput
              label="Model"
              value={sessionForm.model}
              onChange={(v) => setSessionForm({ ...sessionForm, model: v })}
              placeholder="llama-3"
            />
            <FormInput
              label="Tools (comma-separated)"
              value={sessionForm.tools}
              onChange={(v) => setSessionForm({ ...sessionForm, tools: v })}
              placeholder="search, calculator"
            />
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">System Prompt</span>
              <textarea
                value={sessionForm.system_prompt}
                onChange={(e) => setSessionForm({ ...sessionForm, system_prompt: e.target.value })}
                className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none h-20 resize-none"
                placeholder="You are a helpful assistant..."
              />
            </div>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <button
              onClick={() => setShowCreateSession(false)}
              className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleCreateSession}
              className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
            >
              Create
            </button>
          </div>
        </Modal>
      )}

      {/* Create Deployment Modal */}
      {showCreateDeployment && (
        <Modal title="New Deployment" onClose={() => setShowCreateDeployment(false)}>
          <div className="space-y-3">
            <FormInput
              label="Agent Name"
              value={deployForm.name}
              onChange={(v) => setDeployForm({ ...deployForm, name: v })}
              placeholder="sandbox-agent"
            />
            <FormInput
              label="Model"
              value={deployForm.model}
              onChange={(v) => setDeployForm({ ...deployForm, model: v })}
              placeholder="llama-3"
            />
            <FormInput
              label="Tools (comma-separated)"
              value={deployForm.tools}
              onChange={(v) => setDeployForm({ ...deployForm, tools: v })}
              placeholder="search, calculator"
            />
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">System Prompt</span>
              <textarea
                value={deployForm.system_prompt}
                onChange={(e) => setDeployForm({ ...deployForm, system_prompt: e.target.value })}
                className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none h-20 resize-none"
                placeholder="You are a helpful assistant..."
              />
            </div>

            <div className="border-t border-border pt-3">
              <h4 className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
                <ShieldCheck size={12} /> Sandbox Policy
              </h4>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={deployForm.allow_network}
                    onChange={(e) => setDeployForm({ ...deployForm, allow_network: e.target.checked })}
                    className="rounded"
                  />
                  Network Access
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={deployForm.allow_file_write}
                    onChange={(e) => setDeployForm({ ...deployForm, allow_file_write: e.target.checked })}
                    className="rounded"
                  />
                  File Write
                </label>
                <FormInput
                  label="Memory (MB)"
                  type="number"
                  value={String(deployForm.max_memory_mb)}
                  onChange={(v) => setDeployForm({ ...deployForm, max_memory_mb: Number(v) })}
                />
                <FormInput
                  label="CPU (seconds)"
                  type="number"
                  value={String(deployForm.max_cpu_seconds)}
                  onChange={(v) => setDeployForm({ ...deployForm, max_cpu_seconds: Number(v) })}
                />
                <FormInput
                  label="Max Tokens"
                  type="number"
                  value={String(deployForm.max_tokens)}
                  onChange={(v) => setDeployForm({ ...deployForm, max_tokens: Number(v) })}
                />
              </div>
            </div>
          </div>
          <div className="flex justify-end gap-2 mt-4">
            <button
              onClick={() => setShowCreateDeployment(false)}
              className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleCreateDeployment}
              className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
            >
              Deploy
            </button>
          </div>
        </Modal>
      )}
    </div>
  )
}

/* ── Sub-components ──────────────────────────────────────────── */

function HealthIndicator({
  label,
  healthy,
  detail,
}: {
  label: string
  healthy: boolean
  detail: string
}) {
  return (
    <div className="flex items-center gap-2 bg-card border border-border rounded-lg px-4 py-2">
      {healthy ? (
        <CheckCircle2 size={16} className="text-green-500" />
      ) : (
        <AlertCircle size={16} className="text-red-500" />
      )}
      <div>
        <span className="text-sm font-medium">{label}</span>
        {detail && (
          <span className="text-xs text-muted-foreground ml-2">{detail}</span>
        )}
      </div>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
        active
          ? "border-primary text-primary"
          : "border-transparent text-muted-foreground hover:text-foreground"
      }`}
    >
      {children}
    </button>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    created: "bg-blue-500/10 text-blue-500 border-blue-500/30",
    running: "bg-green-500/10 text-green-500 border-green-500/30",
    completed: "bg-gray-500/10 text-gray-400 border-gray-500/30",
    failed: "bg-red-500/10 text-red-500 border-red-500/30",
    stopped: "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
    provisioning: "bg-purple-500/10 text-purple-500 border-purple-500/30",
  }

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full border ${colors[status] || "bg-gray-500/10 text-gray-400 border-gray-500/30"}`}
    >
      {status}
    </span>
  )
}

function PolicySummary({ policy }: { policy: SandboxPolicy }) {
  const parts: string[] = []
  if (policy.allow_network) parts.push("net")
  if (policy.allow_file_write) parts.push("fs")
  parts.push(`${policy.max_memory_mb}MB`)
  parts.push(`${policy.max_tokens}tok`)

  return (
    <span className="text-xs text-muted-foreground">{parts.join(" / ")}</span>
  )
}

function ActionButton({
  icon,
  title,
  onClick,
}: {
  icon: React.ReactNode
  title: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="p-1.5 rounded hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
    >
      {icon}
    </button>
  )
}

function SessionDetail({
  session,
  trace,
  onIngest,
}: {
  session: Session
  trace: Record<string, unknown>[] | null
  onIngest: () => void
}) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-xs text-muted-foreground">Agent</span>
          <p className="font-medium">{session.agent_name}</p>
        </div>
        <div>
          <span className="text-xs text-muted-foreground">Model</span>
          <p className="font-medium">{session.model}</p>
        </div>
        <div>
          <span className="text-xs text-muted-foreground">Tools</span>
          <p className="font-medium">{session.tools.join(", ") || "none"}</p>
        </div>
      </div>

      {trace && trace.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            Execution Trace ({trace.length} steps)
          </h4>
          <div className="space-y-1">
            {trace.map((step, i) => (
              <div
                key={i}
                className="border border-border rounded-md px-3 py-1.5 text-xs bg-input"
              >
                <span className="font-medium">
                  {i + 1}. {String(step.type || "step")}
                  {step.tool ? ` (${String(step.tool)})` : ""}
                </span>
                {(step.content || step.result) && (
                  <pre className="mt-1 whitespace-pre-wrap text-muted-foreground max-h-16 overflow-y-auto font-sans">
                    {String(step.content || step.result || "").slice(0, 300)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={onIngest}
        className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
      >
        <Download size={14} /> Ingest to TraceStore
      </button>
    </div>
  )
}

function Modal({
  title,
  onClose,
  children,
}: {
  title: string
  onClose: () => void
  children: React.ReactNode
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-lg p-6 w-[500px] max-h-[80vh] overflow-y-auto space-y-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        {children}
      </div>
    </div>
  )
}

function FormInput({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  type?: string
  placeholder?: string
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
      />
    </div>
  )
}

function ApprovalBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
    approved: "bg-green-500/10 text-green-500 border-green-500/30",
    rejected: "bg-red-500/10 text-red-500 border-red-500/30",
  }

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full border ${colors[status] || "bg-gray-500/10 text-gray-400 border-gray-500/30"}`}
    >
      {status}
    </span>
  )
}

function AuditEventBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    "session.created": "bg-blue-500/10 text-blue-500 border-blue-500/30",
    "session.started": "bg-green-500/10 text-green-500 border-green-500/30",
    "session.stopped": "bg-yellow-500/10 text-yellow-500 border-yellow-500/30",
    "session.failed": "bg-red-500/10 text-red-500 border-red-500/30",
    "approval.requested": "bg-purple-500/10 text-purple-500 border-purple-500/30",
    "approval.granted": "bg-green-500/10 text-green-500 border-green-500/30",
    "approval.rejected": "bg-red-500/10 text-red-500 border-red-500/30",
  }

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full border ${colors[type] || "bg-gray-500/10 text-gray-400 border-gray-500/30"}`}
    >
      {type}
    </span>
  )
}
