import { useState, useEffect, useCallback, Fragment } from "react"
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Download,
  MessageSquare,
  AlertCircle,
  ThumbsUp,
  ThumbsDown,
  Clock,
} from "lucide-react"
import { FeedbackButtons } from "@/components/FeedbackButtons"

interface Trace {
  trace_id: string
  agent_id: string
  model_name: string
  user_query: string
  response: string
  trace_json: Record<string, unknown>[]
  status: string
  tokens_used: number
  cost: number
  latency_ms: number
  created_at: string
  avg_rating?: number
  feedback_count?: number
}

interface TraceDetail extends Trace {
  feedback: {
    id: string
    feedback_type: string
    rating: number
    reason: string
    user_id: string
    created_at: string
  }[]
}

interface Stats {
  total: number
  with_feedback: number
  avg_rating: number
  status_counts: Record<string, number>
  traces_per_day: Record<string, number>
}

const PAGE_SIZE = 20

export function Traces() {
  const [traces, setTraces] = useState<Trace[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)

  // Filters
  const [dateFrom, setDateFrom] = useState("")
  const [dateTo, setDateTo] = useState("")
  const [modelFilter, setModelFilter] = useState("")
  const [statusFilter, setStatusFilter] = useState("")
  const [ratingFilter, setRatingFilter] = useState("")

  // Selection
  const [selected, setSelected] = useState<Set<string>>(new Set())

  // Detail expansion
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<TraceDetail | null>(null)

  // Dataset build modal
  const [showBuildModal, setShowBuildModal] = useState(false)
  const [buildFormat, setBuildFormat] = useState<"sft" | "dpo">("sft")
  const [buildName, setBuildName] = useState("traces-dataset")
  const [building, setBuilding] = useState(false)
  const [buildResult, setBuildResult] = useState<{
    path: string
    num_examples: number
  } | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/traces/stats")
      if (res.ok) setStats(await res.json())
    } catch {
      /* ignore */
    }
  }, [])

  const fetchTraces = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (dateFrom) params.set("date_from", dateFrom)
      if (dateTo) params.set("date_to", dateTo)
      if (modelFilter) params.set("model_name", modelFilter)
      if (statusFilter) params.set("status", statusFilter)
      if (ratingFilter) params.set("min_rating", ratingFilter)
      params.set("limit", String(PAGE_SIZE))
      params.set("offset", String(offset))

      const res = await fetch(`/api/v1/traces?${params}`)
      if (res.ok) {
        const data = await res.json()
        setTraces(data.traces)
        setTotal(data.total)
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [dateFrom, dateTo, modelFilter, statusFilter, ratingFilter, offset])

  useEffect(() => {
    fetchStats()
  }, [fetchStats])

  useEffect(() => {
    fetchTraces()
  }, [fetchTraces])

  const loadDetail = async (traceId: string) => {
    if (expandedId === traceId) {
      setExpandedId(null)
      setDetail(null)
      return
    }
    setExpandedId(traceId)
    try {
      const res = await fetch(`/api/v1/traces/${traceId}`)
      if (res.ok) setDetail(await res.json())
    } catch {
      /* ignore */
    }
  }

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleAll = () => {
    if (selected.size === traces.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(traces.map((t) => t.trace_id)))
    }
  }

  const handleBuild = async () => {
    setBuilding(true)
    setBuildResult(null)
    try {
      const res = await fetch("/api/v1/traces/build-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trace_ids: Array.from(selected),
          format: buildFormat,
          name: buildName,
        }),
      })
      if (res.ok) {
        setBuildResult(await res.json())
      }
    } catch {
      /* ignore */
    } finally {
      setBuilding(false)
    }
  }

  const truncate = (s: string, n: number) =>
    s.length > n ? s.slice(0, n) + "..." : s

  const errorRate =
    stats && stats.total > 0
      ? (((stats.status_counts["error"] || 0) / stats.total) * 100).toFixed(1)
      : "0"

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Activity size={24} /> Traces
        </h2>
        <p className="text-muted-foreground text-sm mt-1">
          Agent execution traces, feedback, and dataset export
        </p>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <StatCard label="Total Traces" value={stats.total} />
          <StatCard label="With Feedback" value={stats.with_feedback} />
          <StatCard
            label="Avg Rating"
            value={stats.avg_rating > 0 ? stats.avg_rating.toFixed(2) : "--"}
          />
          <StatCard label="Error Rate" value={`${errorRate}%`} />
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3">
        <FilterInput
          label="From"
          type="date"
          value={dateFrom}
          onChange={setDateFrom}
        />
        <FilterInput
          label="To"
          type="date"
          value={dateTo}
          onChange={setDateTo}
        />
        <FilterInput
          label="Model"
          value={modelFilter}
          onChange={setModelFilter}
          placeholder="e.g. gpt-4"
        />
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">Status</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
          >
            <option value="">All</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
          </select>
        </div>
        <FilterInput
          label="Min Rating"
          type="number"
          value={ratingFilter}
          onChange={setRatingFilter}
          placeholder="0-1"
        />
        <div className="flex gap-2 ml-auto">
          <button
            disabled={selected.size === 0}
            onClick={() => {
              setBuildFormat("sft")
              setShowBuildModal(true)
            }}
            className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground disabled:opacity-50"
          >
            <Download size={14} className="inline mr-1" />
            Build SFT
          </button>
          <button
            disabled={selected.size === 0}
            onClick={() => {
              setBuildFormat("dpo")
              setShowBuildModal(true)
            }}
            className="px-3 py-1.5 text-sm rounded-md bg-secondary text-foreground disabled:opacity-50"
          >
            <Download size={14} className="inline mr-1" />
            Build DPO
          </button>
        </div>
      </div>

      {/* Traces table */}
      <div className="border border-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-card border-b border-border">
            <tr>
              <th className="px-3 py-2 text-left w-8">
                <input
                  type="checkbox"
                  checked={
                    traces.length > 0 && selected.size === traces.length
                  }
                  onChange={toggleAll}
                  className="rounded"
                />
              </th>
              <th className="px-3 py-2 text-left">Time</th>
              <th className="px-3 py-2 text-left">Query</th>
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Rating</th>
              <th className="px-3 py-2 text-right">Latency</th>
              <th className="px-3 py-2 text-right">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={8} className="px-3 py-8 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            )}
            {!loading && traces.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-8 text-center text-muted-foreground">
                  No traces found
                </td>
              </tr>
            )}
            {traces.map((t) => (
              <Fragment key={t.trace_id}>
                <tr
                  onClick={() => loadDetail(t.trace_id)}
                  className="border-b border-border hover:bg-secondary/50 cursor-pointer transition-colors"
                >
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected.has(t.trace_id)}
                      onChange={() => toggleSelect(t.trace_id)}
                      className="rounded"
                    />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                    {expandedId === t.trace_id ? (
                      <ChevronDown size={14} className="inline mr-1" />
                    ) : (
                      <ChevronRight size={14} className="inline mr-1" />
                    )}
                    {new Date(t.created_at).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 max-w-[300px] truncate">
                    {truncate(t.user_query, 60)}
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {t.model_name || "--"}
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={t.status} />
                  </td>
                  <td className="px-3 py-2">
                    <RatingDisplay
                      rating={t.avg_rating}
                      count={t.feedback_count}
                    />
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {t.latency_ms > 0 ? `${t.latency_ms}ms` : "--"}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {t.tokens_used || "--"}
                  </td>
                </tr>
                {expandedId === t.trace_id && detail && (
                  <tr key={`${t.trace_id}-detail`}>
                    <td colSpan={8} className="bg-card/50 px-6 py-4">
                      <TraceDetailPanel
                        detail={detail}
                        onFeedbackSaved={() => {
                          fetchTraces()
                          fetchStats()
                        }}
                      />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          Showing {offset + 1}-{Math.min(offset + PAGE_SIZE, total)} of {total}
        </span>
        <div className="flex gap-2">
          <button
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            className="px-3 py-1 rounded border border-border disabled:opacity-50 hover:bg-secondary"
          >
            Previous
          </button>
          <button
            disabled={offset + PAGE_SIZE >= total}
            onClick={() => setOffset(offset + PAGE_SIZE)}
            className="px-3 py-1 rounded border border-border disabled:opacity-50 hover:bg-secondary"
          >
            Next
          </button>
        </div>
      </div>

      {/* Build Dataset Modal */}
      {showBuildModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 w-[400px] space-y-4">
            <h3 className="text-lg font-semibold">
              Build {buildFormat.toUpperCase()} Dataset
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-muted-foreground">
                  Dataset Name
                </label>
                <input
                  value={buildName}
                  onChange={(e) => setBuildName(e.target.value)}
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">Format</label>
                <select
                  value={buildFormat}
                  onChange={(e) =>
                    setBuildFormat(e.target.value as "sft" | "dpo")
                  }
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                >
                  <option value="sft">SFT (Supervised Fine-Tuning)</option>
                  <option value="dpo">DPO (Direct Preference Optimization)</option>
                </select>
              </div>
              <p className="text-sm text-muted-foreground">
                {selected.size} traces selected
              </p>
            </div>
            {buildResult && (
              <div className="bg-green-500/10 border border-green-500/30 rounded-md p-3 text-sm">
                Created dataset with {buildResult.num_examples} examples at{" "}
                <code className="text-xs">{buildResult.path}</code>
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => {
                  setShowBuildModal(false)
                  setBuildResult(null)
                }}
                className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary"
              >
                Close
              </button>
              {!buildResult && (
                <button
                  onClick={handleBuild}
                  disabled={building}
                  className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground disabled:opacity-50"
                >
                  {building ? "Building..." : "Build Dataset"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Sub-components ──────────────────────────────────────────── */

function StatCard({
  label,
  value,
}: {
  label: string
  value: string | number
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  )
}

function FilterInput({
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
        className="bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none w-36"
      />
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors =
    status === "success"
      ? "bg-green-500/10 text-green-500 border-green-500/30"
      : status === "error"
        ? "bg-red-500/10 text-red-500 border-red-500/30"
        : "bg-yellow-500/10 text-yellow-500 border-yellow-500/30"

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full border ${colors}`}
    >
      {status}
    </span>
  )
}

function RatingDisplay({
  rating,
  count,
}: {
  rating?: number
  count?: number
}) {
  if (!count || count === 0) {
    return <span className="text-muted-foreground">--</span>
  }
  if (rating !== undefined && rating > 0.5) {
    return <ThumbsUp size={14} className="text-green-500" />
  }
  return <ThumbsDown size={14} className="text-red-500" />
}

function TraceDetailPanel({
  detail,
  onFeedbackSaved,
}: {
  detail: TraceDetail
  onFeedbackSaved: () => void
}) {
  return (
    <div className="space-y-4">
      {/* Query and response */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1">
            <MessageSquare size={12} /> User Query
          </h4>
          <pre className="text-sm whitespace-pre-wrap bg-input rounded-md p-3 border border-border font-sans">
            {detail.user_query}
          </pre>
        </div>
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1">
            Response
          </h4>
          <pre className="text-sm whitespace-pre-wrap bg-input rounded-md p-3 border border-border font-sans max-h-48 overflow-y-auto">
            {detail.response}
          </pre>
        </div>
      </div>

      {/* Trace timeline */}
      {detail.trace_json && detail.trace_json.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
            <Clock size={12} /> Trace Timeline
          </h4>
          <div className="space-y-2">
            {detail.trace_json.map((step, i) => (
              <TraceStep key={i} step={step} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Feedback */}
      <div className="flex items-center gap-4">
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-1">
            Feedback
          </h4>
          <FeedbackButtons
            traceId={detail.trace_id}
            onFeedback={() => onFeedbackSaved()}
          />
        </div>
        {detail.feedback.length > 0 && (
          <div className="flex-1">
            <h4 className="text-xs font-medium text-muted-foreground mb-1">
              History
            </h4>
            <div className="flex flex-wrap gap-2">
              {detail.feedback.map((fb) => (
                <span
                  key={fb.id}
                  className="text-xs px-2 py-0.5 rounded bg-secondary text-muted-foreground"
                >
                  {fb.rating > 0.5 ? "+" : "-"}
                  {fb.reason ? ` ${fb.reason}` : ""}{" "}
                  <span className="opacity-50">
                    {new Date(fb.created_at).toLocaleDateString()}
                  </span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function TraceStep({
  step,
  index,
}: {
  step: Record<string, unknown>
  index: number
}) {
  const type = String(step.type || "unknown")
  const colors: Record<string, string> = {
    llm_response: "border-blue-500/30 bg-blue-500/5",
    tool_call: "border-yellow-500/30 bg-yellow-500/5",
    observation: "border-green-500/30 bg-green-500/5",
    answer: "border-primary/30 bg-primary/5",
  }

  const labels: Record<string, string> = {
    llm_response: "Thought",
    tool_call: "Action",
    observation: "Observation",
    answer: "Final Answer",
  }

  return (
    <div
      className={`border rounded-md px-3 py-2 text-xs ${colors[type] || "border-border"}`}
    >
      <span className="font-medium">
        {index + 1}. {labels[type] || type}
        {step.tool ? ` (${step.tool})` : ""}
      </span>
      <pre className="mt-1 whitespace-pre-wrap font-sans text-muted-foreground max-h-24 overflow-y-auto">
        {String(
          step.content || step.result || step.raw_arguments || JSON.stringify(step.arguments) || ""
        ).slice(0, 500)}
      </pre>
    </div>
  )
}
