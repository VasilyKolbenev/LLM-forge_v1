import { useState, useEffect, useCallback } from "react"
import {
  ClipboardCheck,
  ChevronDown,
  ChevronRight,
  Play,
  ArrowUpDown,
  Check,
  X,
  Clock,
  Coins,
  Zap,
  Wrench,
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { api } from "@/api/client"

// ── Types ────────────────────────────────────────────────────────

interface CaseResult {
  case_id: string
  query: string
  response: string
  trace: Record<string, unknown>[]
  success: boolean
  score: number
  latency_ms: number
  tokens_used: number
  cost: number
  tools_used: string[]
  tools_match: boolean
  error: string | null
}

interface Report {
  id: string
  suite_name: string
  model_name: string
  timestamp: string
  success_rate: number
  avg_score: number
  avg_latency_ms: number
  total_tokens: number
  total_cost: number
  tools_accuracy: number
  results_json: CaseResult[]
  by_tag_json: Record<string, { count: number; success_rate: number; avg_score: number; avg_latency_ms: number }>
}

interface Suite {
  path: string
  name: string
  description: string
  num_cases: number
  version: string
}

interface Comparison {
  winner: string
  report_a: string
  report_b: string
  model_a: string
  model_b: string
  success_delta: number
  score_delta: number
  latency_delta: number
  cost_delta: number
}

// ── Main Component ───────────────────────────────────────────────

export function AgentEval() {
  const [reports, setReports] = useState<Report[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [expandedCase, setExpandedCase] = useState<string | null>(null)

  // Compare mode
  const [compareMode, setCompareMode] = useState(false)
  const [compareA, setCompareA] = useState<string | null>(null)
  const [compareB, setCompareB] = useState<string | null>(null)
  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [comparing, setComparing] = useState(false)

  // Run eval modal
  const [showRunModal, setShowRunModal] = useState(false)
  const [suites, setSuites] = useState<Suite[]>([])
  const [selectedSuite, setSelectedSuite] = useState("")
  const [scoring, setScoring] = useState<"exact" | "contains" | "judge">("exact")
  const [agentConfigText, setAgentConfigText] = useState("{}")
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  const fetchReports = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getAgentEvalReports()
      setReports(data.reports)
    } catch {
      /* ignore */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchReports()
  }, [fetchReports])

  const handleExpand = async (reportId: string) => {
    if (expandedId === reportId) {
      setExpandedId(null)
      setExpandedCase(null)
      return
    }
    try {
      const full = await api.getAgentEvalReport(reportId)
      const idx = reports.findIndex((r) => r.id === reportId)
      if (idx >= 0) {
        const updated = [...reports]
        updated[idx] = full as Report
        setReports(updated)
      }
      setExpandedId(reportId)
      setExpandedCase(null)
    } catch {
      /* ignore */
    }
  }

  const handleCompareSelect = (reportId: string) => {
    if (!compareMode) return
    if (compareA === reportId) {
      setCompareA(null)
      setComparison(null)
      return
    }
    if (compareB === reportId) {
      setCompareB(null)
      setComparison(null)
      return
    }
    if (!compareA) {
      setCompareA(reportId)
    } else if (!compareB) {
      setCompareB(reportId)
    }
  }

  const runComparison = async () => {
    if (!compareA || !compareB) return
    setComparing(true)
    try {
      const result = await api.compareAgentEvalReports(compareA, compareB)
      setComparison(result as Comparison)
    } catch {
      /* ignore */
    } finally {
      setComparing(false)
    }
  }

  useEffect(() => {
    if (compareA && compareB) {
      runComparison()
    }
  }, [compareA, compareB]) // eslint-disable-line react-hooks/exhaustive-deps

  const openRunModal = async () => {
    setShowRunModal(true)
    setRunError(null)
    try {
      const data = await api.getAgentEvalSuites()
      setSuites(data)
      if (data.length > 0 && !selectedSuite) {
        setSelectedSuite(data[0].path)
      }
    } catch {
      /* ignore */
    }
  }

  const handleRunEval = async () => {
    setRunning(true)
    setRunError(null)
    try {
      let config: Record<string, unknown>
      try {
        config = JSON.parse(agentConfigText)
      } catch {
        setRunError("Invalid JSON in agent config")
        setRunning(false)
        return
      }
      await api.runAgentEval(selectedSuite, config, scoring)
      setShowRunModal(false)
      fetchReports()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Run failed")
    } finally {
      setRunning(false)
    }
  }

  const successRateColor = (rate: number) => {
    if (rate >= 0.8) return "text-green-400"
    if (rate >= 0.5) return "text-yellow-400"
    return "text-red-400"
  }

  const successRateBarColor = (rate: number) => {
    if (rate >= 0.8) return "bg-green-500"
    if (rate >= 0.5) return "bg-yellow-500"
    return "bg-red-500"
  }

  const truncate = (s: string, n: number) =>
    s.length > n ? s.slice(0, n) + "..." : s

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <ClipboardCheck size={24} /> Agent Eval
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Run evaluation suites, review reports, and compare results
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => {
              setCompareMode(!compareMode)
              setCompareA(null)
              setCompareB(null)
              setComparison(null)
            }}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              compareMode
                ? "border-primary bg-primary/10 text-primary"
                : "border-border hover:bg-secondary text-muted-foreground"
            }`}
          >
            <ArrowUpDown size={14} className="inline mr-1" />
            Compare
          </button>
          <button
            onClick={openRunModal}
            className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
          >
            <Play size={14} className="inline mr-1" />
            Run Eval
          </button>
        </div>
      </div>

      {/* Compare panel */}
      <AnimatePresence>
        {compareMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="bg-card border border-border rounded-lg p-4 space-y-3">
              <p className="text-sm text-muted-foreground">
                Select two reports to compare. Click rows to select.
              </p>
              <div className="flex gap-4 text-sm">
                <span>
                  A:{" "}
                  {compareA ? (
                    <span className="text-primary font-mono">{compareA}</span>
                  ) : (
                    <span className="text-muted-foreground">--</span>
                  )}
                </span>
                <span>
                  B:{" "}
                  {compareB ? (
                    <span className="text-primary font-mono">{compareB}</span>
                  ) : (
                    <span className="text-muted-foreground">--</span>
                  )}
                </span>
              </div>
              {comparing && (
                <p className="text-sm text-muted-foreground">Comparing...</p>
              )}
              {comparison && <ComparisonPanel comparison={comparison} />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Reports table */}
      <div className="border border-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-card border-b border-border">
            <tr>
              {compareMode && <th className="px-3 py-2 w-8" />}
              <th className="px-3 py-2 text-left">Suite</th>
              <th className="px-3 py-2 text-left">Model</th>
              <th className="px-3 py-2 text-left">Date</th>
              <th className="px-3 py-2 text-left">Success Rate</th>
              <th className="px-3 py-2 text-right">Avg Score</th>
              <th className="px-3 py-2 text-right">Latency</th>
              <th className="px-3 py-2 text-right">Cost</th>
              <th className="px-3 py-2 text-right">Tokens</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td
                  colSpan={compareMode ? 9 : 8}
                  className="px-3 py-8 text-center text-muted-foreground"
                >
                  Loading...
                </td>
              </tr>
            )}
            {!loading && reports.length === 0 && (
              <tr>
                <td
                  colSpan={compareMode ? 9 : 8}
                  className="px-3 py-8 text-center text-muted-foreground"
                >
                  No evaluation reports yet. Run an eval to get started.
                </td>
              </tr>
            )}
            {reports.map((r) => (
              <ReportRow
                key={r.id}
                report={r}
                expanded={expandedId === r.id}
                expandedCase={expandedCase}
                onExpand={handleExpand}
                onExpandCase={setExpandedCase}
                compareMode={compareMode}
                isCompareSelected={compareA === r.id || compareB === r.id}
                onCompareSelect={handleCompareSelect}
                successRateColor={successRateColor}
                successRateBarColor={successRateBarColor}
                truncate={truncate}
                colSpan={compareMode ? 9 : 8}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Run Eval Modal */}
      {showRunModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-card border border-border rounded-lg p-6 w-[500px] space-y-4"
          >
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Play size={18} /> Run Evaluation
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-muted-foreground">Suite</label>
                <select
                  value={selectedSuite}
                  onChange={(e) => setSelectedSuite(e.target.value)}
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                >
                  {suites.map((s) => (
                    <option key={s.path} value={s.path}>
                      {s.name} ({s.num_cases} cases)
                    </option>
                  ))}
                  {suites.length === 0 && (
                    <option value="">No suites found</option>
                  )}
                </select>
              </div>
              <div>
                <label className="text-sm text-muted-foreground">
                  Scoring Mode
                </label>
                <select
                  value={scoring}
                  onChange={(e) =>
                    setScoring(e.target.value as "exact" | "contains" | "judge")
                  }
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                >
                  <option value="exact">Exact Match</option>
                  <option value="contains">Contains</option>
                  <option value="judge">LLM Judge</option>
                </select>
              </div>
              <div>
                <label className="text-sm text-muted-foreground">
                  Agent Config (JSON)
                </label>
                <textarea
                  value={agentConfigText}
                  onChange={(e) => setAgentConfigText(e.target.value)}
                  rows={6}
                  className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none font-mono"
                />
              </div>
            </div>
            {runError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-md p-3 text-sm text-red-400">
                {runError}
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowRunModal(false)}
                className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleRunEval}
                disabled={running || !selectedSuite}
                className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground disabled:opacity-50"
              >
                {running ? "Running..." : "Run Eval"}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}

// ── Report Row ───────────────────────────────────────────────────

function ReportRow({
  report,
  expanded,
  expandedCase,
  onExpand,
  onExpandCase,
  compareMode,
  isCompareSelected,
  onCompareSelect,
  successRateColor,
  successRateBarColor,
  truncate,
  colSpan,
}: {
  report: Report
  expanded: boolean
  expandedCase: string | null
  onExpand: (id: string) => void
  onExpandCase: (id: string | null) => void
  compareMode: boolean
  isCompareSelected: boolean
  onCompareSelect: (id: string) => void
  successRateColor: (r: number) => string
  successRateBarColor: (r: number) => string
  truncate: (s: string, n: number) => string
  colSpan: number
}) {
  return (
    <>
      <tr
        onClick={() =>
          compareMode ? onCompareSelect(report.id) : onExpand(report.id)
        }
        className={`border-b border-border hover:bg-secondary/50 cursor-pointer transition-colors ${
          isCompareSelected ? "bg-primary/5" : ""
        }`}
      >
        {compareMode && (
          <td className="px-3 py-2">
            <input
              type="checkbox"
              checked={isCompareSelected}
              readOnly
              className="rounded"
            />
          </td>
        )}
        <td className="px-3 py-2">
          {expanded ? (
            <ChevronDown size={14} className="inline mr-1" />
          ) : (
            <ChevronRight size={14} className="inline mr-1" />
          )}
          {report.suite_name}
        </td>
        <td className="px-3 py-2 text-muted-foreground">
          {report.model_name}
        </td>
        <td className="px-3 py-2 text-muted-foreground whitespace-nowrap">
          {new Date(report.timestamp).toLocaleString()}
        </td>
        <td className="px-3 py-2">
          <div className="flex items-center gap-2">
            <div className="w-20 h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${successRateBarColor(report.success_rate)}`}
                style={{ width: `${report.success_rate * 100}%` }}
              />
            </div>
            <span className={`text-xs font-medium ${successRateColor(report.success_rate)}`}>
              {(report.success_rate * 100).toFixed(0)}%
            </span>
          </div>
        </td>
        <td className="px-3 py-2 text-right text-muted-foreground">
          {report.avg_score.toFixed(2)}
        </td>
        <td className="px-3 py-2 text-right text-muted-foreground">
          {report.avg_latency_ms.toFixed(0)}ms
        </td>
        <td className="px-3 py-2 text-right text-muted-foreground">
          ${report.total_cost.toFixed(4)}
        </td>
        <td className="px-3 py-2 text-right text-muted-foreground">
          {report.total_tokens}
        </td>
      </tr>
      <AnimatePresence>
        {expanded && report.results_json && (
          <tr>
            <td colSpan={colSpan} className="bg-card/50 px-0 py-0">
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <ReportDetail
                  report={report}
                  expandedCase={expandedCase}
                  onExpandCase={onExpandCase}
                  truncate={truncate}
                />
              </motion.div>
            </td>
          </tr>
        )}
      </AnimatePresence>
    </>
  )
}

// ── Report Detail ────────────────────────────────────────────────

function ReportDetail({
  report,
  expandedCase,
  onExpandCase,
  truncate,
}: {
  report: Report
  expandedCase: string | null
  onExpandCase: (id: string | null) => void
  truncate: (s: string, n: number) => string
}) {
  const cases = report.results_json || []

  return (
    <div className="px-6 py-4 space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3">
        <MiniStat
          icon={<Check size={14} />}
          label="Success Rate"
          value={`${(report.success_rate * 100).toFixed(1)}%`}
        />
        <MiniStat
          icon={<Zap size={14} />}
          label="Avg Score"
          value={report.avg_score.toFixed(4)}
        />
        <MiniStat
          icon={<Clock size={14} />}
          label="Avg Latency"
          value={`${report.avg_latency_ms.toFixed(0)}ms`}
        />
        <MiniStat
          icon={<Wrench size={14} />}
          label="Tools Accuracy"
          value={`${(report.tools_accuracy * 100).toFixed(1)}%`}
        />
      </div>

      {/* By-tag breakdown */}
      {report.by_tag_json && Object.keys(report.by_tag_json).length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2">
            Per-Tag Breakdown
          </h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(report.by_tag_json).map(([tag, metrics]) => (
              <span
                key={tag}
                className="text-xs px-2 py-1 rounded bg-secondary border border-border"
              >
                <span className="font-medium">{tag}</span>:{" "}
                {(metrics.success_rate * 100).toFixed(0)}% ({metrics.count})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Cases table */}
      <div className="border border-border rounded-md overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-card border-b border-border">
            <tr>
              <th className="px-3 py-1.5 text-left">Case ID</th>
              <th className="px-3 py-1.5 text-left">Query</th>
              <th className="px-3 py-1.5 text-center">Result</th>
              <th className="px-3 py-1.5 text-right">Score</th>
              <th className="px-3 py-1.5 text-right">Latency</th>
              <th className="px-3 py-1.5 text-center">Tools</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <CaseRow
                key={c.case_id}
                caseResult={c}
                expanded={expandedCase === c.case_id}
                onToggle={() =>
                  onExpandCase(
                    expandedCase === c.case_id ? null : c.case_id
                  )
                }
                truncate={truncate}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Case Row ─────────────────────────────────────────────────────

function CaseRow({
  caseResult,
  expanded,
  onToggle,
  truncate,
}: {
  caseResult: CaseResult
  expanded: boolean
  onToggle: () => void
  truncate: (s: string, n: number) => string
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className="border-b border-border hover:bg-secondary/30 cursor-pointer transition-colors"
      >
        <td className="px-3 py-1.5 font-mono">
          {expanded ? (
            <ChevronDown size={12} className="inline mr-1" />
          ) : (
            <ChevronRight size={12} className="inline mr-1" />
          )}
          {caseResult.case_id}
        </td>
        <td className="px-3 py-1.5 max-w-[200px] truncate text-muted-foreground">
          {truncate(caseResult.query, 50)}
        </td>
        <td className="px-3 py-1.5 text-center">
          {caseResult.success ? (
            <span className="inline-flex items-center gap-1 text-green-400 bg-green-500/10 border border-green-500/30 px-1.5 py-0.5 rounded-full text-[10px]">
              <Check size={10} /> PASS
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-red-400 bg-red-500/10 border border-red-500/30 px-1.5 py-0.5 rounded-full text-[10px]">
              <X size={10} /> FAIL
            </span>
          )}
        </td>
        <td className="px-3 py-1.5 text-right text-muted-foreground">
          {caseResult.score.toFixed(2)}
        </td>
        <td className="px-3 py-1.5 text-right text-muted-foreground">
          {caseResult.latency_ms}ms
        </td>
        <td className="px-3 py-1.5 text-center">
          {caseResult.tools_match ? (
            <Check size={12} className="inline text-green-400" />
          ) : (
            <X size={12} className="inline text-red-400" />
          )}
        </td>
      </tr>
      <AnimatePresence>
        {expanded && (
          <tr>
            <td colSpan={6} className="bg-card/30 px-0 py-0">
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <CaseDetail caseResult={caseResult} />
              </motion.div>
            </td>
          </tr>
        )}
      </AnimatePresence>
    </>
  )
}

// ── Case Detail ──────────────────────────────────────────────────

function CaseDetail({ caseResult }: { caseResult: CaseResult }) {
  return (
    <div className="px-6 py-3 space-y-3">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-1">
            Query
          </h5>
          <pre className="text-xs whitespace-pre-wrap bg-input rounded-md p-2 border border-border font-sans max-h-32 overflow-y-auto">
            {caseResult.query}
          </pre>
        </div>
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-1">
            Response
          </h5>
          <pre className="text-xs whitespace-pre-wrap bg-input rounded-md p-2 border border-border font-sans max-h-32 overflow-y-auto">
            {caseResult.response || "(empty)"}
          </pre>
        </div>
      </div>

      {caseResult.tools_used.length > 0 && (
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-1">
            Tools Used
          </h5>
          <div className="flex flex-wrap gap-1">
            {caseResult.tools_used.map((tool, i) => (
              <span
                key={i}
                className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/10 border border-yellow-500/30 text-yellow-400"
              >
                {tool}
              </span>
            ))}
          </div>
        </div>
      )}

      {caseResult.error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-md p-2 text-xs text-red-400">
          {caseResult.error}
        </div>
      )}

      {caseResult.trace && caseResult.trace.length > 0 && (
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-1">
            Trace
          </h5>
          <div className="space-y-1">
            {caseResult.trace.map((step, i) => {
              const type = String((step as Record<string, unknown>).type || "unknown")
              const colors: Record<string, string> = {
                llm_response: "border-blue-500/30 bg-blue-500/5",
                tool_call: "border-yellow-500/30 bg-yellow-500/5",
                observation: "border-green-500/30 bg-green-500/5",
                answer: "border-primary/30 bg-primary/5",
              }
              return (
                <div
                  key={i}
                  className={`border rounded-md px-2 py-1 text-[10px] ${colors[type] || "border-border"}`}
                >
                  <span className="font-medium">
                    {i + 1}. {type}
                    {(step as Record<string, unknown>).tool
                      ? ` (${(step as Record<string, unknown>).tool})`
                      : ""}
                  </span>
                  <pre className="mt-0.5 whitespace-pre-wrap font-sans text-muted-foreground max-h-16 overflow-y-auto">
                    {String(
                      (step as Record<string, unknown>).content ||
                        (step as Record<string, unknown>).result ||
                        ""
                    ).slice(0, 300)}
                  </pre>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <div className="flex gap-4 text-[10px] text-muted-foreground">
        <span>
          <Coins size={10} className="inline mr-0.5" />
          ${caseResult.cost.toFixed(4)}
        </span>
        <span>Tokens: {caseResult.tokens_used}</span>
      </div>
    </div>
  )
}

// ── Comparison Panel ─────────────────────────────────────────────

function ComparisonPanel({ comparison }: { comparison: Comparison }) {
  const deltaDisplay = (
    value: number,
    higherIsBetter: boolean,
    format: (v: number) => string = (v) => v.toFixed(4)
  ) => {
    if (Math.abs(value) < 0.0001) return <span className="text-muted-foreground">--</span>
    const isGood = higherIsBetter ? value > 0 : value < 0
    return (
      <span className={isGood ? "text-green-400" : "text-red-400"}>
        {isGood ? "+" : ""}{format(value)}
      </span>
    )
  }

  return (
    <div className="grid grid-cols-5 gap-3 text-sm">
      <div className="bg-secondary/50 rounded-md p-3 text-center">
        <p className="text-xs text-muted-foreground mb-1">Winner</p>
        <p className="text-lg font-bold text-primary">
          {comparison.winner === "tie" ? "Tie" : comparison.winner}
        </p>
      </div>
      <div className="bg-secondary/50 rounded-md p-3 text-center">
        <p className="text-xs text-muted-foreground mb-1">Success Rate</p>
        <p className="text-lg font-bold">
          {deltaDisplay(comparison.success_delta, true, (v) =>
            `${(v * 100).toFixed(1)}%`
          )}
        </p>
      </div>
      <div className="bg-secondary/50 rounded-md p-3 text-center">
        <p className="text-xs text-muted-foreground mb-1">Score</p>
        <p className="text-lg font-bold">
          {deltaDisplay(comparison.score_delta, true)}
        </p>
      </div>
      <div className="bg-secondary/50 rounded-md p-3 text-center">
        <p className="text-xs text-muted-foreground mb-1">Latency</p>
        <p className="text-lg font-bold">
          {deltaDisplay(comparison.latency_delta, false, (v) =>
            `${v.toFixed(0)}ms`
          )}
        </p>
      </div>
      <div className="bg-secondary/50 rounded-md p-3 text-center">
        <p className="text-xs text-muted-foreground mb-1">Cost</p>
        <p className="text-lg font-bold">
          {deltaDisplay(comparison.cost_delta, false, (v) =>
            `$${v.toFixed(4)}`
          )}
        </p>
      </div>
    </div>
  )
}

// ── Mini Stat Card ───────────────────────────────────────────────

function MiniStat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <div className="bg-secondary/30 border border-border rounded-md p-2">
      <p className="text-[10px] text-muted-foreground flex items-center gap-1">
        {icon} {label}
      </p>
      <p className="text-sm font-bold mt-0.5">{value}</p>
    </div>
  )
}
