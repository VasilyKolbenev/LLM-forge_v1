import { useState, useEffect, useCallback, useMemo } from "react"
import {
  Gauge,
  Play,
  ChevronDown,
  ChevronRight,
  ArrowUpDown,
  Trophy,
  Zap,
  Brain,
  HardDrive,
  BarChart3,
  Trash2,
  ArrowUp,
  ArrowDown,
  Minus,
  X,
  Tag,
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Cell,
} from "recharts"
import { api, type Benchmark, type CompareResult } from "@/api/client"

// ── Types ────────────────────────────────────────────────────────

interface LeaderboardEntry extends Benchmark {
  rank: number
}

type Tab = "table" | "leaderboard" | "compare"
type SortKey =
  | "model_name"
  | "tokens_per_sec"
  | "time_to_first_token_ms"
  | "perplexity"
  | "peak_vram_gb"
  | "timestamp"
  | "estimated_cost_per_1m_tokens"
type SortDir = "asc" | "desc"

// ── Helpers ──────────────────────────────────────────────────────

function formatNumber(n: number, decimals = 1): string {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(decimals) + "B"
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(decimals) + "M"
  if (n >= 1_000) return (n / 1_000).toFixed(decimals) + "K"
  return n.toFixed(decimals)
}

function medalColor(rank: number): string {
  if (rank === 1) return "text-yellow-400"
  if (rank === 2) return "text-gray-300"
  if (rank === 3) return "text-amber-600"
  return "text-muted-foreground"
}

function medalBg(rank: number): string {
  if (rank === 1) return "bg-yellow-500/10 border-yellow-500/30"
  if (rank === 2) return "bg-gray-400/10 border-gray-400/30"
  if (rank === 3) return "bg-amber-600/10 border-amber-600/30"
  return ""
}

function deltaArrow(delta: number, higherIsBetter: boolean) {
  if (Math.abs(delta) < 0.001) {
    return <Minus size={12} className="inline text-muted-foreground" />
  }
  const isGood = higherIsBetter ? delta > 0 : delta < 0
  return isGood ? (
    <ArrowUp size={12} className="inline text-green-400" />
  ) : (
    <ArrowDown size={12} className="inline text-red-400" />
  )
}

function deltaColor(delta: number, higherIsBetter: boolean): string {
  if (Math.abs(delta) < 0.001) return "text-muted-foreground"
  const isGood = higherIsBetter ? delta > 0 : delta < 0
  return isGood ? "text-green-400" : "text-red-400"
}

// ── Main Component ───────────────────────────────────────────────

export function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<Tab>("table")

  // Table state
  const [sortKey, setSortKey] = useState<SortKey>("timestamp")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  // Leaderboard state
  const [leaderboardMetric, setLeaderboardMetric] = useState("tokens_per_sec")
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [leaderboardLoading, setLeaderboardLoading] = useState(false)

  // Compare state
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)

  // Error state
  const [error, setError] = useState<string | null>(null)

  // Run modal
  const [showRunModal, setShowRunModal] = useState(false)
  const [runForm, setRunForm] = useState({
    model_path: "",
    model_name: "",
    gpu_cost: "",
    tags: "",
  })
  const [runLoading, setRunLoading] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  // ── Fetch benchmarks ────────────────────────────────────────────

  const fetchBenchmarks = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.getBenchmarks()
      setBenchmarks(data.benchmarks || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load benchmarks")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchBenchmarks()
  }, [fetchBenchmarks])

  // ── Fetch leaderboard ───────────────────────────────────────────

  const fetchLeaderboard = useCallback(async (metric: string) => {
    setLeaderboardLoading(true)
    setError(null)
    try {
      const order =
        metric === "perplexity" ||
        metric === "eval_loss" ||
        metric === "time_to_first_token_ms" ||
        metric === "peak_vram_gb" ||
        metric === "estimated_cost_per_1m_tokens"
          ? "asc"
          : "desc"
      const data = await api.getBenchmarkLeaderboard(metric, order)
      setLeaderboard(
        (data.leaderboard || []).map((b: Benchmark, i: number) => ({
          ...b,
          rank: i + 1,
        }))
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load leaderboard")
    } finally {
      setLeaderboardLoading(false)
    }
  }, [])

  useEffect(() => {
    if (activeTab === "leaderboard") {
      fetchLeaderboard(leaderboardMetric)
    }
  }, [activeTab, leaderboardMetric])

  // ── Compare ─────────────────────────────────────────────────────

  const runCompare = useCallback(async () => {
    const ids = Array.from(selectedIds)
    if (ids.length < 2) return
    setCompareLoading(true)
    setError(null)
    try {
      const data = await api.compareBenchmarks(ids)
      setCompareResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compare benchmarks")
      setCompareResult(null)
    } finally {
      setCompareLoading(false)
    }
  }, [selectedIds])

  useEffect(() => {
    if (activeTab === "compare" && selectedIds.size >= 2) {
      runCompare()
    }
  }, [activeTab, selectedIds, runCompare])

  // ── Delete ──────────────────────────────────────────────────────

  const handleDelete = async (id: string) => {
    try {
      await api.deleteBenchmark(id)
      setBenchmarks((prev) => prev.filter((b) => b.id !== id))
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete benchmark")
    }
  }

  // ── Run benchmark ───────────────────────────────────────────────

  const handleRun = async () => {
    setRunLoading(true)
    setRunError(null)
    try {
      const payload: {
        model_path: string
        model_name: string
        gpu_cost?: number
        tags?: string[]
      } = {
        model_path: runForm.model_path,
        model_name: runForm.model_name,
      }
      if (runForm.gpu_cost) payload.gpu_cost = parseFloat(runForm.gpu_cost)
      if (runForm.tags)
        payload.tags = runForm.tags.split(",").map((t) => t.trim())
      await api.runBenchmark(payload)
      setShowRunModal(false)
      setRunForm({ model_path: "", model_name: "", gpu_cost: "", tags: "" })
      fetchBenchmarks()
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Run failed")
    } finally {
      setRunLoading(false)
    }
  }

  // ── Sorted benchmarks ──────────────────────────────────────────

  const sorted = useMemo(() => {
    const arr = [...benchmarks]
    arr.sort((a, b) => {
      let va: number | string
      let vb: number | string
      if (sortKey === "model_name") {
        va = a.model_name.toLowerCase()
        vb = b.model_name.toLowerCase()
      } else if (sortKey === "timestamp") {
        va = new Date(a.timestamp).getTime()
        vb = new Date(b.timestamp).getTime()
      } else if (sortKey === "perplexity") {
        va = a.perplexity ?? Infinity
        vb = b.perplexity ?? Infinity
      } else {
        va = a[sortKey]
        vb = b[sortKey]
      }
      if (va < vb) return sortDir === "asc" ? -1 : 1
      if (va > vb) return sortDir === "asc" ? 1 : -1
      return 0
    })
    return arr
  }, [benchmarks, sortKey, sortDir])

  // ── Summary stats ──────────────────────────────────────────────

  const summary = useMemo(() => {
    if (benchmarks.length === 0)
      return {
        bestTokSec: 0,
        lowestPerplexity: null as number | null,
        totalRuns: 0,
        avgPeakVram: 0,
      }
    const tokSecs = benchmarks.map((b) => b.tokens_per_sec)
    const perplexities = benchmarks
      .map((b) => b.perplexity)
      .filter((p): p is number => p !== null)
    const vrams = benchmarks.map((b) => b.peak_vram_gb)
    return {
      bestTokSec: Math.max(...tokSecs),
      lowestPerplexity: perplexities.length > 0 ? Math.min(...perplexities) : null,
      totalRuns: benchmarks.length,
      avgPeakVram: vrams.reduce((a, b) => a + b, 0) / vrams.length,
    }
  }, [benchmarks])

  // ── Sort handler ───────────────────────────────────────────────

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setSortKey(key)
      setSortDir("desc")
    }
  }

  // ── Selection handler ──────────────────────────────────────────

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  // ── Tab bar ────────────────────────────────────────────────────

  const tabs: { key: Tab; label: string }[] = [
    { key: "table", label: "Table" },
    { key: "leaderboard", label: "Leaderboard" },
    { key: "compare", label: "Compare" },
  ]

  // ── Radar chart data for compare ───────────────────────────────

  const radarData = useMemo(() => {
    if (!compareResult) return []
    const all = [compareResult.baseline, ...compareResult.candidates]
    const maxTokSec = Math.max(...all.map((b) => b.tokens_per_sec), 1)
    const maxTTFT = Math.max(...all.map((b) => b.time_to_first_token_ms), 1)
    const maxTrainSps = Math.max(
      ...all.map((b) => b.training_samples_per_sec),
      1
    )
    const maxVram = Math.max(...all.map((b) => b.peak_vram_gb), 1)
    const perplexities = all
      .map((b) => b.perplexity)
      .filter((p): p is number => p !== null)
    const maxPerplexity = perplexities.length > 0 ? Math.max(...perplexities) : 1

    const metrics = [
      { key: "Tokens/sec", extract: (b: Benchmark) => (b.tokens_per_sec / maxTokSec) * 100 },
      {
        key: "TTFT (inv)",
        extract: (b: Benchmark) => ((1 - b.time_to_first_token_ms / maxTTFT) * 100),
      },
      {
        key: "Train Sps",
        extract: (b: Benchmark) => (b.training_samples_per_sec / maxTrainSps) * 100,
      },
      {
        key: "VRAM Eff (inv)",
        extract: (b: Benchmark) => ((1 - b.peak_vram_gb / maxVram) * 100),
      },
      {
        key: "Perplexity (inv)",
        extract: (b: Benchmark) =>
          b.perplexity !== null ? ((1 - b.perplexity / maxPerplexity) * 100) : 0,
      },
    ]

    return metrics.map((m) => {
      const row: Record<string, any> = { metric: m.key }
      all.forEach((b) => {
        row[b.model_name || b.id] = Math.max(0, m.extract(b))
      })
      return row
    })
  }, [compareResult])

  const radarColors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#22c55e"]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Gauge size={24} /> Benchmarks
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Performance benchmarks, leaderboards, and model comparisons
          </p>
        </div>
        <button
          onClick={() => {
            setShowRunModal(true)
            setRunError(null)
          }}
          className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
        >
          <Play size={14} className="inline mr-1" />
          Run Benchmark
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <SummaryCard
          icon={<Zap size={16} className="text-green-400" />}
          label="Best Tokens/sec"
          value={summary.bestTokSec > 0 ? formatNumber(summary.bestTokSec, 1) : "--"}
          color="text-green-400"
        />
        <SummaryCard
          icon={<Brain size={16} className="text-purple-400" />}
          label="Lowest Perplexity"
          value={
            summary.lowestPerplexity !== null
              ? summary.lowestPerplexity.toFixed(2)
              : "--"
          }
          color="text-purple-400"
        />
        <SummaryCard
          icon={<BarChart3 size={16} className="text-blue-400" />}
          label="Total Runs"
          value={String(summary.totalRuns)}
          color="text-blue-400"
        />
        <SummaryCard
          icon={<HardDrive size={16} className="text-orange-400" />}
          label="Avg Peak VRAM"
          value={
            summary.avgPeakVram > 0
              ? summary.avgPeakVram.toFixed(1) + " GB"
              : "--"
          }
          color="text-orange-400"
        />
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-md p-3 flex items-center justify-between">
          <span className="text-sm text-red-400">{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-300"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Tab Bar */}
      <div className="flex gap-1 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === tab.key
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
            {tab.key === "compare" && selectedIds.size > 0 && (
              <span className="ml-1.5 text-xs bg-primary/20 text-primary px-1.5 py-0.5 rounded-full">
                {selectedIds.size}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Table View ──────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {activeTab === "table" && (
          <motion.div
            key="table"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            <div className="border border-border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-card border-b border-border">
                  <tr>
                    <th className="px-3 py-2 w-8">
                      <input
                        type="checkbox"
                        checked={
                          benchmarks.length > 0 &&
                          selectedIds.size === benchmarks.length
                        }
                        onChange={() => {
                          if (selectedIds.size === benchmarks.length) {
                            setSelectedIds(new Set())
                          } else {
                            setSelectedIds(new Set(benchmarks.map((b) => b.id)))
                          }
                        }}
                        className="rounded"
                      />
                    </th>
                    <th className="px-3 py-2 w-6" />
                    <SortableHeader
                      label="Model"
                      sortKey="model_name"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                    />
                    <SortableHeader
                      label="Tok/s"
                      sortKey="tokens_per_sec"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <SortableHeader
                      label="TTFT"
                      sortKey="time_to_first_token_ms"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <SortableHeader
                      label="Perplexity"
                      sortKey="perplexity"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <SortableHeader
                      label="Peak VRAM"
                      sortKey="peak_vram_gb"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <SortableHeader
                      label="Cost/1M tok"
                      sortKey="estimated_cost_per_1m_tokens"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <SortableHeader
                      label="Date"
                      sortKey="timestamp"
                      currentKey={sortKey}
                      dir={sortDir}
                      onClick={handleSort}
                      align="right"
                    />
                    <th className="px-3 py-2 w-8" />
                  </tr>
                </thead>
                <tbody>
                  {loading && (
                    <tr>
                      <td
                        colSpan={10}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        Loading...
                      </td>
                    </tr>
                  )}
                  {!loading && benchmarks.length === 0 && (
                    <tr>
                      <td
                        colSpan={10}
                        className="px-3 py-8 text-center text-muted-foreground"
                      >
                        No benchmarks yet. Run a benchmark to get started.
                      </td>
                    </tr>
                  )}
                  {sorted.map((b) => (
                    <BenchmarkRow
                      key={b.id}
                      benchmark={b}
                      expanded={expandedId === b.id}
                      selected={selectedIds.has(b.id)}
                      onExpand={() =>
                        setExpandedId(expandedId === b.id ? null : b.id)
                      }
                      onSelect={() => toggleSelect(b.id)}
                      onDelete={() => handleDelete(b.id)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* ── Leaderboard View ──────────────────────────────────── */}
        {activeTab === "leaderboard" && (
          <motion.div
            key="leaderboard"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="space-y-4"
          >
            <div className="flex items-center gap-3">
              <label className="text-sm text-muted-foreground">Metric:</label>
              <select
                value={leaderboardMetric}
                onChange={(e) => setLeaderboardMetric(e.target.value)}
                className="bg-input border border-border rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-ring focus:outline-none"
              >
                <option value="tokens_per_sec">Tokens/sec</option>
                <option value="time_to_first_token_ms">
                  Time to First Token (ms)
                </option>
                <option value="training_samples_per_sec">
                  Training Samples/sec
                </option>
                <option value="peak_vram_gb">Peak VRAM (GB)</option>
                <option value="perplexity">Perplexity</option>
                <option value="eval_loss">Eval Loss</option>
                <option value="estimated_cost_per_1m_tokens">
                  Cost per 1M Tokens
                </option>
              </select>
            </div>

            {leaderboardLoading ? (
              <div className="text-center text-muted-foreground py-8">
                Loading leaderboard...
              </div>
            ) : leaderboard.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No data for leaderboard.
              </div>
            ) : (
              <div className="space-y-4">
                {/* Top 3 podium */}
                <div className="grid grid-cols-3 gap-3">
                  {leaderboard.slice(0, 3).map((entry) => (
                    <div
                      key={entry.id}
                      className={`border rounded-lg p-4 text-center ${medalBg(entry.rank)}`}
                    >
                      <Trophy
                        size={20}
                        className={`mx-auto mb-1 ${medalColor(entry.rank)}`}
                      />
                      <p className="text-xs text-muted-foreground">
                        #{entry.rank}
                      </p>
                      <p className="text-sm font-semibold mt-1 truncate">
                        {entry.model_name}
                      </p>
                      <p className={`text-lg font-bold mt-1 ${medalColor(entry.rank)}`}>
                        {(entry as any)[leaderboardMetric] != null
                          ? typeof (entry as any)[leaderboardMetric] === "number"
                            ? (entry as any)[leaderboardMetric].toFixed(2)
                            : String((entry as any)[leaderboardMetric])
                          : "--"}
                      </p>
                      <p className="text-[10px] text-muted-foreground mt-1">
                        {entry.hardware_info.gpu_name} x{entry.hardware_info.num_gpus}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Bar chart */}
                <div className="bg-card border border-border rounded-lg p-4">
                  <ResponsiveContainer width="100%" height={Math.max(200, leaderboard.length * 40)}>
                    <BarChart
                      data={leaderboard.map((e) => ({
                        name:
                          e.model_name.length > 25
                            ? e.model_name.slice(0, 25) + "..."
                            : e.model_name,
                        value: (e as any)[leaderboardMetric] ?? 0,
                        rank: e.rank,
                      }))}
                      layout="vertical"
                      margin={{ left: 10, right: 30, top: 5, bottom: 5 }}
                    >
                      <XAxis type="number" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        width={180}
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "6px",
                          fontSize: "12px",
                        }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {leaderboard.map((entry, i) => (
                          <Cell
                            key={entry.id}
                            fill={
                              entry.rank === 1
                                ? "#eab308"
                                : entry.rank === 2
                                  ? "#9ca3af"
                                  : entry.rank === 3
                                    ? "#d97706"
                                    : "hsl(var(--primary))"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Full rankings list */}
                {leaderboard.length > 3 && (
                  <div className="border border-border rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-card border-b border-border">
                        <tr>
                          <th className="px-3 py-2 text-left w-12">#</th>
                          <th className="px-3 py-2 text-left">Model</th>
                          <th className="px-3 py-2 text-left">GPU</th>
                          <th className="px-3 py-2 text-right">
                            {leaderboardMetric.replace(/_/g, " ")}
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {leaderboard.slice(3).map((entry) => (
                          <tr
                            key={entry.id}
                            className="border-b border-border hover:bg-secondary/50"
                          >
                            <td className="px-3 py-2 text-muted-foreground">
                              {entry.rank}
                            </td>
                            <td className="px-3 py-2">{entry.model_name}</td>
                            <td className="px-3 py-2 text-muted-foreground">
                              {entry.hardware_info.gpu_name}
                            </td>
                            <td className="px-3 py-2 text-right font-mono">
                              {(entry as any)[leaderboardMetric] != null
                                ? typeof (entry as any)[leaderboardMetric] ===
                                  "number"
                                  ? (
                                      (entry as any)[
                                        leaderboardMetric
                                      ] as number
                                    ).toFixed(2)
                                  : String((entry as any)[leaderboardMetric])
                                : "--"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}

        {/* ── Compare View ──────────────────────────────────────── */}
        {activeTab === "compare" && (
          <motion.div
            key="compare"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="space-y-4"
          >
            {selectedIds.size < 2 ? (
              <div className="bg-card border border-border rounded-lg p-6 text-center text-muted-foreground">
                <ArrowUpDown size={24} className="mx-auto mb-2 opacity-50" />
                <p className="text-sm">
                  Select at least 2 benchmarks from the Table view to compare.
                </p>
                <p className="text-xs mt-1">
                  Currently selected: {selectedIds.size}
                </p>
                <button
                  onClick={() => setActiveTab("table")}
                  className="mt-3 px-3 py-1.5 text-sm rounded-md border border-border hover:bg-secondary"
                >
                  Go to Table
                </button>
              </div>
            ) : compareLoading ? (
              <div className="text-center text-muted-foreground py-8">
                Loading comparison...
              </div>
            ) : compareResult ? (
              <div className="space-y-4">
                {/* Side-by-side cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {[
                    compareResult.baseline,
                    ...compareResult.candidates,
                  ].map((b) => {
                    const deltas = compareResult.deltas[b.id] || {}
                    return (
                      <div
                        key={b.id}
                        className={`bg-card border rounded-lg p-4 space-y-3 ${
                          b.is_baseline
                            ? "border-primary/50"
                            : "border-border"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold truncate">
                            {b.model_name}
                          </h4>
                          {b.is_baseline && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30">
                              BASELINE
                            </span>
                          )}
                        </div>
                        <p className="text-[10px] text-muted-foreground truncate">
                          {b.model_path}
                        </p>
                        <div className="space-y-2 text-xs">
                          <CompareMetricRow
                            label="Tokens/sec"
                            value={b.tokens_per_sec.toFixed(1)}
                            delta={deltas.tokens_per_sec}
                            higherIsBetter={true}
                          />
                          <CompareMetricRow
                            label="TTFT"
                            value={b.time_to_first_token_ms.toFixed(1) + "ms"}
                            delta={deltas.time_to_first_token_ms}
                            higherIsBetter={false}
                          />
                          <CompareMetricRow
                            label="Train Sps"
                            value={b.training_samples_per_sec.toFixed(1)}
                            delta={deltas.training_samples_per_sec}
                            higherIsBetter={true}
                          />
                          <CompareMetricRow
                            label="Peak VRAM"
                            value={b.peak_vram_gb.toFixed(1) + " GB"}
                            delta={deltas.peak_vram_gb}
                            higherIsBetter={false}
                          />
                          <CompareMetricRow
                            label="Perplexity"
                            value={
                              b.perplexity !== null
                                ? b.perplexity.toFixed(2)
                                : "--"
                            }
                            delta={deltas.perplexity}
                            higherIsBetter={false}
                          />
                          <CompareMetricRow
                            label="Cost/1M"
                            value={"$" + b.estimated_cost_per_1m_tokens.toFixed(2)}
                            delta={deltas.estimated_cost_per_1m_tokens}
                            higherIsBetter={false}
                          />
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          {b.hardware_info.gpu_name} x
                          {b.hardware_info.num_gpus} ({b.hardware_info.vram_gb}
                          GB)
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Radar Chart */}
                {radarData.length > 0 && (
                  <div className="bg-card border border-border rounded-lg p-4">
                    <h4 className="text-sm font-semibold mb-3">
                      Performance Radar
                    </h4>
                    <ResponsiveContainer width="100%" height={350}>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="hsl(var(--border))" />
                        <PolarAngleAxis
                          dataKey="metric"
                          tick={{
                            fill: "hsl(var(--muted-foreground))",
                            fontSize: 11,
                          }}
                        />
                        <PolarRadiusAxis
                          angle={90}
                          domain={[0, 100]}
                          tick={{ fontSize: 9 }}
                          stroke="hsl(var(--border))"
                        />
                        {[
                          compareResult.baseline,
                          ...compareResult.candidates,
                        ].map((b, i) => (
                          <Radar
                            key={b.id}
                            name={b.model_name}
                            dataKey={b.model_name || b.id}
                            stroke={radarColors[i % radarColors.length]}
                            fill={radarColors[i % radarColors.length]}
                            fillOpacity={0.15}
                          />
                        ))}
                        <Legend
                          wrapperStyle={{ fontSize: 11 }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                <button
                  onClick={runCompare}
                  className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground"
                >
                  Compare {selectedIds.size} Benchmarks
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Run Benchmark Modal ──────────────────────────────── */}
      {showRunModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-card border border-border rounded-lg p-6 w-[500px] space-y-4"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Play size={18} /> Run Benchmark
              </h3>
              <button
                onClick={() => setShowRunModal(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X size={18} />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-sm text-muted-foreground">
                  Model Path *
                </label>
                <input
                  type="text"
                  value={runForm.model_path}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, model_path: e.target.value }))
                  }
                  placeholder="/path/to/model or huggingface/model-name"
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">
                  Model Name *
                </label>
                <input
                  type="text"
                  value={runForm.model_name}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, model_name: e.target.value }))
                  }
                  placeholder="e.g. llama-3-8b-lora-v2"
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">
                  GPU Cost ($/hr)
                </label>
                <input
                  type="text"
                  value={runForm.gpu_cost}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, gpu_cost: e.target.value }))
                  }
                  placeholder="e.g. 2.50"
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
                />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">
                  Tags (comma-separated)
                </label>
                <input
                  type="text"
                  value={runForm.tags}
                  onChange={(e) =>
                    setRunForm((f) => ({ ...f, tags: e.target.value }))
                  }
                  placeholder="e.g. production, lora, 8b"
                  className="w-full bg-input border border-border rounded-md px-3 py-1.5 text-sm mt-1 focus:ring-2 focus:ring-ring focus:outline-none"
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
                onClick={handleRun}
                disabled={
                  runLoading || !runForm.model_path || !runForm.model_name
                }
                className="px-3 py-1.5 text-sm rounded-md bg-primary text-primary-foreground disabled:opacity-50"
              >
                {runLoading ? "Running..." : "Run"}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}

// ── Sub-components ───────────────────────────────────────────────

function SummaryCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode
  label: string
  value: string
  color: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border border-border rounded-lg p-4"
    >
      <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
        {icon}
        {label}
      </div>
      <p className={`text-xl font-bold ${color}`}>{value}</p>
    </motion.div>
  )
}

function SortableHeader({
  label,
  sortKey,
  currentKey,
  dir,
  onClick,
  align = "left",
}: {
  label: string
  sortKey: SortKey
  currentKey: SortKey
  dir: SortDir
  onClick: (key: SortKey) => void
  align?: "left" | "right"
}) {
  const isActive = currentKey === sortKey
  return (
    <th
      className={`px-3 py-2 text-${align} cursor-pointer select-none hover:text-foreground transition-colors ${
        isActive ? "text-foreground" : "text-muted-foreground"
      }`}
      onClick={() => onClick(sortKey)}
    >
      {label}
      {isActive && (
        <span className="ml-1 text-[10px]">{dir === "asc" ? "^" : "v"}</span>
      )}
    </th>
  )
}

function BenchmarkRow({
  benchmark: b,
  expanded,
  selected,
  onExpand,
  onSelect,
  onDelete,
}: {
  benchmark: Benchmark
  expanded: boolean
  selected: boolean
  onExpand: () => void
  onSelect: () => void
  onDelete: () => void
}) {
  return (
    <>
      <tr
        className={`border-b border-border hover:bg-secondary/50 cursor-pointer transition-colors ${
          selected ? "bg-primary/5" : ""
        }`}
        onClick={onExpand}
      >
        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
          <input
            type="checkbox"
            checked={selected}
            onChange={onSelect}
            className="rounded"
          />
        </td>
        <td className="px-3 py-2">
          {expanded ? (
            <ChevronDown size={14} />
          ) : (
            <ChevronRight size={14} />
          )}
        </td>
        <td className="px-3 py-2">
          <div className="flex items-center gap-2">
            <span className="font-medium">{b.model_name}</span>
            {b.is_baseline && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30">
                BASELINE
              </span>
            )}
          </div>
          <p className="text-[10px] text-muted-foreground truncate max-w-[200px]">
            {b.model_path}
          </p>
        </td>
        <td className="px-3 py-2 text-right font-mono text-green-400">
          {b.tokens_per_sec.toFixed(1)}
        </td>
        <td className="px-3 py-2 text-right font-mono text-muted-foreground">
          {b.time_to_first_token_ms.toFixed(0)}ms
        </td>
        <td className="px-3 py-2 text-right font-mono text-muted-foreground">
          {b.perplexity !== null ? b.perplexity.toFixed(2) : "--"}
        </td>
        <td className="px-3 py-2 text-right font-mono text-muted-foreground">
          {b.peak_vram_gb.toFixed(1)} GB
        </td>
        <td className="px-3 py-2 text-right font-mono text-muted-foreground">
          ${b.estimated_cost_per_1m_tokens.toFixed(2)}
        </td>
        <td className="px-3 py-2 text-right text-muted-foreground whitespace-nowrap text-xs">
          {new Date(b.timestamp).toLocaleDateString()}
        </td>
        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={onDelete}
            className="text-muted-foreground hover:text-red-400 transition-colors"
            title="Delete"
          >
            <Trash2 size={14} />
          </button>
        </td>
      </tr>
      <AnimatePresence>
        {expanded && (
          <tr>
            <td colSpan={10} className="bg-card/50 px-0 py-0">
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <BenchmarkDetail benchmark={b} />
              </motion.div>
            </td>
          </tr>
        )}
      </AnimatePresence>
    </>
  )
}

function BenchmarkDetail({ benchmark: b }: { benchmark: Benchmark }) {
  return (
    <div className="px-6 py-4 space-y-4">
      {/* Metric grid */}
      <div className="grid grid-cols-4 gap-3">
        <MiniStat
          icon={<Zap size={14} className="text-green-400" />}
          label="Tokens/sec"
          value={b.tokens_per_sec.toFixed(1)}
        />
        <MiniStat
          icon={<Zap size={14} className="text-blue-400" />}
          label="TTFT"
          value={b.time_to_first_token_ms.toFixed(1) + "ms"}
        />
        <MiniStat
          icon={<Zap size={14} className="text-purple-400" />}
          label="Train Samples/sec"
          value={b.training_samples_per_sec.toFixed(1)}
        />
        <MiniStat
          icon={<HardDrive size={14} className="text-orange-400" />}
          label="Peak VRAM"
          value={b.peak_vram_gb.toFixed(1) + " GB"}
        />
        <MiniStat
          icon={<Brain size={14} className="text-purple-400" />}
          label="Perplexity"
          value={b.perplexity !== null ? b.perplexity.toFixed(4) : "--"}
        />
        <MiniStat
          icon={<Brain size={14} className="text-red-400" />}
          label="Eval Loss"
          value={b.eval_loss !== null ? b.eval_loss.toFixed(4) : "--"}
        />
        <MiniStat
          icon={<BarChart3 size={14} className="text-blue-400" />}
          label="Model Size"
          value={formatNumber(b.model_size_params) + " params"}
        />
        <MiniStat
          icon={<HardDrive size={14} className="text-gray-400" />}
          label="Disk Size"
          value={b.model_size_disk_mb.toFixed(0) + " MB"}
        />
      </div>

      {/* Hardware info */}
      <div className="flex gap-4 text-xs text-muted-foreground">
        <span>
          GPU: {b.hardware_info.gpu_name} x{b.hardware_info.num_gpus}
        </span>
        <span>VRAM: {b.hardware_info.vram_gb} GB</span>
        <span>Status: {b.status}</span>
        <span>Experiment: {b.experiment_id}</span>
      </div>

      {/* Tags */}
      {b.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {b.tags.map((tag) => (
            <span
              key={tag}
              className="text-[10px] px-1.5 py-0.5 rounded bg-secondary border border-border flex items-center gap-0.5"
            >
              <Tag size={8} />
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Task metrics */}
      {Object.keys(b.task_metrics).length > 0 && (
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-2">
            Task Metrics
          </h5>
          <div className="grid grid-cols-4 gap-2">
            {Object.entries(b.task_metrics).map(([key, val]) => (
              <div
                key={key}
                className="bg-secondary/30 border border-border rounded-md p-2"
              >
                <p className="text-[10px] text-muted-foreground truncate">
                  {key}
                </p>
                <p className="text-sm font-mono">{val.toFixed(4)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Config */}
      {Object.keys(b.config).length > 0 && (
        <div>
          <h5 className="text-[10px] font-medium text-muted-foreground mb-1">
            Config
          </h5>
          <pre className="text-xs whitespace-pre-wrap bg-input rounded-md p-2 border border-border font-mono max-h-32 overflow-y-auto">
            {JSON.stringify(b.config, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

function CompareMetricRow({
  label,
  value,
  delta,
  higherIsBetter,
}: {
  label: string
  value: string
  delta?: number
  higherIsBetter: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <div className="flex items-center gap-1.5">
        <span className="font-mono">{value}</span>
        {delta !== undefined && Math.abs(delta) > 0.001 && (
          <span className={`text-[10px] ${deltaColor(delta, higherIsBetter)}`}>
            {deltaArrow(delta, higherIsBetter)}{" "}
            {delta > 0 ? "+" : ""}
            {delta.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  )
}

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
