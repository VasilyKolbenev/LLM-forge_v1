import { useEffect, useState } from "react"
import { api } from "@/api/client"
import { LossChart } from "@/components/training/LossChart"
import type { SSEMetrics } from "@/hooks/useSSE"
import { Trash2, GitCompare } from "lucide-react"

export function Experiments() {
  const [experiments, setExperiments] = useState<Array<Record<string, unknown>>>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [compareData, setCompareData] = useState<Record<string, unknown> | null>(null)
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null)

  const load = () => {
    api.getExperiments().then(setExperiments).catch(() => {})
  }

  useEffect(load, [])

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleCompare = async () => {
    if (selected.size < 2) return
    try {
      const res = await api.compareExperiments([...selected])
      setCompareData(res)
    } catch {
      // ignore
    }
  }

  const handleDelete = async (id: string) => {
    await api.deleteExperiment(id)
    setSelected((prev) => {
      const next = new Set(prev)
      next.delete(id)
      return next
    })
    load()
  }

  const handleDetail = async (id: string) => {
    try {
      const exp = await api.getExperiment(id)
      setDetail(exp)
    } catch {
      // ignore
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Experiments</h2>
          <p className="text-muted-foreground text-sm mt-1">{experiments.length} total</p>
        </div>
        <button
          onClick={handleCompare}
          disabled={selected.size < 2}
          className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium disabled:opacity-50"
        >
          <GitCompare size={16} />
          Compare ({selected.size})
        </button>
      </div>

      {experiments.length === 0 ? (
        <p className="text-muted-foreground">No experiments yet.</p>
      ) : (
        <div className="border border-border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-secondary/50">
                <th className="w-10 px-3 py-2"></th>
                <th className="text-left px-4 py-2 font-medium">Name</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-left px-4 py-2 font-medium">Model</th>
                <th className="text-left px-4 py-2 font-medium">Loss</th>
                <th className="text-left px-4 py-2 font-medium">Created</th>
                <th className="w-10 px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={String(exp.id)} className="border-t border-border hover:bg-secondary/30">
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selected.has(String(exp.id))}
                      onChange={() => toggleSelect(String(exp.id))}
                      className="accent-primary"
                    />
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() => handleDetail(String(exp.id))}
                      className="text-primary hover:underline"
                    >
                      {String(exp.name)}
                    </button>
                  </td>
                  <td className="px-4 py-2">
                    <StatusBadge status={String(exp.status)} />
                  </td>
                  <td className="px-4 py-2 text-muted-foreground text-xs">{String(exp.model || "—")}</td>
                  <td className="px-4 py-2 font-mono">
                    {exp.final_loss != null ? Number(exp.final_loss).toFixed(4) : "—"}
                  </td>
                  <td className="px-4 py-2 text-muted-foreground">{String(exp.created_at || "").slice(0, 16)}</td>
                  <td className="px-3 py-2">
                    <button onClick={() => handleDelete(String(exp.id))} className="text-destructive hover:text-destructive/80">
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Detail panel */}
      {detail && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold">{String(detail.name)}</h3>
            <button onClick={() => setDetail(null)} className="text-muted-foreground text-sm hover:text-foreground">
              Close
            </button>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div><span className="text-muted-foreground">Status:</span> {String(detail.status)}</div>
            <div><span className="text-muted-foreground">Task:</span> {String(detail.task || "sft")}</div>
            <div><span className="text-muted-foreground">Model:</span> {String(detail.model || "—")}</div>
            <div><span className="text-muted-foreground">Final Loss:</span> {detail.final_loss != null ? Number(detail.final_loss).toFixed(4) : "—"}</div>
          </div>
          {Array.isArray(detail.training_history) && detail.training_history.length > 0 && (
            <LossChart
              data={(detail.training_history as Array<Record<string, unknown>>).map((h) => ({
                step: Number(h.step || 0),
                epoch: Number(h.epoch || 0),
                loss: h.loss != null ? Number(h.loss) : null,
                learning_rate: null,
                gpu_mem_gb: null,
              })) as SSEMetrics[]}
            />
          )}
        </div>
      )}

      {/* Compare view */}
      {compareData && (
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold">Comparison</h3>
            <button onClick={() => setCompareData(null)} className="text-muted-foreground text-sm hover:text-foreground">
              Close
            </button>
          </div>
          <pre className="text-xs text-muted-foreground overflow-auto max-h-60">
            {JSON.stringify(compareData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    running: "bg-warning/20 text-warning",
    completed: "bg-success/20 text-success",
    failed: "bg-destructive/20 text-destructive",
    queued: "bg-muted text-muted-foreground",
  }
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${colors[status] || colors.queued}`}>
      {status}
    </span>
  )
}
