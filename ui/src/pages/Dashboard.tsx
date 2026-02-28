import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import {
  FlaskConical,
  Database,
  Cpu,
  MonitorDot,
  Workflow,
  FileText,
  Server,
  ArrowRight,
} from "lucide-react"
import {
  AreaChart,
  Area,
  ResponsiveContainer,
} from "recharts"
import { api } from "@/api/client"
import { useMetrics } from "@/hooks/useMetrics"
import { AnimatedPage, FadeIn } from "@/components/ui/AnimatedPage"

export function Dashboard() {
  const [experiments, setExperiments] = useState<Array<Record<string, unknown>>>([])
  const [hardware, setHardware] = useState<Record<string, unknown> | null>(null)
  const [datasets, setDatasets] = useState<Array<Record<string, unknown>>>([])
  const [workflows, setWorkflows] = useState<Array<Record<string, unknown>>>([])
  const [prompts, setPrompts] = useState<Array<Record<string, unknown>>>([])
  const { current, history } = useMetrics()

  useEffect(() => {
    api.getExperiments().then(setExperiments).catch(() => {})
    api.getHardware().then(setHardware).catch(() => {})
    api.getDatasets().then(setDatasets).catch(() => {})
    api.listWorkflows().then(setWorkflows).catch(() => {})
    api.listPrompts().then(setPrompts).catch(() => {})
  }, [])

  const running = experiments.filter((e) => e.status === "running").length
  const completed = experiments.filter((e) => e.status === "completed").length
  const gpu = current?.gpus[0]

  // Mini chart data (last 60 points = 2 min)
  const miniChart = history.slice(-60).map((m, i) => ({
    t: i,
    gpu: m.gpus[0]?.utilization_percent ?? 0,
    cpu: m.cpu_percent,
  }))

  return (
    <AnimatedPage>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold">Dashboard</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Overview of your fine-tuning platform
          </p>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <FadeIn delay={0}>
            <StatCard
              icon={FlaskConical}
              label="Experiments"
              value={experiments.length}
              sub={`${running} running · ${completed} completed`}
              color="#6d5dfc"
            />
          </FadeIn>
          <FadeIn delay={0.05}>
            <StatCard
              icon={Database}
              label="Datasets"
              value={datasets.length}
              sub="uploaded"
              color="#3b82f6"
            />
          </FadeIn>
          <FadeIn delay={0.1}>
            <StatCard
              icon={Workflow}
              label="Workflows"
              value={workflows.length}
              sub="saved pipelines"
              color="#22c55e"
            />
          </FadeIn>
          <FadeIn delay={0.15}>
            <StatCard
              icon={FileText}
              label="Prompts"
              value={prompts.length}
              sub="versioned"
              color="#06b6d4"
            />
          </FadeIn>
        </div>

        {/* GPU + CPU mini charts */}
        <FadeIn delay={0.2}>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* GPU card */}
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-[#6d5dfc]/10">
                  <MonitorDot size={16} className="text-[#6d5dfc]" />
                </div>
                <div>
                  <div className="text-sm font-medium">GPU</div>
                  <div className="text-[10px] text-muted-foreground">
                    {gpu ? gpu.name : hardware ? String(hardware.gpu_name || "N/A") : "detecting..."}
                  </div>
                </div>
                <div className="ml-auto text-right">
                  <div className="text-lg font-bold">
                    {gpu ? `${Math.round(gpu.utilization_percent)}%` : "—"}
                  </div>
                  <div className="text-[10px] text-muted-foreground">
                    {gpu ? `${gpu.memory_used_gb.toFixed(1)}/${gpu.memory_total_gb.toFixed(1)} GB` : ""}
                  </div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={80}>
                <AreaChart data={miniChart}>
                  <defs>
                    <linearGradient id="gpuGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6d5dfc" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#6d5dfc" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="gpu"
                    stroke="#6d5dfc"
                    fill="url(#gpuGrad)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* CPU card */}
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-[#22c55e]/10">
                  <Cpu size={16} className="text-[#22c55e]" />
                </div>
                <div>
                  <div className="text-sm font-medium">CPU</div>
                  <div className="text-[10px] text-muted-foreground">
                    {current ? `${current.cpu_count} cores` : "detecting..."}
                  </div>
                </div>
                <div className="ml-auto text-right">
                  <div className="text-lg font-bold">
                    {current ? `${Math.round(current.cpu_percent)}%` : "—"}
                  </div>
                  <div className="text-[10px] text-muted-foreground">
                    {current ? `${current.ram_used_gb.toFixed(1)}/${current.ram_total_gb.toFixed(1)} GB RAM` : ""}
                  </div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={80}>
                <AreaChart data={miniChart}>
                  <defs>
                    <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="cpu"
                    stroke="#22c55e"
                    fill="url(#cpuGrad)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Quick actions card */}
            <div className="bg-card border border-border rounded-lg p-4 flex flex-col">
              <div className="text-sm font-medium mb-3">Quick Actions</div>
              <div className="space-y-2 flex-1">
                <QuickAction to="/new" icon={FlaskConical} label="New Experiment" color="#6d5dfc" />
                <QuickAction to="/workflows" icon={Workflow} label="Build Workflow" color="#22c55e" />
                <QuickAction to="/prompts" icon={FileText} label="New Prompt" color="#06b6d4" />
                <QuickAction to="/compute" icon={Server} label="Add Compute" color="#f97316" />
              </div>
            </div>
          </div>
        </FadeIn>

        {/* Recent experiments */}
        <FadeIn delay={0.25}>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold">Recent Experiments</h3>
              <Link
                to="/experiments"
                className="flex items-center gap-1 text-xs text-primary hover:text-primary/80"
              >
                View all <ArrowRight size={12} />
              </Link>
            </div>
            {experiments.length === 0 ? (
              <p className="text-muted-foreground text-sm py-4 text-center">
                No experiments yet. Start a new one!
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-muted-foreground border-b border-border">
                      <th className="text-left px-3 py-2 font-medium">Name</th>
                      <th className="text-left px-3 py-2 font-medium">Status</th>
                      <th className="text-left px-3 py-2 font-medium">Task</th>
                      <th className="text-right px-3 py-2 font-medium">Loss</th>
                      <th className="text-right px-3 py-2 font-medium">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.slice(0, 8).map((exp) => (
                      <tr
                        key={String(exp.id)}
                        className="border-t border-border/50 hover:bg-secondary/30 transition-colors"
                      >
                        <td className="px-3 py-2">
                          <Link to="/experiments" className="text-primary hover:underline text-xs">
                            {String(exp.name)}
                          </Link>
                        </td>
                        <td className="px-3 py-2">
                          <StatusBadge status={String(exp.status)} />
                        </td>
                        <td className="px-3 py-2 text-xs text-muted-foreground">
                          {String(exp.task || "sft").toUpperCase()}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-xs">
                          {exp.final_loss != null ? Number(exp.final_loss).toFixed(4) : "—"}
                        </td>
                        <td className="px-3 py-2 text-right text-xs text-muted-foreground">
                          {String(exp.created_at || "").slice(0, 10)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </FadeIn>
      </div>
    </AnimatedPage>
  )
}

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ComponentType<{ size?: number }>
  label: string
  value: string | number
  sub: string
  color: string
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:border-border/80 transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon size={16} />
        </div>
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {label}
        </span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs text-muted-foreground mt-1">{sub}</div>
    </div>
  )
}

function QuickAction({
  to,
  icon: Icon,
  label,
  color,
}: {
  to: string
  icon: React.ComponentType<{ size?: number }>
  label: string
  color: string
}) {
  return (
    <Link
      to={to}
      className="flex items-center gap-2.5 px-3 py-2 rounded-md hover:bg-secondary transition-colors group"
    >
      <div
        className="w-6 h-6 rounded flex items-center justify-center shrink-0"
        style={{ backgroundColor: `${color}15`, color }}
      >
        <Icon size={13} />
      </div>
      <span className="text-xs">{label}</span>
      <ArrowRight
        size={12}
        className="ml-auto text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
      />
    </Link>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    running: "bg-warning/20 text-warning",
    completed: "bg-success/20 text-success",
    failed: "bg-destructive/20 text-destructive",
    queued: "bg-muted text-muted-foreground",
    cancelled: "bg-muted text-muted-foreground",
  }
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium ${
        colors[status] || colors.queued
      }`}
    >
      {status}
    </span>
  )
}
