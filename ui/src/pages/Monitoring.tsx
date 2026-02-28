import { useMemo } from "react"
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"
import { Cpu, HardDrive, MonitorDot } from "lucide-react"
import { useMetrics } from "@/hooks/useMetrics"

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ElementType
  label: string
  value: string
  sub?: string
  color: string
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon size={16} style={{ color }} />
        </div>
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      {sub && <div className="text-xs text-muted-foreground mt-1">{sub}</div>}
    </div>
  )
}

function MetricChart({
  data,
  dataKey,
  color,
  title,
  unit,
  max,
  type = "line",
}: {
  data: Array<Record<string, unknown>>
  dataKey: string
  color: string
  title: string
  unit: string
  max?: number
  type?: "line" | "area"
}) {
  const Chart = type === "area" ? AreaChart : LineChart

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium">{title}</span>
        <span className="text-xs text-muted-foreground">{unit}</span>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <Chart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, max || "auto"]}
            tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "var(--color-card)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          {type === "area" ? (
            <Area
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              fill={`${color}30`}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ) : (
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  )
}

export function Monitoring() {
  const { current, history, connected } = useMetrics()

  const chartData = useMemo(() => {
    return history.map((m, i) => {
      const gpu = m.gpus[0]
      return {
        time: i % 30 === 0 ? new Date(m.timestamp * 1000).toLocaleTimeString() : "",
        cpu: m.cpu_percent,
        ram: m.ram_used_gb,
        ramPct: m.ram_percent,
        gpuUtil: gpu?.utilization_percent ?? 0,
        gpuMem: gpu?.memory_used_gb ?? 0,
        gpuTemp: gpu?.temperature_c ?? 0,
        gpuPower: gpu?.power_watts ?? 0,
      }
    })
  }, [history])

  const gpu = current?.gpus[0]
  const hasGpu = !!gpu

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">System Monitoring</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time hardware metrics
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-success" : "bg-destructive animate-pulse"
            }`}
          />
          {connected ? "Connected" : "Reconnecting..."}
        </div>
      </div>

      {/* Stat cards */}
      <div className={`grid gap-4 ${hasGpu ? "grid-cols-2 lg:grid-cols-4" : "grid-cols-2 lg:grid-cols-3"}`}>
        {hasGpu && (
          <StatCard
            icon={MonitorDot}
            label="GPU Utilization"
            value={`${Math.round(gpu.utilization_percent)}%`}
            sub={gpu.name}
            color="#6d5dfc"
          />
        )}
        {hasGpu && (
          <StatCard
            icon={HardDrive}
            label="VRAM"
            value={`${gpu.memory_used_gb.toFixed(1)} GB`}
            sub={`of ${gpu.memory_total_gb.toFixed(1)} GB`}
            color="#8b5cf6"
          />
        )}
        <StatCard
          icon={Cpu}
          label="CPU"
          value={`${Math.round(current?.cpu_percent ?? 0)}%`}
          sub={`${current?.cpu_count ?? 0} cores`}
          color="#22c55e"
        />
        <StatCard
          icon={HardDrive}
          label="RAM"
          value={`${current?.ram_used_gb.toFixed(1) ?? "0"} GB`}
          sub={`of ${current?.ram_total_gb.toFixed(1) ?? "0"} GB (${Math.round(current?.ram_percent ?? 0)}%)`}
          color="#3b82f6"
        />
      </div>

      {/* GPU charts */}
      {hasGpu && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={chartData}
              dataKey="gpuUtil"
              color="#6d5dfc"
              title="GPU Utilization"
              unit="%"
              max={100}
              type="area"
            />
            <MetricChart
              data={chartData}
              dataKey="gpuMem"
              color="#8b5cf6"
              title="VRAM Usage"
              unit="GB"
              max={gpu.memory_total_gb}
              type="area"
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={chartData}
              dataKey="gpuTemp"
              color="#ef4444"
              title="GPU Temperature"
              unit="°C"
              max={100}
            />
            <MetricChart
              data={chartData}
              dataKey="gpuPower"
              color="#eab308"
              title="Power Draw"
              unit="W"
            />
          </div>
        </>
      )}

      {/* CPU & RAM charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={chartData}
          dataKey="cpu"
          color="#22c55e"
          title="CPU Usage"
          unit="%"
          max={100}
          type="area"
        />
        <MetricChart
          data={chartData}
          dataKey="ram"
          color="#3b82f6"
          title="RAM Usage"
          unit="GB"
          max={current?.ram_total_gb}
          type="area"
        />
      </div>

      {/* Multi-GPU table */}
      {current && current.gpus.length > 1 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium mb-3">All GPUs</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-muted-foreground border-b border-border">
                  <th className="text-left py-2 px-2">#</th>
                  <th className="text-left py-2 px-2">Name</th>
                  <th className="text-right py-2 px-2">Util %</th>
                  <th className="text-right py-2 px-2">VRAM</th>
                  <th className="text-right py-2 px-2">Temp</th>
                  <th className="text-right py-2 px-2">Power</th>
                </tr>
              </thead>
              <tbody>
                {current.gpus.map((g) => (
                  <tr key={g.index} className="border-b border-border/50">
                    <td className="py-2 px-2 text-muted-foreground">{g.index}</td>
                    <td className="py-2 px-2">{g.name}</td>
                    <td className="py-2 px-2 text-right font-mono">
                      {Math.round(g.utilization_percent)}%
                    </td>
                    <td className="py-2 px-2 text-right font-mono">
                      {g.memory_used_gb.toFixed(1)}/{g.memory_total_gb.toFixed(1)} GB
                    </td>
                    <td className="py-2 px-2 text-right font-mono">{g.temperature_c}&deg;C</td>
                    <td className="py-2 px-2 text-right font-mono">{g.power_watts}W</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
