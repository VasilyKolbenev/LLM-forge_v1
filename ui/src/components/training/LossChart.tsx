import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import type { SSEMetrics } from "@/hooks/useSSE"

interface Props {
  data: SSEMetrics[]
  height?: number
}

export function LossChart({ data, height = 300 }: Props) {
  const chartData = data
    .filter((d) => d.loss != null)
    .map((d) => ({ step: d.step, loss: d.loss }))

  if (chartData.length === 0) {
    return <p className="text-muted-foreground text-sm">Waiting for loss data...</p>
  }

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <h4 className="text-sm font-medium mb-3">Training Loss</h4>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="step"
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#27272a" }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            axisLine={{ stroke: "#27272a" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111113",
              border: "1px solid #27272a",
              borderRadius: "6px",
              fontSize: "12px",
            }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#6d5dfc"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
