import { memo } from "react"
import { Handle, Position, type NodeProps } from "@xyflow/react"
import { motion } from "framer-motion"
import {
  Database,
  FlaskConical,
  Box,
  Globe,
  Activity,
  ThumbsUp,
} from "lucide-react"

export interface LineageNodeData {
  type: "dataset" | "experiment" | "model" | "deployment" | "traces" | "feedback"
  label: string
  status?: string
  metadata?: Record<string, unknown>
}

const TYPE_CONFIG: Record<
  LineageNodeData["type"],
  { color: string; Icon: React.ComponentType<{ size?: number; className?: string }> }
> = {
  dataset: { color: "#14b8a6", Icon: Database },
  experiment: { color: "#6d5dfc", Icon: FlaskConical },
  model: { color: "#3b82f6", Icon: Box },
  deployment: { color: "#ef4444", Icon: Globe },
  traces: { color: "#6366f1", Icon: Activity },
  feedback: { color: "#0ea5e9", Icon: ThumbsUp },
}

const STATUS_DOT_COLOR: Record<string, string> = {
  completed: "#22c55e",
  running: "#3b82f6",
  pending: "#eab308",
  failed: "#ef4444",
}

function buildSubtitle(
  type: LineageNodeData["type"],
  metadata?: Record<string, unknown>,
): string {
  if (!metadata) return ""
  const parts: string[] = []

  switch (type) {
    case "dataset": {
      if (metadata.row_count != null) parts.push(`${metadata.row_count} rows`)
      if (metadata.format) parts.push(String(metadata.format))
      break
    }
    case "experiment": {
      if (metadata.task) parts.push(String(metadata.task))
      if (metadata.loss != null) parts.push(`loss ${metadata.loss}`)
      break
    }
    case "model": {
      if (metadata.format) parts.push(String(metadata.format))
      break
    }
    case "deployment": {
      if (metadata.backend) parts.push(String(metadata.backend))
      if (metadata.status) parts.push(String(metadata.status))
      break
    }
    case "traces": {
      if (metadata.count != null) parts.push(`${metadata.count} traces`)
      if (metadata.avg_latency != null) parts.push(`${metadata.avg_latency}ms`)
      break
    }
    case "feedback": {
      const up = metadata.thumbs_up ?? 0
      const down = metadata.thumbs_down ?? 0
      parts.push(`+${up} / -${down}`)
      break
    }
  }

  return parts.join(" \u00B7 ")
}

function LineageNodeInner({ data }: NodeProps) {
  const nodeData = data as unknown as LineageNodeData
  const config = TYPE_CONFIG[nodeData.type] ?? TYPE_CONFIG.model
  const { color, Icon } = config
  const subtitle = buildSubtitle(nodeData.type, nodeData.metadata)
  const statusColor = STATUS_DOT_COLOR[nodeData.status ?? ""] ?? "#52525b"

  return (
    <div className="relative">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="bg-card border border-border rounded-lg shadow-lg w-[180px]"
        style={{ borderColor: `${color}40` }}
      >
        {/* Header */}
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-t-lg border-b border-border"
          style={{ backgroundColor: `${color}12` }}
        >
          <div
            className="w-6 h-6 rounded flex items-center justify-center shrink-0"
            style={{ backgroundColor: `${color}25`, color }}
          >
            <Icon size={14} />
          </div>
          <span className="text-xs font-semibold truncate flex-1">
            {nodeData.label}
          </span>
        </div>

        {/* Body */}
        <div className="px-3 py-2 space-y-1">
          {subtitle && (
            <div className="text-[10px] text-muted-foreground truncate">
              {subtitle}
            </div>
          )}
          {nodeData.status && (
            <div className="flex items-center gap-1.5">
              <span
                className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ backgroundColor: statusColor }}
              />
              <span className="text-[10px] text-muted-foreground">
                {nodeData.status}
              </span>
            </div>
          )}
        </div>

        {/* Handles */}
        <Handle
          type="target"
          position={Position.Left}
          style={{
            background: color,
            width: 8,
            height: 8,
            border: "2px solid var(--color-card)",
          }}
        />
        <Handle
          type="source"
          position={Position.Right}
          style={{
            background: color,
            width: 8,
            height: 8,
            border: "2px solid var(--color-card)",
          }}
        />
      </motion.div>
    </div>
  )
}

export const LineageNode = memo(LineageNodeInner)
