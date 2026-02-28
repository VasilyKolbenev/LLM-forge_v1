import { Handle, Position } from "@xyflow/react"
import type { ReactNode } from "react"

export interface BaseNodeProps {
  label: string
  icon: ReactNode
  color: string
  inputs?: { id: string; label: string }[]
  outputs?: { id: string; label: string }[]
  children?: ReactNode
  status?: "idle" | "running" | "done" | "error"
}

const statusColors: Record<string, string> = {
  idle: "bg-muted",
  running: "bg-warning animate-pulse",
  done: "bg-success",
  error: "bg-destructive",
}

export function BaseNode({
  label,
  icon,
  color,
  inputs = [],
  outputs = [],
  children,
  status = "idle",
}: BaseNodeProps) {
  return (
    <div className="bg-card border border-border rounded-lg shadow-lg min-w-[180px] max-w-[240px]">
      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 py-2 rounded-t-lg border-b border-border"
        style={{ backgroundColor: `${color}15` }}
      >
        <div
          className="w-6 h-6 rounded flex items-center justify-center shrink-0"
          style={{ backgroundColor: `${color}30`, color }}
        >
          {icon}
        </div>
        <span className="text-xs font-medium truncate flex-1">{label}</span>
        <div className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
      </div>

      {/* Body */}
      {children && (
        <div className="px-3 py-2 text-[10px] text-muted-foreground space-y-1">
          {children}
        </div>
      )}

      {/* Handles */}
      {inputs.map((inp, i) => (
        <Handle
          key={inp.id}
          type="target"
          position={Position.Left}
          id={inp.id}
          style={{
            top: `${((i + 1) / (inputs.length + 1)) * 100}%`,
            background: color,
            width: 8,
            height: 8,
            border: "2px solid var(--color-card)",
          }}
        />
      ))}
      {outputs.map((out, i) => (
        <Handle
          key={out.id}
          type="source"
          position={Position.Right}
          id={out.id}
          style={{
            top: `${((i + 1) / (outputs.length + 1)) * 100}%`,
            background: color,
            width: 8,
            height: 8,
            border: "2px solid var(--color-card)",
          }}
        />
      ))}
    </div>
  )
}
