import { Database } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function DataSourceNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string>
  return (
    <BaseNode
      label={String(data.label || "Data Source")}
      icon={<Database size={14} />}
      color="#22c55e"
      outputs={[{ id: "dataset", label: "dataset" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.path ? <div>Path: {config.path}</div> : null}
      {config.format ? <div>Format: {config.format}</div> : null}
    </BaseNode>
  )
}
