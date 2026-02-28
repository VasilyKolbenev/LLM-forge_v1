import { Sparkles } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function DataGenNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "Data Gen")}
      icon={<Sparkles size={14} />}
      color="#14b8a6"
      inputs={[{ id: "agent", label: "agent" }]}
      outputs={[{ id: "dataset", label: "dataset" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.output_format ? <div>Format: {String(config.output_format)}</div> : null}
      {config.num_samples ? <div>Samples: {String(config.num_samples)}</div> : null}
    </BaseNode>
  )
}
