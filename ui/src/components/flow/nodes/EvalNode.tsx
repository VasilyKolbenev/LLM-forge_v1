import { ListChecks } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function EvalNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string | number>
  return (
    <BaseNode
      label={String(data.label || "Evaluation")}
      icon={<ListChecks size={14} />}
      color="#eab308"
      inputs={[
        { id: "adapter", label: "adapter" },
        { id: "dataset", label: "dataset" },
      ]}
      outputs={[{ id: "metrics", label: "metrics" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.batch_size ? <div>Batch: {String(config.batch_size)}</div> : null}
    </BaseNode>
  )
}
