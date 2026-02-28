import { FlaskConical } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function TrainingNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string | number>
  return (
    <BaseNode
      label={String(data.label || "Training")}
      icon={<FlaskConical size={14} />}
      color="#6d5dfc"
      inputs={[
        { id: "model", label: "model" },
        { id: "dataset", label: "dataset" },
      ]}
      outputs={[{ id: "adapter", label: "adapter" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.task ? <div>Task: {String(config.task).toUpperCase()}</div> : null}
      {config.lr ? <div>LR: {String(config.lr)}</div> : null}
      {config.epochs ? <div>Epochs: {String(config.epochs)}</div> : null}
    </BaseNode>
  )
}
