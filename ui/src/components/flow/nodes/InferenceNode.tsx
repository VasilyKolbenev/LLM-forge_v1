import { Zap } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function InferenceNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "Inference")}
      icon={<Zap size={14} />}
      color="#a855f7"
      inputs={[
        { id: "model", label: "model" },
        { id: "data", label: "data" },
      ]}
      outputs={[{ id: "output", label: "output" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.max_tokens ? <div>Max tokens: {String(config.max_tokens)}</div> : null}
      {config.temperature ? <div>Temp: {String(config.temperature)}</div> : null}
    </BaseNode>
  )
}
