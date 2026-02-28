import { Globe } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function ServeNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "Serve")}
      icon={<Globe size={14} />}
      color="#ec4899"
      inputs={[{ id: "model", label: "model" }]}
      outputs={[{ id: "endpoint", label: "endpoint" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.engine ? <div>Engine: {String(config.engine)}</div> : null}
      {config.port ? <div>Port: {String(config.port)}</div> : null}
    </BaseNode>
  )
}
