import { Box } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function ModelNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string>
  const modelId = config.model_id || ""
  const shortName = modelId.split("/").pop() || "Model"
  return (
    <BaseNode
      label={String(data.label || "Model")}
      icon={<Box size={14} />}
      color="#3b82f6"
      outputs={[{ id: "model", label: "model" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {modelId ? <div className="truncate">{shortName}</div> : null}
      {config.quantization ? <div>Quant: {config.quantization}</div> : null}
    </BaseNode>
  )
}
