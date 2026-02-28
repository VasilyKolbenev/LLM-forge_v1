import { Download } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function ExportNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string>
  return (
    <BaseNode
      label={String(data.label || "Export")}
      icon={<Download size={14} />}
      color="#ef4444"
      inputs={[{ id: "adapter", label: "adapter" }]}
      outputs={[{ id: "artifact", label: "artifact" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.format ? <div>Format: {config.format}</div> : null}
      {config.quantization ? <div>Quant: {config.quantization}</div> : null}
    </BaseNode>
  )
}
