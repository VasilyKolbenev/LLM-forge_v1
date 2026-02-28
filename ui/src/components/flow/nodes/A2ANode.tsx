import { ArrowLeftRight } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function A2ANode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "A2A")}
      icon={<ArrowLeftRight size={14} />}
      color="#0891b2"
      inputs={[
        { id: "agent_a", label: "agent A" },
        { id: "agent_b", label: "agent B" },
      ]}
      outputs={[
        { id: "result", label: "result" },
        { id: "transcript", label: "transcript" },
      ]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.protocol ? <div>Protocol: {String(config.protocol)}</div> : null}
      {config.delegation_mode ? <div>Mode: {String(config.delegation_mode)}</div> : null}
    </BaseNode>
  )
}
