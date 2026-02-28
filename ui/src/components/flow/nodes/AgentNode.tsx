import { MessageSquare } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function AgentNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  const tools = (config.tools ?? []) as string[]
  return (
    <BaseNode
      label={String(data.label || "Agent")}
      icon={<MessageSquare size={14} />}
      color="#8b5cf6"
      inputs={[{ id: "model", label: "model" }]}
      outputs={[{ id: "agent", label: "agent" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {tools.length > 0 ? (
        <div>Tools: {tools.slice(0, 3).join(", ")}{tools.length > 3 ? "..." : ""}</div>
      ) : null}
    </BaseNode>
  )
}
