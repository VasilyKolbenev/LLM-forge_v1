import { MessageSquare } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

const FRAMEWORK_LABELS: Record<string, string> = {
  "forge-react": "Forge ReAct",
  langgraph: "LangGraph",
  crewai: "CrewAI",
  autogen: "AutoGen",
  custom: "Custom",
}

export function AgentNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  const tools = (config.tools ?? []) as string[]
  const framework = String(config.framework || "forge-react")
  return (
    <BaseNode
      label={String(data.label || "Agent")}
      icon={<MessageSquare size={14} />}
      color="#8b5cf6"
      inputs={[{ id: "model", label: "model" }]}
      outputs={[{ id: "agent", label: "agent" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      <div>{FRAMEWORK_LABELS[framework] || framework}</div>
      {tools.length > 0 ? (
        <div>Tools: {tools.slice(0, 3).join(", ")}{tools.length > 3 ? "..." : ""}</div>
      ) : null}
    </BaseNode>
  )
}
