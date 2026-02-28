import { Plug } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

const ROLE_LABELS: Record<string, string> = {
  server: "MCP Server",
  client: "MCP Client",
}

export function MCPNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  const role = String(config.role || "server")
  return (
    <BaseNode
      label={String(data.label || "MCP")}
      icon={<Plug size={14} />}
      color="#7c3aed"
      inputs={[
        { id: "agent", label: "agent" },
        { id: "tools", label: "tools" },
      ]}
      outputs={[{ id: "endpoint", label: "endpoint" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      <div>{ROLE_LABELS[role] || role}</div>
      {config.transport ? <div>Transport: {String(config.transport)}</div> : null}
      {config.tools_exposed ? <div>Tools: {String(config.tools_exposed)}</div> : null}
    </BaseNode>
  )
}
