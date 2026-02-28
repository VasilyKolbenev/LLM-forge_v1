import { Network } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function GatewayNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "Gateway")}
      icon={<Network size={14} />}
      color="#d97706"
      inputs={[
        { id: "agents", label: "agents" },
        { id: "config", label: "config" },
      ]}
      outputs={[
        { id: "api", label: "API" },
        { id: "webhook", label: "webhook" },
      ]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.protocols ? <div>Protocols: {String(config.protocols)}</div> : null}
      {config.auth_method ? <div>Auth: {String(config.auth_method)}</div> : null}
      {config.rate_limit ? <div>Rate: {String(config.rate_limit)}/min</div> : null}
    </BaseNode>
  )
}
