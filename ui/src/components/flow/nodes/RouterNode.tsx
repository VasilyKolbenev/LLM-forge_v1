import { Route } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function RouterNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  const routes = (config.routes ?? []) as string[]
  return (
    <BaseNode
      label={String(data.label || "Router")}
      icon={<Route size={14} />}
      color="#f43f5e"
      inputs={[{ id: "input", label: "input" }]}
      outputs={[
        { id: "route_a", label: "route A" },
        { id: "route_b", label: "route B" },
        { id: "fallback", label: "fallback" },
      ]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.strategy ? <div>Strategy: {String(config.strategy)}</div> : null}
      {routes.length > 0 ? (
        <div>Routes: {routes.slice(0, 3).join(", ")}{routes.length > 3 ? "..." : ""}</div>
      ) : null}
    </BaseNode>
  )
}
