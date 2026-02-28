import { GitBranch } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function ConditionalNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, string | number>
  return (
    <BaseNode
      label={String(data.label || "Condition")}
      icon={<GitBranch size={14} />}
      color="#f97316"
      inputs={[{ id: "metrics", label: "metrics" }]}
      outputs={[
        { id: "pass", label: "pass" },
        { id: "fail", label: "fail" },
      ]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.metric_name ? (
        <div>{String(config.metric_name)} {String(config.operator || "≥")} {String(config.threshold || "")}</div>
      ) : null}
    </BaseNode>
  )
}
