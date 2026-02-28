import { FileText } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function PromptNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  const template = String(config.template || "")
  const preview = template.length > 40 ? template.slice(0, 40) + "..." : template
  const variables = (config.variables ?? []) as string[]
  return (
    <BaseNode
      label={String(data.label || "Prompt")}
      icon={<FileText size={14} />}
      color="#06b6d4"
      outputs={[{ id: "prompt", label: "prompt" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {preview ? <div className="italic truncate">{preview}</div> : null}
      {variables.length > 0 ? <div>Vars: {variables.join(", ")}</div> : null}
    </BaseNode>
  )
}
