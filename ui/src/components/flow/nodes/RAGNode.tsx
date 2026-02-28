import { Search } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function RAGNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "RAG")}
      icon={<Search size={14} />}
      color="#0ea5e9"
      inputs={[
        { id: "model", label: "model" },
        { id: "data", label: "data" },
      ]}
      outputs={[{ id: "output", label: "output" }]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.embedding_model ? (
        <div>Embed: {String(config.embedding_model)}</div>
      ) : null}
      {config.top_k ? <div>Top-K: {String(config.top_k)}</div> : null}
      {config.vector_store ? <div>Store: {String(config.vector_store)}</div> : null}
    </BaseNode>
  )
}
