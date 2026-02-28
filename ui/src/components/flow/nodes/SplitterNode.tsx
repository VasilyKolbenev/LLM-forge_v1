import { Scissors } from "lucide-react"
import { BaseNode } from "./BaseNode"
import type { NodeProps } from "@xyflow/react"

export function SplitterNode({ data }: NodeProps) {
  const config = (data.config ?? {}) as Record<string, unknown>
  return (
    <BaseNode
      label={String(data.label || "Splitter")}
      icon={<Scissors size={14} />}
      color="#84cc16"
      inputs={[{ id: "data", label: "data" }]}
      outputs={[
        { id: "train", label: "train" },
        { id: "val", label: "val" },
        { id: "test", label: "test" },
      ]}
      status={String(data.status || "idle") as "idle" | "running" | "done" | "error"}
    >
      {config.train_ratio ? (
        <div>
          Split: {String(config.train_ratio)}/{String(config.val_ratio ?? 0.1)}/{String(config.test_ratio ?? 0.1)}
        </div>
      ) : null}
      {config.strategy ? <div>Strategy: {String(config.strategy)}</div> : null}
    </BaseNode>
  )
}
