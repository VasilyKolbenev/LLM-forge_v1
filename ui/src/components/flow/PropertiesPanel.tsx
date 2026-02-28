import { X } from "lucide-react"
import type { Node } from "@xyflow/react"

interface PropertiesPanelProps {
  node: Node | null
  onClose: () => void
  onUpdate: (nodeId: string, data: Record<string, unknown>) => void
}

interface FieldDef {
  key: string
  label: string
  type: "text" | "number" | "select" | "textarea"
  options?: string[]
  placeholder?: string
}

const FIELDS_BY_TYPE: Record<string, FieldDef[]> = {
  dataSource: [
    { key: "path", label: "File Path", type: "text", placeholder: "/data/train.jsonl" },
    { key: "format", label: "Format", type: "select", options: ["jsonl", "csv", "parquet", "huggingface"] },
    { key: "split", label: "Split", type: "select", options: ["train", "validation", "test"] },
  ],
  model: [
    { key: "model_id", label: "Model ID", type: "text", placeholder: "meta-llama/Llama-3-8B" },
    { key: "quantization", label: "Quantization", type: "select", options: ["none", "4bit", "8bit"] },
  ],
  training: [
    { key: "task", label: "Task", type: "select", options: ["sft", "dpo"] },
    { key: "lr", label: "Learning Rate", type: "number", placeholder: "2e-5" },
    { key: "epochs", label: "Epochs", type: "number", placeholder: "3" },
    { key: "batch_size", label: "Batch Size", type: "number", placeholder: "4" },
    { key: "max_seq_length", label: "Max Seq Length", type: "number", placeholder: "2048" },
  ],
  eval: [
    { key: "batch_size", label: "Batch Size", type: "number", placeholder: "8" },
    { key: "max_tokens", label: "Max Tokens", type: "number", placeholder: "512" },
  ],
  export: [
    { key: "format", label: "Format", type: "select", options: ["gguf", "merged", "lora", "huggingface"] },
    { key: "quantization", label: "Quantization", type: "select", options: ["none", "q4_k_m", "q5_k_m", "q8_0", "f16"] },
  ],
  agent: [
    { key: "system_prompt", label: "System Prompt", type: "textarea", placeholder: "You are a helpful assistant..." },
    { key: "tools", label: "Tools (comma-separated)", type: "text", placeholder: "search,calculator,code" },
  ],
  prompt: [
    { key: "template", label: "Template", type: "textarea", placeholder: "{{input}}\n\nRespond as..." },
    { key: "variables", label: "Variables (comma-separated)", type: "text", placeholder: "input,context" },
  ],
  conditional: [
    { key: "metric_name", label: "Metric", type: "text", placeholder: "accuracy" },
    { key: "operator", label: "Operator", type: "select", options: [">=", ">", "<=", "<", "==", "!="] },
    { key: "threshold", label: "Threshold", type: "number", placeholder: "0.8" },
  ],
}

export function PropertiesPanel({ node, onClose, onUpdate }: PropertiesPanelProps) {
  if (!node) return null

  const nodeType = node.type || "default"
  const fields = FIELDS_BY_TYPE[nodeType] || []
  const config = (node.data.config as Record<string, unknown>) || {}

  function handleChange(key: string, value: string) {
    const newConfig = { ...config }
    const field = fields.find((f) => f.key === key)
    if (field?.type === "number" && value) {
      newConfig[key] = parseFloat(value)
    } else {
      newConfig[key] = value
    }
    onUpdate(node!.id, { ...node!.data, config: newConfig })
  }

  function handleLabelChange(value: string) {
    onUpdate(node!.id, { ...node!.data, label: value })
  }

  return (
    <div className="w-64 shrink-0 border-l border-border bg-card overflow-y-auto">
      <div className="flex items-center justify-between p-3 border-b border-border">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Properties
        </h3>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X size={14} />
        </button>
      </div>
      <div className="p-3 space-y-3">
        {/* Label */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase">
            Label
          </label>
          <input
            type="text"
            value={String(node.data.label || "")}
            onChange={(e) => handleLabelChange(e.target.value)}
            className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Type-specific fields */}
        {fields.map((field) => (
          <div key={field.key}>
            <label className="text-[10px] font-medium text-muted-foreground uppercase">
              {field.label}
            </label>
            {field.type === "select" ? (
              <select
                value={String(config[field.key] || "")}
                onChange={(e) => handleChange(field.key, e.target.value)}
                className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="">Select...</option>
                {field.options?.map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            ) : field.type === "textarea" ? (
              <textarea
                value={String(config[field.key] || "")}
                onChange={(e) => handleChange(field.key, e.target.value)}
                placeholder={field.placeholder}
                rows={3}
                className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary resize-none"
              />
            ) : (
              <input
                type={field.type === "number" ? "text" : "text"}
                value={String(config[field.key] ?? "")}
                onChange={(e) => handleChange(field.key, e.target.value)}
                placeholder={field.placeholder}
                className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              />
            )}
          </div>
        ))}

        {/* Node info */}
        <div className="pt-2 border-t border-border text-[10px] text-muted-foreground space-y-1">
          <div>ID: {node.id}</div>
          <div>Type: {nodeType}</div>
        </div>
      </div>
    </div>
  )
}
