import type { DragEvent } from "react"
import { PERSONAS } from "./personas"

interface PaletteItem {
  type: string
  label: string
  group: string
}

const PALETTE_ITEMS: PaletteItem[] = [
  { type: "dataSource", label: "Data Source", group: "Data" },
  { type: "model", label: "Model", group: "Data" },
  { type: "prompt", label: "Prompt", group: "Data" },
  { type: "training", label: "Training", group: "Training" },
  { type: "eval", label: "Evaluation", group: "Evaluation" },
  { type: "conditional", label: "Condition", group: "Evaluation" },
  { type: "splitter", label: "Splitter", group: "Data" },
  { type: "export", label: "Export", group: "Export" },
  { type: "serve", label: "Serve", group: "Export" },
  { type: "agent", label: "Agent", group: "Agent" },
  { type: "rag", label: "RAG", group: "Agent" },
  { type: "router", label: "Router", group: "Agent" },
  { type: "inference", label: "Inference", group: "Agent" },
  { type: "dataGen", label: "Data Gen", group: "Agent" },
  { type: "mcp", label: "MCP", group: "Protocols" },
  { type: "a2a", label: "A2A", group: "Protocols" },
  { type: "gateway", label: "Gateway", group: "Protocols" },
  { type: "inputGuard", label: "Input Guard", group: "Safety" },
  { type: "outputGuard", label: "Output Guard", group: "Safety" },
  { type: "llmJudge", label: "LLM Judge", group: "Evaluation" },
  { type: "abTest", label: "A/B Test", group: "Evaluation" },
  { type: "cache", label: "Cache", group: "Ops" },
  { type: "canary", label: "Canary", group: "Ops" },
  { type: "feedback", label: "Feedback", group: "Ops" },
  { type: "tracer", label: "Tracer", group: "Ops" },
  { type: "group", label: "Group", group: "Structure" },
]

function onDragStart(e: DragEvent, nodeType: string, label: string) {
  e.dataTransfer.setData("application/reactflow-type", nodeType)
  e.dataTransfer.setData("application/reactflow-label", label)
  e.dataTransfer.effectAllowed = "move"
}

export function NodePalette() {
  const groups = PALETTE_ITEMS.reduce<Record<string, PaletteItem[]>>((acc, item) => {
    ;(acc[item.group] ??= []).push(item)
    return acc
  }, {})

  return (
    <div className="w-56 shrink-0 border-r border-border bg-card overflow-y-auto">
      <div className="p-3 border-b border-border">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Agents
        </h3>
      </div>
      <div className="p-2 space-y-3">
        {Object.entries(groups).map(([group, items]) => (
          <div key={group}>
            <div className="text-[10px] font-medium text-muted-foreground uppercase px-1 mb-1">
              {group}
            </div>
            <div className="space-y-0.5">
              {items.map((item) => {
                const persona = PERSONAS[item.type]
                const color = persona?.color || "#6d5dfc"
                return (
                  <div
                    key={item.type}
                    draggable
                    onDragStart={(e) => onDragStart(e, item.type, item.label)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded cursor-grab active:cursor-grabbing hover:bg-secondary transition-colors"
                  >
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold"
                      style={{
                        backgroundColor: `${color}25`,
                        color,
                        border: `1.5px solid ${color}50`,
                      }}
                    >
                      {persona?.name.charAt(0) || "?"}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-medium truncate">
                        {persona?.name || item.label}
                      </div>
                      <div className="text-[9px] text-muted-foreground truncate">
                        {persona?.role || ""}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
