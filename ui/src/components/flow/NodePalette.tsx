import {
  Database,
  Box,
  FlaskConical,
  ListChecks,
  Download,
  MessageSquare,
  FileText,
  GitBranch,
} from "lucide-react"
import type { DragEvent } from "react"

interface PaletteItem {
  type: string
  label: string
  icon: React.ElementType
  color: string
  group: string
}

const PALETTE_ITEMS: PaletteItem[] = [
  { type: "dataSource", label: "Data Source", icon: Database, color: "#22c55e", group: "Data" },
  { type: "model", label: "Model", icon: Box, color: "#3b82f6", group: "Data" },
  { type: "prompt", label: "Prompt", icon: FileText, color: "#06b6d4", group: "Data" },
  { type: "training", label: "Training", icon: FlaskConical, color: "#6d5dfc", group: "Training" },
  { type: "eval", label: "Evaluation", icon: ListChecks, color: "#eab308", group: "Evaluation" },
  { type: "conditional", label: "Condition", icon: GitBranch, color: "#f97316", group: "Evaluation" },
  { type: "export", label: "Export", icon: Download, color: "#ef4444", group: "Export" },
  { type: "agent", label: "Agent", icon: MessageSquare, color: "#8b5cf6", group: "Agent" },
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
    <div className="w-48 shrink-0 border-r border-border bg-card overflow-y-auto">
      <div className="p-3 border-b border-border">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
          Nodes
        </h3>
      </div>
      <div className="p-2 space-y-3">
        {Object.entries(groups).map(([group, items]) => (
          <div key={group}>
            <div className="text-[10px] font-medium text-muted-foreground uppercase px-1 mb-1">
              {group}
            </div>
            <div className="space-y-1">
              {items.map((item) => (
                <div
                  key={item.type}
                  draggable
                  onDragStart={(e) => onDragStart(e, item.type, item.label)}
                  className="flex items-center gap-2 px-2 py-1.5 rounded cursor-grab active:cursor-grabbing hover:bg-secondary transition-colors"
                >
                  <div
                    className="w-5 h-5 rounded flex items-center justify-center shrink-0"
                    style={{ backgroundColor: `${item.color}20`, color: item.color }}
                  >
                    <item.icon size={12} />
                  </div>
                  <span className="text-xs">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
