import { useMemo } from "react"
import type { Node } from "@xyflow/react"
import { OfficeGrid } from "./OfficeGrid"
import { Desk } from "./Desk"
import { PERSONAS } from "../personas"

const DEFAULT_PERSONA = {
  name: "Node",
  role: "Generic",
  category: "observer" as const,
  color: "#6d5dfc",
}

const DESKS_PER_ROW = 4
const H_SPACING = 200
const V_SPACING = 120

function toIsometric(gridX: number, gridY: number): { x: number; y: number } {
  return {
    x: (gridX - gridY) * (H_SPACING / 2),
    y: (gridX + gridY) * (V_SPACING / 2),
  }
}

interface AgentOfficeProps {
  nodes: Node[]
  selectedNodeId: string | null
  onSelectNode: (id: string) => void
}

function getStatusMessage(
  persona: { idleMessage: string; workingMessages: string[]; doneMessage: string; errorMessage: string },
  status: string,
): string | undefined {
  if (status === "running") {
    const msgs = persona.workingMessages
    return msgs[Math.floor(Math.random() * msgs.length)]
  }
  if (status === "done") return persona.doneMessage
  if (status === "error") return persona.errorMessage
  return undefined
}

export function AgentOffice({ nodes, selectedNodeId, onSelectNode }: AgentOfficeProps) {
  const desks = useMemo(() => {
    return nodes.map((node, index) => {
      const personaKey = String(node.type || "")
      const persona = PERSONAS[personaKey] ?? DEFAULT_PERSONA
      const data = (node.data ?? {}) as Record<string, unknown>
      const status = String(data.status || "idle") as "idle" | "running" | "done" | "error"

      const gridX = index % DESKS_PER_ROW
      const gridY = Math.floor(index / DESKS_PER_ROW)
      const iso = toIsometric(gridX, gridY)

      const fullPersona = PERSONAS[personaKey]
      const message = fullPersona
        ? getStatusMessage(fullPersona, status)
        : undefined

      return {
        id: node.id,
        x: iso.x,
        y: iso.y,
        agentName: persona.name,
        agentRole: persona.role,
        color: persona.color,
        status,
        category: persona.category,
        selected: node.id === selectedNodeId,
        message: status !== "idle" ? message : undefined,
        index,
      }
    })
  }, [nodes, selectedNodeId])

  const viewBox = useMemo(() => {
    if (desks.length === 0) {
      return "-400 -200 800 600"
    }
    const xs = desks.map((d) => d.x)
    const ys = desks.map((d) => d.y)
    const minX = Math.min(...xs) - 160
    const maxX = Math.max(...xs) + 160
    const minY = Math.min(...ys) - 140
    const maxY = Math.max(...ys) + 140
    return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`
  }, [desks])

  return (
    <div className="flex-1 h-full bg-[#0a0a0b] overflow-hidden relative">
      <svg
        width="100%"
        height="100%"
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full"
      >
        <OfficeGrid />
        {desks.map((desk) => (
          <Desk
            key={desk.id}
            x={desk.x}
            y={desk.y}
            agentName={desk.agentName}
            agentRole={desk.agentRole}
            color={desk.color}
            status={desk.status}
            category={desk.category}
            selected={desk.selected}
            onClick={() => onSelectNode(desk.id)}
            message={desk.message}
            index={desk.index}
          />
        ))}
      </svg>
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">
            No agents in this workflow. Add nodes to see the office view.
          </p>
        </div>
      )}
    </div>
  )
}
