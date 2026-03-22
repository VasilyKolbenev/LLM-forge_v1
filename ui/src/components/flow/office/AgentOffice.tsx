import { useMemo } from "react"
import type { Node } from "@xyflow/react"
import { OfficeGrid } from "./OfficeGrid"
import { Desk } from "./Desk"
import { PERSONAS } from "../personas"
import type { CustomPersona } from "../PersonaEditor"
import type { OfficeEnvironment } from "../EnvironmentPicker"
import { EnvironmentPicker } from "../EnvironmentPicker"

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
  onDoubleClickNode?: (id: string) => void
  customPersonas?: Record<string, CustomPersona>
  environment?: OfficeEnvironment
  onEnvironmentChange?: (env: OfficeEnvironment) => void
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

export function AgentOffice({
  nodes,
  selectedNodeId,
  onSelectNode,
  onDoubleClickNode,
  customPersonas = {},
  environment = "modern-office",
  onEnvironmentChange,
}: AgentOfficeProps) {
  const desks = useMemo(() => {
    return nodes.map((node, index) => {
      const personaKey = String(node.type || "")
      const basePersona = PERSONAS[personaKey] ?? DEFAULT_PERSONA
      const custom = customPersonas[node.id]

      const mergedName = custom?.name ?? basePersona.name
      const mergedRole = custom?.role ?? basePersona.role
      const mergedColor = custom?.avatarColor ?? basePersona.color
      const mergedCategory = basePersona.category
      const avatarEmoji = custom?.avatarEmoji

      const data = (node.data ?? {}) as Record<string, unknown>
      const status = String(data.status || "idle") as "idle" | "running" | "done" | "error"

      const gridX = index % DESKS_PER_ROW
      const gridY = Math.floor(index / DESKS_PER_ROW)
      const iso = toIsometric(gridX, gridY)

      const fullPersona = PERSONAS[personaKey]
      const messagePersona = fullPersona
        ? {
            idleMessage: custom?.idleMessage ?? fullPersona.idleMessage,
            workingMessages: custom?.workingMessages ?? fullPersona.workingMessages,
            doneMessage: custom?.doneMessage ?? fullPersona.doneMessage,
            errorMessage: custom?.errorMessage ?? fullPersona.errorMessage,
          }
        : undefined
      const message = messagePersona ? getStatusMessage(messagePersona, status) : undefined

      return {
        id: node.id,
        x: iso.x,
        y: iso.y,
        agentName: mergedName,
        agentRole: mergedRole,
        color: mergedColor,
        status,
        category: mergedCategory,
        selected: node.id === selectedNodeId,
        message: status !== "idle" ? message : undefined,
        index,
        avatarEmoji,
      }
    })
  }, [nodes, selectedNodeId, customPersonas])

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
      {/* Environment picker */}
      {onEnvironmentChange && (
        <div className="absolute top-3 right-3 z-10">
          <EnvironmentPicker environment={environment} onChange={onEnvironmentChange} />
        </div>
      )}

      <svg
        width="100%"
        height="100%"
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full"
      >
        <OfficeGrid environment={environment} />
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
            onDoubleClick={onDoubleClickNode ? () => onDoubleClickNode(desk.id) : undefined}
            message={desk.message}
            index={desk.index}
            avatarEmoji={desk.avatarEmoji}
            environment={environment}
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
