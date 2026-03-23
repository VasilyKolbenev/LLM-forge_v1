import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Canvas } from "@react-three/fiber"
import type { Node } from "@xyflow/react"
import { PERSONAS } from "../personas"
import type { CustomPersona } from "../PersonaEditor"
import type { OfficeEnvironment } from "../EnvironmentPicker"
import { EnvironmentPicker } from "../EnvironmentPicker"
import { Scene } from "./three/Scene"
import { gridTo3D, CHAR_OFFSET as CHAR_OFFSET_3D } from "./three/constants"
import type { AgentData3D, WalkerData3D, CharacterState } from "./three/types"

const DEFAULT_PERSONA = { name: "Node", role: "Generic", category: "observer" as const, color: "#6d5dfc" }

function colorToHex(color: string): number {
  return parseInt(color.replace("#", ""), 16)
}

interface NodeReplayStatus { status: "idle" | "running" | "done" | "error"; message?: string; progress?: number }

interface AgentOfficeProps {
  nodes: Node[]
  selectedNodeId: string | null
  onSelectNode: (id: string) => void
  onDoubleClickNode?: (id: string) => void
  customPersonas?: Record<string, CustomPersona>
  environment?: OfficeEnvironment
  onEnvironmentChange?: (env: OfficeEnvironment) => void
  replayNodeStatuses?: Record<string, NodeReplayStatus>
}

function mapState(status: "idle" | "running" | "done" | "error"): CharacterState {
  return status === "running" ? "working"
    : status === "done" ? "celebrating"
    : status === "error" ? "error"
    : "idle"
}

interface Walker {
  id: string; agentIdx: number; targetIdx: number
  name: string; color: string; category: string
  fromX: number; fromZ: number; toX: number; toZ: number
  phase: "going" | "delivering" | "returning"; progress: number
}

export function AgentOffice({
  nodes, selectedNodeId, onSelectNode, onDoubleClickNode,
  customPersonas = {}, environment = "modern-office",
  onEnvironmentChange, replayNodeStatuses,
}: AgentOfficeProps) {

  // ============================================================
  // AGENTS DATA (same logic as before, but 3D coordinates)
  // ============================================================
  const agents = useMemo((): AgentData3D[] => {
    return nodes.map((node, index) => {
      const personaKey = String(node.type || "")
      const base = PERSONAS[personaKey] ?? DEFAULT_PERSONA
      const custom = customPersonas[node.id]
      const data = (node.data ?? {}) as Record<string, unknown>
      const replayInfo = replayNodeStatuses?.[node.id]
      const status = replayInfo?.status ?? (String(data.status || "idle") as "idle" | "running" | "done" | "error")
      const pos = gridTo3D(index, nodes.length)
      const color = custom?.avatarColor ?? base.color
      return {
        id: node.id, index,
        worldX: pos.x, worldY: pos.y, worldZ: pos.z,
        name: custom?.name ?? base.name,
        color,
        colorHex: colorToHex(color),
        category: base.category,
        characterState: mapState(status),
        selected: node.id === selectedNodeId,
      }
    })
  }, [nodes, selectedNodeId, customPersonas, replayNodeStatuses])

  // ============================================================
  // DEMO MODE
  // ============================================================
  const [demoActive, setDemoActive] = useState(false)
  const [demoOverrides, setDemoOverrides] = useState<Record<string, CharacterState>>({})
  const [walkers, setWalkers] = useState<Walker[]>([])
  const demoTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  const walkingAgentIndices = useMemo(() => new Set(walkers.map((w) => w.agentIdx)), [walkers])

  const startDemo = useCallback(() => {
    setDemoActive(true)
    let step = 0
    const ids = agents.map((a) => a.id)
    if (ids.length === 0) return

    demoTimer.current = setInterval(() => {
      step++
      const overrides: Record<string, CharacterState> = {}
      for (let i = 0; i < ids.length; i++) {
        const s = step - i * 5
        if (s < 0) overrides[ids[i]] = "idle"
        else if (s < 4) overrides[ids[i]] = "working"
        else if (s < 5) overrides[ids[i]] = "celebrating"
        else if (s === 8 && i === 2) overrides[ids[i]] = "error"
        else overrides[ids[i]] = "idle"
      }
      setDemoOverrides(overrides)

      // Spawn walkers
      setWalkers((prev) => {
        const next = [...prev]
        for (let i = 0; i < ids.length - 1; i++) {
          const s = step - i * 5
          if (s === 5 && !next.find((w) => w.id === `w-${step}-${i}`)) {
            const from = agents[i], to = agents[i + 1]
            if (from && to) {
              next.push({
                id: `w-${step}-${i}`, agentIdx: i, targetIdx: i + 1,
                name: from.name, color: from.color, category: from.category,
                fromX: from.worldX + CHAR_OFFSET_3D.x,
                fromZ: from.worldZ + CHAR_OFFSET_3D.z,
                toX: to.worldX + CHAR_OFFSET_3D.x + 0.5,
                toZ: to.worldZ + CHAR_OFFSET_3D.z + 0.3,
                phase: "going", progress: 0,
              })
            }
          }
        }
        return next.map((w) => {
          const spd = 0.07
          if (w.phase === "going") {
            const p = Math.min(w.progress + spd, 1)
            return p >= 1 ? { ...w, phase: "delivering" as const, progress: 0 } : { ...w, progress: p }
          }
          if (w.phase === "delivering") {
            const p = w.progress + 0.12
            return p >= 1 ? { ...w, phase: "returning" as const, progress: 0 } : { ...w, progress: p }
          }
          const p = Math.min(w.progress + spd, 1)
          return p >= 1 ? null as unknown as Walker : { ...w, progress: p }
        }).filter(Boolean)
      })

      if (step > ids.length * 5 + 12) { step = 0; setWalkers([]) }
    }, 500)
  }, [agents])

  const stopDemo = useCallback(() => {
    setDemoActive(false); setDemoOverrides({}); setWalkers([])
    if (demoTimer.current) clearInterval(demoTimer.current)
  }, [])

  useEffect(() => () => { if (demoTimer.current) clearInterval(demoTimer.current) }, [])

  // ============================================================
  // EFFECTIVE AGENTS WITH DEMO OVERRIDES
  // ============================================================
  const effectiveAgents = useMemo((): AgentData3D[] => {
    if (!demoActive) return agents
    return agents.map((a) => ({
      ...a,
      characterState: demoOverrides[a.id] ?? a.characterState,
    }))
  }, [agents, demoActive, demoOverrides])

  // ============================================================
  // WALKER 3D DATA
  // ============================================================
  const walkers3D = useMemo((): WalkerData3D[] => {
    return walkers.map((w) => {
      let wx: number, wz: number, state: CharacterState
      if (w.phase === "going") {
        wx = w.fromX + (w.toX - w.fromX) * w.progress
        wz = w.fromZ + (w.toZ - w.fromZ) * w.progress
        state = "walking"
      } else if (w.phase === "delivering") {
        wx = w.toX; wz = w.toZ; state = "delivering"
      } else {
        wx = w.toX + (w.fromX - w.toX) * w.progress
        wz = w.toZ + (w.fromZ - w.toZ) * w.progress
        state = "walking"
      }
      return {
        id: w.id, agentIdx: w.agentIdx, targetIdx: w.targetIdx,
        name: w.name, color: w.color, colorHex: colorToHex(w.color),
        category: w.category, state,
        x: wx, y: 0, z: wz,
        phase: w.phase,
      }
    })
  }, [walkers])

  return (
    <div className="w-full h-full bg-[#050810] overflow-hidden relative">
      {/* HTML OVERLAY — Demo button */}
      <div className="absolute top-3 left-3 z-10 flex gap-2">
        <button onClick={demoActive ? stopDemo : startDemo}
          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
            demoActive ? "bg-red-500/20 text-red-400 border border-red-500/30" : "bg-primary/20 text-primary border border-primary/30"
          }`}>
          {demoActive ? "⏹ Stop Demo" : "▶ Run Demo"}
        </button>
      </div>

      {/* HTML OVERLAY — Environment picker */}
      {onEnvironmentChange && (
        <div className="absolute top-3 right-3 z-10">
          <EnvironmentPicker environment={environment} onChange={onEnvironmentChange} />
        </div>
      )}

      {/* 3D CANVAS */}
      <Canvas
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        dpr={[1, 2]}
        style={{ width: "100%", height: "100%" }}
        onPointerMissed={() => {}}
      >
        <Suspense fallback={null}>
          <Scene
            agents={effectiveAgents}
            walkers={walkers3D}
            environment={environment}
            selectedNodeId={selectedNodeId}
            onSelectNode={onSelectNode}
            onDoubleClickNode={onDoubleClickNode}
            walkingAgentIndices={walkingAgentIndices}
          />
        </Suspense>
      </Canvas>

      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">No agents in this workflow. Add nodes to see the office view.</p>
        </div>
      )}
    </div>
  )
}
