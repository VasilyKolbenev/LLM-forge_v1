import { memo, useMemo } from "react"
import { Handle, Position, type NodeProps } from "@xyflow/react"
import { motion } from "framer-motion"
import { PERSONAS } from "../personas"
import { SpeechBubble } from "../SpeechBubble"
import type { CustomPersona } from "../PersonaEditor"

function buildConfigSummary(config: Record<string, unknown>): string {
  const parts: string[] = []
  for (const [key, val] of Object.entries(config)) {
    if (val == null || val === "" || typeof val === "object") continue
    const short = String(val).length > 12 ? String(val).slice(0, 12) + ".." : String(val)
    parts.push(short)
    if (parts.length >= 3) break
  }
  return parts.join(" \u00B7 ")
}

const CATEGORY_ICON: Record<string, string> = {
  scientist: "\u{1F9EA}",
  engineer: "\u{1F527}",
  devops: "\u{1F4BB}",
  architect: "\u{1F3D7}",
  security: "\u{1F6E1}",
  observer: "\u{1F50D}",
}

const DEFAULT_PERSONA = {
  name: "Node",
  role: "Generic",
  category: "observer" as const,
  color: "#6d5dfc",
  icon: "Box",
  idleMessage: "Idle...",
  workingMessages: ["Working..."],
  doneMessage: "Done!",
  errorMessage: "Error...",
  inputs: [{ id: "input", label: "input" }],
  outputs: [{ id: "output", label: "output" }],
}

function AgentPersonaNodeInner({ data, type: nodeType }: NodeProps) {
  const personaKey = String(data.type || nodeType || "")
  const basePersona = PERSONAS[personaKey] ?? DEFAULT_PERSONA
  const custom = (data.customPersona as CustomPersona | undefined) ?? undefined

  const persona = useMemo(() => {
    if (!custom) return basePersona
    return {
      ...basePersona,
      name: custom.name ?? basePersona.name,
      role: custom.role ?? basePersona.role,
      color: custom.avatarColor ?? basePersona.color,
      idleMessage: custom.idleMessage ?? basePersona.idleMessage,
      workingMessages: custom.workingMessages ?? basePersona.workingMessages,
      doneMessage: custom.doneMessage ?? basePersona.doneMessage,
      errorMessage: custom.errorMessage ?? basePersona.errorMessage,
    }
  }, [basePersona, custom])

  const avatarEmoji = custom?.avatarEmoji
  const replayStatus = data.replayStatus as
    | { status: string; message?: string; progress?: number }
    | undefined
  const status = replayStatus
    ? (String(replayStatus.status) as "idle" | "running" | "done" | "error")
    : (String(data.status || "idle") as "idle" | "running" | "done" | "error")
  const config = (data.config ?? {}) as Record<string, unknown>
  const progress = replayStatus?.progress
    ?? (typeof data.progress === "number" ? (data.progress as number) : undefined)

  const configSummary = useMemo(() => buildConfigSummary(config), [config])

  const speechMessages = useMemo(() => {
    if (replayStatus?.message) return [replayStatus.message]
    if (status === "running") return persona.workingMessages
    if (status === "done") return [persona.doneMessage]
    if (status === "error") return [persona.errorMessage]
    return [persona.idleMessage]
  }, [status, persona, replayStatus])

  const showBubble = status === "running" || status === "done" || status === "error"

  const borderClass =
    status === "done"
      ? "shadow-[0_0_12px_rgba(34,197,94,0.4)]"
      : status === "error"
        ? "border-red-500/60"
        : "border-border"

  return (
    <div className="relative">
      <SpeechBubble messages={speechMessages} status={status} visible={showBubble} />

      <motion.div
        className={`bg-card border rounded-lg shadow-lg w-[240px] ${borderClass}`}
        animate={
          status === "error"
            ? { x: [0, -3, 3, -3, 3, 0] }
            : undefined
        }
        transition={
          status === "error"
            ? { duration: 0.4, ease: "easeInOut" }
            : undefined
        }
      >
        {/* Header */}
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-t-lg border-b border-border"
          style={{ backgroundColor: `${persona.color}15` }}
        >
          {/* Avatar */}
          <motion.div
            className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-bold"
            style={{
              backgroundColor: `${persona.color}25`,
              color: persona.color,
              border: `2px solid ${persona.color}50`,
            }}
            animate={
              status === "idle"
                ? { opacity: [1, 0.6, 1] }
                : status === "running"
                  ? { scale: [1, 1.08, 1] }
                  : undefined
            }
            transition={
              status === "idle"
                ? { duration: 3, repeat: Infinity, ease: "easeInOut" }
                : status === "running"
                  ? { duration: 1.2, repeat: Infinity, ease: "easeInOut" }
                  : undefined
            }
          >
            {avatarEmoji || persona.name.charAt(0)}
          </motion.div>

          <div className="flex-1 min-w-0">
            <div className="text-xs font-semibold truncate">{persona.name}</div>
            <div className="text-[10px] text-muted-foreground truncate">{persona.role}</div>
          </div>

          <div className="text-xs" title={persona.category}>
            {CATEGORY_ICON[persona.category] ?? ""}
          </div>
        </div>

        {/* Body */}
        <div className="px-3 py-2 space-y-1.5">
          {configSummary && (
            <div className="text-[10px] text-muted-foreground truncate">
              {configSummary}
            </div>
          )}

          {/* Progress bar */}
          {status === "running" && (
            <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
              {progress != null ? (
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: persona.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                  transition={{ duration: 0.5 }}
                />
              ) : (
                <motion.div
                  className="h-full rounded-full w-1/3"
                  style={{ backgroundColor: persona.color }}
                  animate={{ x: ["0%", "200%", "0%"] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                />
              )}
            </div>
          )}
        </div>

        {/* Handles */}
        {persona.inputs.map((inp, i) => (
          <Handle
            key={inp.id}
            type="target"
            position={Position.Left}
            id={inp.id}
            style={{
              top: `${((i + 1) / (persona.inputs.length + 1)) * 100}%`,
              background: persona.color,
              width: 8,
              height: 8,
              border: "2px solid var(--color-card)",
            }}
          />
        ))}
        {persona.outputs.map((out, i) => (
          <Handle
            key={out.id}
            type="source"
            position={Position.Right}
            id={out.id}
            style={{
              top: `${((i + 1) / (persona.outputs.length + 1)) * 100}%`,
              background: persona.color,
              width: 8,
              height: 8,
              border: "2px solid var(--color-card)",
            }}
          />
        ))}
      </motion.div>
    </div>
  )
}

export const AgentPersonaNode = memo(AgentPersonaNodeInner)
