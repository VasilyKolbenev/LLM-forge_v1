import { memo } from "react"
import { motion } from "framer-motion"
import type { OfficeEnvironment } from "../EnvironmentPicker"
import {
  ScientistAvatar,
  EngineerAvatar,
  DevopsAvatar,
  ArchitectAvatar,
  SecurityAvatar,
  ObserverAvatar,
} from "../avatars"

const AVATAR_MAP: Record<string, React.FC<{ size?: number; color?: string }>> = {
  scientist: ScientistAvatar,
  engineer: EngineerAvatar,
  devops: DevopsAvatar,
  architect: ArchitectAvatar,
  security: SecurityAvatar,
  observer: ObserverAvatar,
}

interface DeskProps {
  x: number
  y: number
  agentName: string
  agentRole: string
  color: string
  status: "idle" | "running" | "done" | "error"
  category: string
  selected: boolean
  onClick: () => void
  onDoubleClick?: () => void
  message?: string
  index: number
  avatarEmoji?: string
  environment?: OfficeEnvironment
}

const STATUS_LAMP: Record<string, string> = {
  idle: "#3f3f46",
  running: "#3b82f6",
  done: "#22c55e",
  error: "#ef4444",
}

interface DeskStyle {
  surface: string
  front: string
  right: string
  stroke: string
  monitor: string
  monitorBezel: string
}

const ENV_DESK_STYLES: Record<OfficeEnvironment, DeskStyle> = {
  "modern-office": {
    surface: "#1c1c1e",
    front: "#161618",
    right: "#141416",
    stroke: "#27272a",
    monitor: "#0a0a0b",
    monitorBezel: "#18181b",
  },
  lab: {
    surface: "#141e28",
    front: "#0e1820",
    right: "#0c151c",
    stroke: "#1e3040",
    monitor: "#081018",
    monitorBezel: "#0e1a28",
  },
  "server-room": {
    surface: "#1e1416",
    front: "#180e10",
    right: "#150c0e",
    stroke: "#302020",
    monitor: "#100808",
    monitorBezel: "#1a1012",
  },
  "command-center": {
    surface: "#16162a",
    front: "#101024",
    right: "#0e0e20",
    stroke: "#202040",
    monitor: "#08081a",
    monitorBezel: "#121228",
  },
  "open-space": {
    surface: "#1e1f18",
    front: "#181914",
    right: "#151610",
    stroke: "#2a2c22",
    monitor: "#0c0d08",
    monitorBezel: "#1a1c14",
  },
}

function DeskInner({
  x,
  y,
  agentName,
  agentRole,
  color,
  status,
  selected,
  onClick,
  onDoubleClick,
  message,
  index,
  avatarEmoji,
  environment = "modern-office",
}: DeskProps) {
  const lampColor = STATUS_LAMP[status] ?? STATUS_LAMP.idle
  const style = ENV_DESK_STYLES[environment] ?? ENV_DESK_STYLES["modern-office"]

  const avatarDisplay = avatarEmoji || agentName.charAt(0)

  return (
    <motion.g
      transform={`translate(${x}, ${y})`}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      style={{ cursor: "pointer" }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08, ease: "easeOut" }}
    >
      {/* Selected glow */}
      {selected && (
        <motion.ellipse
          cx={0}
          cy={30}
          rx={70}
          ry={35}
          fill="none"
          stroke={color}
          strokeWidth={2}
          opacity={0.5}
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
      )}

      {/* Desk surface — isometric parallelogram */}
      <polygon
        points="0,-20 80,20 0,60 -80,20"
        fill={style.surface}
        stroke={selected ? color : style.stroke}
        strokeWidth={selected ? 1.5 : 1}
      />
      {/* Desk front face */}
      <polygon
        points="-80,20 0,60 0,72 -80,32"
        fill={style.front}
        stroke={style.stroke}
        strokeWidth={0.5}
      />
      {/* Desk right face */}
      <polygon
        points="0,60 80,20 80,32 0,72"
        fill={style.right}
        stroke={style.stroke}
        strokeWidth={0.5}
      />

      {/* Monitor — small isometric rectangle on desk */}
      <polygon
        points="0,-18 30,-3 0,12 -30,-3"
        fill={style.monitor}
        stroke={style.stroke}
        strokeWidth={0.5}
      />
      {/* Monitor screen bezel */}
      <polygon
        points="-10,-40 20,-25 20,-5 -10,-20"
        fill={style.monitorBezel}
        stroke={style.stroke}
        strokeWidth={0.5}
      />
      {/* Monitor screen */}
      <polygon
        points="-7,-37 17,-24 17,-8 -7,-21"
        fill={`${color}15`}
        stroke={`${color}30`}
        strokeWidth={0.5}
      />

      {/* Status lamp on desk */}
      <motion.circle
        cx={40}
        cy={8}
        r={4}
        fill={lampColor}
        animate={
          status === "running"
            ? { opacity: [1, 0.3, 1], r: [4, 5, 4] }
            : undefined
        }
        transition={
          status === "running"
            ? { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
            : undefined
        }
      />
      {/* Lamp glow */}
      {status !== "idle" && (
        <circle
          cx={40}
          cy={8}
          r={8}
          fill={lampColor}
          opacity={0.15}
        />
      )}

      {/* Agent avatar */}
      <motion.g
        animate={
          status === "running"
            ? { scale: [1, 1.06, 1] }
            : status === "idle"
              ? { opacity: [1, 0.7, 1] }
              : undefined
        }
        transition={
          status === "running"
            ? { duration: 1.2, repeat: Infinity, ease: "easeInOut" }
            : status === "idle"
              ? { duration: 3, repeat: Infinity, ease: "easeInOut" }
              : undefined
        }
      >
        <circle cx={0} cy={-60} r={20} fill={`${color}15`} stroke={color} strokeWidth={1.5} />
        <foreignObject x={-24} y={-84} width={48} height={48}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 48, height: 48 }}>
            {(() => {
              const AvatarComponent = AVATAR_MAP[category] ?? ObserverAvatar
              return <AvatarComponent size={48} color={color} />
            })()}
          </div>
        </foreignObject>
      </motion.g>

      {/* Speech bubble */}
      {message && (
        <motion.g
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <rect
            x={-60}
            y={-105}
            width={120}
            height={26}
            rx={6}
            fill={style.surface}
            stroke={style.stroke}
            strokeWidth={0.5}
          />
          {/* Bubble tail */}
          <polygon
            points="-4,-79 4,-79 0,-73"
            fill={style.surface}
          />
          <text
            x={0}
            y={-88}
            textAnchor="middle"
            fill="#a1a1aa"
            fontSize={9}
            style={{ pointerEvents: "none", userSelect: "none" }}
          >
            {message.length > 22 ? message.slice(0, 22) + ".." : message}
          </text>
        </motion.g>
      )}

      {/* Name label */}
      <text
        x={0}
        y={90}
        textAnchor="middle"
        fill="#e4e4e7"
        fontSize={11}
        fontWeight="600"
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        {agentName}
      </text>
      {/* Role label */}
      <text
        x={0}
        y={104}
        textAnchor="middle"
        fill="#71717a"
        fontSize={9}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        {agentRole}
      </text>
    </motion.g>
  )
}

export const Desk = memo(DeskInner)
