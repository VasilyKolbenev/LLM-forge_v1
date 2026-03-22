import { memo } from "react"
import { motion } from "framer-motion"

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
  message?: string
  index: number
}

const STATUS_LAMP: Record<string, string> = {
  idle: "#3f3f46",
  running: "#3b82f6",
  done: "#22c55e",
  error: "#ef4444",
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
  message,
  index,
}: DeskProps) {
  const lampColor = STATUS_LAMP[status] ?? STATUS_LAMP.idle

  return (
    <motion.g
      transform={`translate(${x}, ${y})`}
      onClick={onClick}
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
        fill="#1c1c1e"
        stroke={selected ? color : "#27272a"}
        strokeWidth={selected ? 1.5 : 1}
      />
      {/* Desk front face */}
      <polygon
        points="-80,20 0,60 0,72 -80,32"
        fill="#161618"
        stroke="#27272a"
        strokeWidth={0.5}
      />
      {/* Desk right face */}
      <polygon
        points="0,60 80,20 80,32 0,72"
        fill="#141416"
        stroke="#27272a"
        strokeWidth={0.5}
      />

      {/* Monitor — small isometric rectangle on desk */}
      <polygon
        points="0,-18 30,-3 0,12 -30,-3"
        fill="#0a0a0b"
        stroke="#27272a"
        strokeWidth={0.5}
      />
      {/* Monitor screen bezel */}
      <polygon
        points="-10,-40 20,-25 20,-5 -10,-20"
        fill="#18181b"
        stroke="#27272a"
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
      <motion.circle
        cx={0}
        cy={-60}
        r={18}
        fill={`${color}25`}
        stroke={color}
        strokeWidth={2}
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
      />
      {/* Agent initial letter */}
      <text
        x={0}
        y={-55}
        textAnchor="middle"
        fill={color}
        fontSize={14}
        fontWeight="bold"
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        {agentName.charAt(0)}
      </text>

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
            fill="#1c1c1e"
            stroke="#27272a"
            strokeWidth={0.5}
          />
          {/* Bubble tail */}
          <polygon
            points="-4,-79 4,-79 0,-73"
            fill="#1c1c1e"
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
