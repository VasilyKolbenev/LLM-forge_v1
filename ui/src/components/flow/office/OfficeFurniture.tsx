import { memo } from "react"
import { motion } from "framer-motion"
import type { OfficeEnvironment } from "../EnvironmentPicker"

interface FurnitureProps {
  x: number
  y: number
  environment: OfficeEnvironment
}

interface EnvAccent { main: string; rgb: string; secondary: string }

const ACCENTS: Record<OfficeEnvironment, EnvAccent> = {
  "modern-office": { main: "#06b6d4", rgb: "6,182,212", secondary: "#8b5cf6" },
  lab:             { main: "#22c55e", rgb: "34,197,94", secondary: "#0ea5e9" },
  "command-center":{ main: "#8b5cf6", rgb: "139,92,246", secondary: "#ec4899" },
  "server-room":   { main: "#ef4444", rgb: "239,68,68", secondary: "#f97316" },
  "open-space":    { main: "#f59e0b", rgb: "245,158,11", secondary: "#84cc16" },
}

// ===== OFFICE CHAIR — minimal neon outline =====
export function OfficeChair({ x, y, chairColor = "#333" }: { x: number; y: number; chairColor?: string }) {
  return (
    <g transform={`translate(${x}, ${y})`} opacity={0.6}>
      {/* Base */}
      <line x1={-5} y1={10} x2={5} y2={10} stroke="#444" strokeWidth={0.8} />
      <circle cx={-5} cy={10.5} r={1} fill="#333" />
      <circle cx={5} cy={10.5} r={1} fill="#333" />
      {/* Cylinder */}
      <rect x={-1} y={3} width={2} height={7} fill="#444" />
      {/* Seat outline */}
      <polygon points="-8,0 8,0 6,4 -6,4" fill={`${chairColor}40`} stroke={chairColor} strokeWidth={0.4} />
      {/* Back outline */}
      <polygon points="-7,-12 7,-12 8,0 -8,0" fill={`${chairColor}30`} stroke={chairColor} strokeWidth={0.4} />
    </g>
  )
}

export const CHAIR_COLORS: Record<OfficeEnvironment, string> = {
  "modern-office": "#06b6d4",
  lab: "#22c55e",
  "server-room": "#ef4444",
  "command-center": "#8b5cf6",
  "open-space": "#f59e0b",
}

// ===== HOLOGRAPHIC DESK =====
function HoloDesk({ x, y, accent }: { x: number; y: number; accent: EnvAccent }) {
  const c = accent.main
  const rgb = accent.rgb

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Desk surface — transparent with neon edges */}
      <polygon points="0,-14 50,11 0,36 -50,11"
        fill={`rgba(${rgb}, 0.03)`} stroke={c} strokeWidth={0.8} opacity={0.6} />
      {/* Desk front face — subtle */}
      <polygon points="-50,11 0,36 0,42 -50,17"
        fill={`rgba(${rgb}, 0.02)`} stroke={c} strokeWidth={0.3} opacity={0.3} />
      <polygon points="0,36 50,11 50,17 0,42"
        fill={`rgba(${rgb}, 0.02)`} stroke={c} strokeWidth={0.3} opacity={0.3} />

      {/* Neon edge glow on surface */}
      <motion.polygon points="0,-14 50,11 0,36 -50,11"
        fill="none" stroke={c} strokeWidth={1.5} opacity={0.15}
        filter="url(#neonGlow)"
        animate={{ opacity: [0.1, 0.2, 0.1] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      {/* Holographic monitor — FLOATING above desk */}
      <g transform="translate(5, -28)">
        {/* Monitor frame — thin neon */}
        <rect x={-14} y={-18} width={28} height={18} rx={1}
          fill={`rgba(${rgb}, 0.06)`} stroke={c} strokeWidth={0.6} opacity={0.7} />
        {/* Scanline effect — simple opacity pulse */}
        <motion.rect x={-14} y={-18} width={28} height={18}
          fill={c} opacity={0.02}
          animate={{ opacity: [0.01, 0.04, 0.01] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        {/* Code lines — glowing */}
        <motion.line x1={-10} y1={-14} x2={0} y2={-14} stroke={c} strokeWidth={0.8} opacity={0.5}
          animate={{ opacity: [0.3, 0.7, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }} />
        <motion.line x1={-10} y1={-10} x2={5} y2={-10} stroke={accent.secondary} strokeWidth={0.8} opacity={0.4}
          animate={{ opacity: [0.2, 0.6, 0.2] }}
          transition={{ duration: 2.5, repeat: Infinity, delay: 0.3 }} />
        <motion.line x1={-10} y1={-6} x2={3} y2={-6} stroke={c} strokeWidth={0.8} opacity={0.3}
          animate={{ opacity: [0.2, 0.5, 0.2] }}
          transition={{ duration: 3, repeat: Infinity, delay: 0.6 }} />
        {/* Float indicator — thin line to desk */}
        <line x1={0} y1={0} x2={0} y2={8} stroke={c} strokeWidth={0.3} opacity={0.2} strokeDasharray="2 2" />
      </g>

      {/* Holographic keyboard — glowing dots on desk */}
      <g transform="translate(-8, 8)" opacity={0.4}>
        {[0, 3, 6].map((dy) => (
          <g key={dy}>
            {[-8, -5, -2, 1, 4, 7].map((dx) => (
              <motion.circle key={dx} cx={dx} cy={dy} r={0.8}
                fill={c} opacity={0.5}
                animate={{ opacity: [0.3, 0.7, 0.3] }}
                transition={{ duration: 1 + Math.abs(dx) * 0.1, repeat: Infinity, delay: dy * 0.1 }}
              />
            ))}
          </g>
        ))}
      </g>

      {/* Data crystal — glowing cube */}
      <g transform="translate(28, 2)">
        <motion.g animate={{ y: [0, -3, 0], rotate: [0, 5, 0] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}>
          <polygon points="0,-6 5,-3 0,0 -5,-3" fill={`rgba(${rgb}, 0.2)`} stroke={c} strokeWidth={0.5} />
          <polygon points="-5,-3 0,0 0,4 -5,1" fill={`rgba(${rgb}, 0.1)`} stroke={c} strokeWidth={0.3} />
          <polygon points="0,0 5,-3 5,1 0,4" fill={`rgba(${rgb}, 0.15)`} stroke={c} strokeWidth={0.3} />
        </motion.g>
        {/* Crystal glow */}
        <circle cx={0} cy={-2} r={8} fill={c} opacity={0.04} />
      </g>

      {/* Mini hologram dashboard */}
      <g transform="translate(-28, -2)" opacity={0.35}>
        <rect x={-8} y={-8} width={16} height={10} rx={1}
          fill={`rgba(${rgb}, 0.05)`} stroke={c} strokeWidth={0.3} />
        {/* Mini bar chart */}
        {[0, 1, 2, 3].map((i) => (
          <motion.rect key={i} x={-6 + i * 4} y={0} width={2} height={0}
            fill={c} opacity={0.5}
            animate={{ height: [3 + i, 6 + i, 3 + i], y: [-3 - i, -6 - i, -3 - i] }}
            transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </g>

      {/* Ground glow under desk */}
      <ellipse cx={0} cy={38} rx={40} ry={12}
        fill={c} opacity={0.03} />
    </g>
  )
}

function OfficeFurnitureInner({ x, y, environment }: FurnitureProps) {
  const accent = ACCENTS[environment] ?? ACCENTS["modern-office"]
  return <HoloDesk x={x} y={y} accent={accent} />
}

export const OfficeFurniture = memo(OfficeFurnitureInner)
