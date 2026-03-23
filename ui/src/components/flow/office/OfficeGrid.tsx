import { memo } from "react"
import { motion } from "framer-motion"
import type { OfficeEnvironment } from "../EnvironmentPicker"

interface OfficeGridProps {
  environment?: OfficeEnvironment
}

interface EnvTheme {
  accent: string
  accentRgb: string
  secondary: string
  floorBase: string
}

const THEMES: Record<OfficeEnvironment, EnvTheme> = {
  "modern-office": { accent: "#06b6d4", accentRgb: "6,182,212", secondary: "#8b5cf6", floorBase: "#08090c" },
  lab:             { accent: "#22c55e", accentRgb: "34,197,94", secondary: "#0ea5e9", floorBase: "#060c08" },
  "command-center":{ accent: "#8b5cf6", accentRgb: "139,92,246", secondary: "#ec4899", floorBase: "#0a0810" },
  "server-room":   { accent: "#ef4444", accentRgb: "239,68,68", secondary: "#f97316", floorBase: "#0c0808" },
  "open-space":    { accent: "#f59e0b", accentRgb: "245,158,11", secondary: "#84cc16", floorBase: "#0c0a06" },
}

const ROWS = 10
const TILE = 100
const TH = 60

function OfficeGridInner({ environment = "modern-office" }: OfficeGridProps) {
  const t = THEMES[environment] ?? THEMES["modern-office"]

  const left = -ROWS * TILE
  const right = ROWS * TILE
  const top = -TH
  const midY = ROWS * TH
  const wallH = 100

  return (
    <g>
      {/* SVG filter defs */}
      <defs>
        <radialGradient id="floorGlow" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor={t.accent} stopOpacity={0.06} />
          <stop offset="100%" stopColor={t.accent} stopOpacity={0} />
        </radialGradient>
        <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Dark floor */}
      <polygon
        points={`${left},${midY} 0,${top} ${right},${midY} 0,${midY * 2 - top}`}
        fill={t.floorBase}
      />

      {/* Neon grid lines */}
      <g opacity={0.3}>
        {Array.from({ length: 13 }, (_, i) => {
          const r = i - 1
          return (
            <line key={`a-${i}`}
              x1={-r * TILE} y1={r * TH}
              x2={(ROWS - r) * TILE} y2={(ROWS + r) * TH}
              stroke={t.accent} strokeWidth={0.4} />
          )
        })}
        {Array.from({ length: 13 }, (_, i) => {
          const c = i - 1
          return (
            <line key={`b-${i}`}
              x1={c * TILE} y1={c * TH}
              x2={(c - ROWS) * TILE} y2={(c + ROWS) * TH}
              stroke={t.accent} strokeWidth={0.4} />
          )
        })}
      </g>

      {/* Center glow */}
      <ellipse cx={0} cy={midY * 0.4} rx={250} ry={100} fill="url(#floorGlow)" />

      {/* Pulsing center hex */}
      <motion.polygon
        points={`0,${midY * 0.4 - 40} 35,${midY * 0.4 - 20} 35,${midY * 0.4 + 20} 0,${midY * 0.4 + 40} -35,${midY * 0.4 + 20} -35,${midY * 0.4 - 20}`}
        fill="none" stroke={t.accent} strokeWidth={1}
        animate={{ opacity: [0.1, 0.25, 0.1] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* Walls */}
      <polygon points={`${left},${midY} 0,${top} 0,${top - wallH} ${left},${midY - wallH}`}
        fill="#0c0e14" stroke={`${t.accent}40`} strokeWidth={0.8} opacity={0.5} />
      <polygon points={`0,${top} ${right},${midY} ${right},${midY - wallH} 0,${top - wallH}`}
        fill="#0c0e14" stroke={`${t.accent}40`} strokeWidth={0.8} opacity={0.5} />

      {/* Neon ceiling lines */}
      <line x1={left} y1={midY - wallH} x2={0} y2={top - wallH}
        stroke={t.accent} strokeWidth={1.5} opacity={0.4} filter="url(#neonGlow)" />
      <line x1={0} y1={top - wallH} x2={right} y2={midY - wallH}
        stroke={t.accent} strokeWidth={1.5} opacity={0.4} filter="url(#neonGlow)" />

      {/* Corner glow */}
      <circle cx={0} cy={top - wallH} r={6} fill={t.accent} opacity={0.5} filter="url(#neonGlow)" />

      {/* Wall hologram panels */}
      {[0.3, 0.65].map((pos, i) => {
        const wx = left + (0 - left) * pos
        const wy = midY + (top - midY) * pos
        return (
          <g key={`hl-${i}`} opacity={0.4}>
            <rect x={wx - 25} y={wy - wallH + 15} width={50} height={35} rx={2}
              fill={`rgba(${t.accentRgb}, 0.05)`} stroke={t.accent} strokeWidth={0.5} />
            <path
              d={`M ${wx - 20} ${wy - wallH + 32} Q ${wx - 10} ${wy - wallH + 25} ${wx} ${wy - wallH + 32} Q ${wx + 10} ${wy - wallH + 39} ${wx + 20} ${wy - wallH + 32}`}
              fill="none" stroke={t.accent} strokeWidth={0.8} opacity={0.5} />
            <text x={wx} y={wy - wallH + 22} textAnchor="middle" fill={t.accent} fontSize={5} opacity={0.5}>
              {i === 0 ? "NEURAL LOAD" : "DATA FLOW"}
            </text>
          </g>
        )
      })}

      {[0.35, 0.7].map((pos, i) => {
        const wx = right * pos
        const wy = top + (midY - top) * pos
        return (
          <g key={`hr-${i}`} opacity={0.4}>
            <rect x={wx - 25} y={wy - wallH + 15} width={50} height={35} rx={2}
              fill={`rgba(${t.accentRgb}, 0.05)`} stroke={t.accent} strokeWidth={0.5} />
            {[0, 1, 2, 3, 4].map((j) => (
              <rect key={j} x={wx - 18 + j * 8} y={wy - wallH + 30 - j * 2}
                width={5} height={8 + j * 3} rx={0.5} fill={t.accent} opacity={0.3} />
            ))}
            <text x={wx} y={wy - wallH + 22} textAnchor="middle" fill={t.accent} fontSize={5} opacity={0.5}>
              {i === 0 ? "AGENTS ACTIVE" : "THROUGHPUT"}
            </text>
          </g>
        )
      })}

      {/* Floating particles — simple, no complex animations */}
      {Array.from({ length: 20 }, (_, i) => {
        const px = (i * 137.5 % 800) - 400
        const py = (i * 97.3 % 600) - 100
        return (
          <motion.circle key={`p-${i}`}
            cx={px} cy={py} r={1 + (i % 3) * 0.4}
            fill={i % 3 === 0 ? t.accent : t.secondary}
            animate={{ opacity: [0.1, 0.35, 0.1] }}
            transition={{ duration: 5 + (i % 5), repeat: Infinity, ease: "easeInOut", delay: i * 0.3 }}
          />
        )
      })}

      {/* Floor running data lines */}
      <motion.line
        x1={left + 80} y1={midY - 20} x2={-20} y2={top + 20}
        stroke={t.accent} strokeWidth={1} strokeDasharray="8 20"
        animate={{ strokeDashoffset: [0, -56], opacity: [0.05, 0.15, 0.05] }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
      />
      <motion.line
        x1={20} y1={top + 20} x2={right - 80} y2={midY - 20}
        stroke={t.secondary} strokeWidth={1} strokeDasharray="8 20"
        animate={{ strokeDashoffset: [0, -56], opacity: [0.05, 0.15, 0.05] }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear", delay: 1 }}
      />
    </g>
  )
}

export const OfficeGrid = memo(OfficeGridInner)
