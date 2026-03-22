import { memo } from "react"
import type { OfficeEnvironment } from "../EnvironmentPicker"

const TILE_ROWS = 12
const TILE_COLS = 12

interface OfficeGridProps {
  environment?: OfficeEnvironment
}

interface EnvColors {
  dark: string
  light: string
  stroke: string
}

const ENV_COLORS: Record<OfficeEnvironment, EnvColors> = {
  "modern-office": { dark: "#0e0e10", light: "#111113", stroke: "#1a1a1d" },
  lab: { dark: "#0a1015", light: "#0e1520", stroke: "#152030" },
  "server-room": { dark: "#100a0a", light: "#150e0e", stroke: "#201515" },
  "command-center": { dark: "#0a0a12", light: "#0e0e18", stroke: "#151525" },
  "open-space": { dark: "#0f100a", light: "#13150e", stroke: "#1e2015" },
}

function OfficeGridInner({ environment = "modern-office" }: OfficeGridProps) {
  const colors = ENV_COLORS[environment] ?? ENV_COLORS["modern-office"]
  const tiles: JSX.Element[] = []

  for (let row = 0; row < TILE_ROWS; row++) {
    for (let col = 0; col < TILE_COLS; col++) {
      const cx = (col - row) * 100
      const cy = (col + row) * 60
      const isDark = (row + col) % 2 === 0
      tiles.push(
        <polygon
          key={`${row}-${col}`}
          points={`${cx},${cy - 60} ${cx + 100},${cy} ${cx},${cy + 60} ${cx - 100},${cy}`}
          fill={isDark ? colors.dark : colors.light}
          stroke={colors.stroke}
          strokeWidth={0.5}
        />
      )
    }
  }

  return (
    <g opacity={0.6}>
      {tiles}
    </g>
  )
}

export const OfficeGrid = memo(OfficeGridInner)
