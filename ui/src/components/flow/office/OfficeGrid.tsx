import { memo } from "react"

const TILE_ROWS = 12
const TILE_COLS = 12

function OfficeGridInner() {
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
          fill={isDark ? "#0e0e10" : "#111113"}
          stroke="#1a1a1d"
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
