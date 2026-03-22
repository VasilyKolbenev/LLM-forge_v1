import { type EdgeProps, getBezierPath, useReactFlow } from "@xyflow/react"

function getEdgeStatus(
  sourceStatus: string | undefined,
  targetStatus: string | undefined,
): "idle" | "active" | "done" | "error" {
  if (sourceStatus === "error" || targetStatus === "error") return "error"
  if (sourceStatus === "running" || targetStatus === "running") return "active"
  if (sourceStatus === "done" && targetStatus === "done") return "done"
  if (sourceStatus === "done" || targetStatus === "done") return "done"
  return "idle"
}

export function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  source,
  target,
  style,
  markerEnd,
}: EdgeProps) {
  const { getNode } = useReactFlow()
  const sourceNode = getNode(source)
  const targetNode = getNode(target)

  const sourceStatus = sourceNode?.data?.status as string | undefined
  const targetStatus = targetNode?.data?.status as string | undefined
  const edgeStatus = getEdgeStatus(sourceStatus, targetStatus)

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  })

  const pathId = `edge-path-${id}`

  const strokeColor =
    edgeStatus === "error"
      ? "#ef4444"
      : edgeStatus === "active"
        ? "#6d5dfc"
        : edgeStatus === "done"
          ? "#22c55e"
          : "#52525b"

  const strokeDasharray =
    edgeStatus === "idle" ? "6 4" : edgeStatus === "error" ? "6 4" : undefined

  const strokeWidth = edgeStatus === "active" ? 2 : 1.5

  return (
    <g>
      <defs>
        <path id={pathId} d={edgePath} />
      </defs>

      {/* Main edge line */}
      <path
        d={edgePath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={strokeDasharray}
        style={style}
        markerEnd={markerEnd}
        className={edgeStatus === "done" ? "drop-shadow-[0_0_4px_rgba(34,197,94,0.5)]" : ""}
      />

      {/* Animated particles for active state */}
      {edgeStatus === "active" && (
        <>
          {[0, 33, 66].map((offset) => (
            <circle key={offset} r="3" fill={strokeColor} opacity="0.8">
              <animateMotion
                dur="2s"
                repeatCount="indefinite"
                begin={`${offset / 100 * 2}s`}
              >
                <mpath href={`#${pathId}`} />
              </animateMotion>
            </circle>
          ))}
        </>
      )}
    </g>
  )
}
