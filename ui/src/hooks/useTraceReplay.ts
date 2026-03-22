import { useState, useCallback, useRef, useEffect } from "react"
import type { Node, Edge } from "@xyflow/react"
import { api } from "@/api/client"

export interface TraceEvent {
  timestamp: number
  nodeId: string
  type: "start" | "progress" | "complete" | "error" | "message"
  message?: string
  progress?: number
  duration?: number
  metadata?: Record<string, unknown>
}

export interface NodeReplayStatus {
  status: "idle" | "running" | "done" | "error"
  message?: string
  progress?: number
}

export interface UseTraceReplayReturn {
  trace: TraceEvent[]
  isPlaying: boolean
  currentIndex: number
  speed: number
  nodeStatuses: Record<string, NodeReplayStatus>
  play: () => void
  pause: () => void
  stepForward: () => void
  seek: (index: number) => void
  setSpeed: (speed: number) => void
  loadTrace: (workflowId: string) => Promise<void>
  isReplayMode: boolean
  enterReplay: () => void
  exitReplay: () => void
  currentTime: number
  totalTime: number
}

/**
 * Sort nodes in topological order using edges (Kahn's algorithm).
 */
function topologicalSort(nodes: Node[], edges: Edge[]): Node[] {
  const adjList = new Map<string, string[]>()
  const inDegree = new Map<string, number>()

  for (const node of nodes) {
    adjList.set(node.id, [])
    inDegree.set(node.id, 0)
  }

  for (const edge of edges) {
    const neighbors = adjList.get(edge.source)
    if (neighbors) neighbors.push(edge.target)
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1)
  }

  const queue: string[] = []
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id)
  }

  const sorted: string[] = []
  while (queue.length > 0) {
    const current = queue.shift()!
    sorted.push(current)
    for (const neighbor of adjList.get(current) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1
      inDegree.set(neighbor, newDeg)
      if (newDeg === 0) queue.push(neighbor)
    }
  }

  const orderMap = new Map(sorted.map((id, i) => [id, i]))
  return [...nodes].sort((a, b) => (orderMap.get(a.id) ?? 0) - (orderMap.get(b.id) ?? 0))
}

const WORKING_MESSAGES: string[][] = [
  ["Analyzing input data...", "Processing records...", "Validating schema..."],
  ["Running inference...", "Computing embeddings...", "Generating response..."],
  ["Checking compliance rules...", "Applying policies...", "Verifying constraints..."],
  ["Routing request...", "Evaluating conditions...", "Selecting pathway..."],
  ["Aggregating results...", "Building report...", "Finalizing output..."],
]

/**
 * Generate mock trace data from workflow structure for MVP.
 */
export function generateMockTrace(nodes: Node[], edges: Edge[]): TraceEvent[] {
  const sorted = topologicalSort(nodes, edges)
  const events: TraceEvent[] = []
  let cumulativeTime = 0

  for (let i = 0; i < sorted.length; i++) {
    const node = sorted[i]
    const nodeLabel = String((node.data as Record<string, unknown>)?.label ?? node.id)
    const nodeDuration = 2000 + Math.random() * 4000 // 2-6s per node
    const msgs = WORKING_MESSAGES[i % WORKING_MESSAGES.length]

    events.push({
      timestamp: cumulativeTime,
      nodeId: node.id,
      type: "start",
      message: `${nodeLabel}: Starting...`,
    })

    const progressSteps = 2 + Math.floor(Math.random() * 2) // 2-3 progress events
    for (let p = 1; p <= progressSteps; p++) {
      const progressTime = cumulativeTime + (nodeDuration * p) / (progressSteps + 1)
      events.push({
        timestamp: Math.round(progressTime),
        nodeId: node.id,
        type: "progress",
        progress: p / (progressSteps + 1),
        message: msgs[(p - 1) % msgs.length],
      })
    }

    events.push({
      timestamp: Math.round(cumulativeTime + nodeDuration),
      nodeId: node.id,
      type: "complete",
      message: `${nodeLabel}: Done!`,
      duration: nodeDuration / 1000,
    })

    cumulativeTime += nodeDuration + 300 // small gap between nodes
  }

  return events.sort((a, b) => a.timestamp - b.timestamp)
}

/**
 * Compute node statuses from trace events up to a given index.
 */
function computeNodeStatuses(
  trace: TraceEvent[],
  upToIndex: number,
): Record<string, NodeReplayStatus> {
  const statuses: Record<string, NodeReplayStatus> = {}

  for (let i = 0; i <= upToIndex && i < trace.length; i++) {
    const event = trace[i]
    switch (event.type) {
      case "start":
        statuses[event.nodeId] = { status: "running", message: event.message }
        break
      case "progress":
        statuses[event.nodeId] = {
          status: "running",
          message: event.message,
          progress: event.progress != null ? event.progress * 100 : undefined,
        }
        break
      case "complete":
        statuses[event.nodeId] = { status: "done", message: event.message }
        break
      case "error":
        statuses[event.nodeId] = { status: "error", message: event.message }
        break
      case "message":
        if (statuses[event.nodeId]) {
          statuses[event.nodeId] = { ...statuses[event.nodeId], message: event.message }
        }
        break
    }
  }

  return statuses
}

export function useTraceReplay(
  nodes: Node[],
  edges: Edge[],
): UseTraceReplayReturn {
  const [trace, setTrace] = useState<TraceEvent[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(-1)
  const [speed, setSpeed] = useState(1)
  const [isReplayMode, setIsReplayMode] = useState(false)
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, NodeReplayStatus>>({})
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const totalTime = trace.length > 0 ? trace[trace.length - 1].timestamp : 0
  const currentTime = currentIndex >= 0 && currentIndex < trace.length
    ? trace[currentIndex].timestamp
    : 0

  const clearTimer = useCallback(() => {
    if (timerRef.current != null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const updateStatuses = useCallback(
    (index: number) => {
      setNodeStatuses(computeNodeStatuses(trace, index))
    },
    [trace],
  )

  const scheduleNext = useCallback(
    (fromIndex: number) => {
      if (fromIndex >= trace.length - 1) {
        setIsPlaying(false)
        return
      }

      const currentTs = trace[fromIndex]?.timestamp ?? 0
      const nextTs = trace[fromIndex + 1]?.timestamp ?? 0
      const delay = Math.max(50, (nextTs - currentTs) / speed)

      timerRef.current = setTimeout(() => {
        const nextIndex = fromIndex + 1
        setCurrentIndex(nextIndex)
        updateStatuses(nextIndex)
        scheduleNext(nextIndex)
      }, delay)
    },
    [trace, speed, updateStatuses],
  )

  // When playing state or speed changes, restart scheduling
  useEffect(() => {
    clearTimer()
    if (isPlaying && currentIndex < trace.length - 1) {
      scheduleNext(currentIndex)
    }
    return clearTimer
  }, [isPlaying, speed]) // eslint-disable-line react-hooks/exhaustive-deps

  const play = useCallback(() => {
    if (trace.length === 0) return
    if (currentIndex >= trace.length - 1) {
      // Restart from beginning
      setCurrentIndex(0)
      updateStatuses(0)
      setIsPlaying(true)
    } else {
      setIsPlaying(true)
    }
  }, [trace, currentIndex, updateStatuses])

  const pause = useCallback(() => {
    setIsPlaying(false)
    clearTimer()
  }, [clearTimer])

  const stepForward = useCallback(() => {
    if (currentIndex >= trace.length - 1) return
    pause()
    const nextIndex = currentIndex + 1
    setCurrentIndex(nextIndex)
    updateStatuses(nextIndex)
  }, [currentIndex, trace.length, pause, updateStatuses])

  const seek = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, trace.length - 1))
      setCurrentIndex(clamped)
      updateStatuses(clamped)
      if (isPlaying) {
        clearTimer()
        scheduleNext(clamped)
      }
    },
    [trace.length, isPlaying, clearTimer, scheduleNext, updateStatuses],
  )

  const loadTrace = useCallback(
    async (workflowId: string) => {
      try {
        const data = await api.getPipelineTrace(workflowId) as {
          events?: TraceEvent[]
        }
        if (data.events && data.events.length > 0) {
          setTrace(data.events)
        } else {
          // Fallback to mock generation
          setTrace(generateMockTrace(nodes, edges))
        }
      } catch {
        // Backend not available, generate mock
        setTrace(generateMockTrace(nodes, edges))
      }
      setCurrentIndex(-1)
      setNodeStatuses({})
      setIsPlaying(false)
    },
    [nodes, edges],
  )

  const enterReplay = useCallback(() => {
    setIsReplayMode(true)
    const mockTrace = generateMockTrace(nodes, edges)
    setTrace(mockTrace)
    setCurrentIndex(-1)
    setNodeStatuses({})
    setIsPlaying(false)
  }, [nodes, edges])

  const exitReplay = useCallback(() => {
    pause()
    setIsReplayMode(false)
    setTrace([])
    setCurrentIndex(-1)
    setNodeStatuses({})
  }, [pause])

  return {
    trace,
    isPlaying,
    currentIndex,
    speed,
    nodeStatuses,
    play,
    pause,
    stepForward,
    seek,
    setSpeed,
    loadTrace,
    isReplayMode,
    enterReplay,
    exitReplay,
    currentTime,
    totalTime,
  }
}
