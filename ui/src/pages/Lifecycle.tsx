import { useState, useEffect, useCallback, useMemo } from "react"
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  useReactFlow,
  type Node,
  type Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { motion } from "framer-motion"
import { GitBranch, Loader2 } from "lucide-react"
import { LineageNode, type LineageNodeData } from "@/components/lineage/LineageNode"
import { Timeline, type TimelineEvent } from "@/components/lineage/Timeline"
import { AnimatedEdge } from "@/components/flow/AnimatedEdge"
import { api } from "@/api/client"

const nodeTypes = { lineage: LineageNode }
const edgeTypes = { default: AnimatedEdge }

const TYPE_COLUMN: Record<string, number> = {
  dataset: 0,
  experiment: 300,
  model: 600,
  deployment: 900,
  traces: 900,
  feedback: 1200,
}

const TYPE_COLOR: Record<string, string> = {
  dataset: "#14b8a6",
  experiment: "#6d5dfc",
  model: "#3b82f6",
  deployment: "#ef4444",
  traces: "#6366f1",
  feedback: "#0ea5e9",
}

function layoutLineageNodes(
  rawNodes: Array<{ id: string; data: LineageNodeData }>,
): Node[] {
  const groups: Record<string, Array<{ id: string; data: LineageNodeData }>> = {}

  for (const n of rawNodes) {
    const t = n.data.type ?? "model"
    if (!groups[t]) groups[t] = []
    groups[t].push(n)
  }

  const positioned: Node[] = []
  for (const [type, nodes] of Object.entries(groups)) {
    const x = TYPE_COLUMN[type] ?? 600
    nodes.forEach((n, i) => {
      positioned.push({
        id: n.id,
        type: "lineage",
        position: { x, y: i * 150 },
        data: n.data,
      })
    })
  }

  return positioned
}

function LifecycleInner() {
  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState<LineageNodeData | null>(null)

  const { fitView, setCenter } = useReactFlow()

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const [graphRes, timelineRes] = await Promise.all([
        api.getLineageGraph(),
        api.getLineageTimeline(),
      ])

      const layouted = layoutLineageNodes(graphRes.nodes)
      setNodes(layouted)
      setEdges(
        graphRes.edges.map((e: { id: string; source: string; target: string }) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          type: "default",
          animated: true,
        })),
      )
      setEvents(timelineRes.events)

      setTimeout(() => fitView({ padding: 0.2 }), 100)
    } catch {
      /* API not available yet */
    } finally {
      setLoading(false)
    }
  }, [fitView])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node.data as unknown as LineageNodeData)
    },
    [],
  )

  const handleTimelineClick = useCallback(
    (event: TimelineEvent) => {
      if (!event.entity_id) return
      const node = nodes.find((n) => n.id === event.entity_id)
      if (node) {
        setCenter(node.position.x + 90, node.position.y + 40, {
          zoom: 1.5,
          duration: 600,
        })
        setSelectedNode(node.data as unknown as LineageNodeData)
      }
    },
    [nodes, setCenter],
  )

  const minimapNodeColor = useCallback((n: Node) => {
    const data = n.data as unknown as LineageNodeData
    return TYPE_COLOR[data.type] ?? "#6d5dfc"
  }, [])

  return (
    <div className="space-y-6 h-full flex flex-col">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <GitBranch size={24} /> Lifecycle
        </h2>
        <p className="text-muted-foreground text-sm mt-1">
          Model and data lineage across the full training lifecycle
        </p>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Graph */}
        <div className="flex-1 border border-border rounded-lg overflow-hidden relative">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
              <Loader2 size={24} className="animate-spin text-muted-foreground" />
            </div>
          )}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodeClick={handleNodeClick}
            onPaneClick={() => setSelectedNode(null)}
            fitView
            deleteKeyCode={[]}
            className="bg-background"
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="var(--color-border)"
            />
            <Controls
              className="!bg-card !border-border !shadow-lg [&>button]:!bg-card [&>button]:!border-border [&>button]:!fill-foreground [&>button:hover]:!bg-secondary"
            />
            <MiniMap
              className="!bg-card !border-border"
              maskColor="rgba(0,0,0,0.6)"
              nodeColor={minimapNodeColor}
            />
          </ReactFlow>
        </div>

        {/* Timeline sidebar */}
        <div className="w-72 shrink-0 border border-border rounded-lg bg-card flex flex-col">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold">Timeline</h3>
            <p className="text-[10px] text-muted-foreground">
              {events.length} events
            </p>
          </div>
          <div className="flex-1 min-h-0 px-2">
            <Timeline events={events} onEventClick={handleTimelineClick} />
          </div>
        </div>
      </div>

      {/* Detail panel */}
      {selectedNode && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="border border-border rounded-lg bg-card p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold">{selectedNode.label}</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Close
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-muted-foreground">Type</span>
              <p className="font-medium capitalize">{selectedNode.type}</p>
            </div>
            {selectedNode.status && (
              <div>
                <span className="text-muted-foreground">Status</span>
                <p className="font-medium capitalize">{selectedNode.status}</p>
              </div>
            )}
            {selectedNode.metadata &&
              Object.entries(selectedNode.metadata).map(([key, value]) => (
                <div key={key}>
                  <span className="text-muted-foreground">{key}</span>
                  <p className="font-medium">{String(value)}</p>
                </div>
              ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

export function Lifecycle() {
  return (
    <ReactFlowProvider>
      <LifecycleInner />
    </ReactFlowProvider>
  )
}
