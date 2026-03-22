import { useCallback, useMemo, useRef, type DragEvent } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { AgentPersonaNode } from "./nodes/AgentPersonaNode"
import { AnimatedEdge } from "./AnimatedEdge"
import { PERSONAS } from "./personas"
import type { useWorkflow } from "@/hooks/useWorkflow"
import type { CustomPersona } from "./PersonaEditor"

const nodeTypes = {
  dataSource: AgentPersonaNode,
  model: AgentPersonaNode,
  training: AgentPersonaNode,
  eval: AgentPersonaNode,
  export: AgentPersonaNode,
  agent: AgentPersonaNode,
  prompt: AgentPersonaNode,
  conditional: AgentPersonaNode,
  rag: AgentPersonaNode,
  inference: AgentPersonaNode,
  router: AgentPersonaNode,
  dataGen: AgentPersonaNode,
  serve: AgentPersonaNode,
  splitter: AgentPersonaNode,
  mcp: AgentPersonaNode,
  a2a: AgentPersonaNode,
  gateway: AgentPersonaNode,
  inputGuard: AgentPersonaNode,
  outputGuard: AgentPersonaNode,
  llmJudge: AgentPersonaNode,
  abTest: AgentPersonaNode,
  cache: AgentPersonaNode,
  canary: AgentPersonaNode,
  feedback: AgentPersonaNode,
  tracer: AgentPersonaNode,
  collectTraces: AgentPersonaNode,
  buildDataset: AgentPersonaNode,
  regressionEval: AgentPersonaNode,
  group: AgentPersonaNode,
}

const edgeTypes = {
  default: AnimatedEdge,
}

type WorkflowHook = ReturnType<typeof useWorkflow>

interface FlowCanvasProps {
  workflow: WorkflowHook
  onNodeDoubleClick?: (nodeId: string) => void
  customPersonas?: Record<string, CustomPersona>
}

export function FlowCanvas({ workflow, onNodeDoubleClick, customPersonas = {} }: FlowCanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const rfInstance = useRef<ReactFlowInstance | null>(null)

  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
  }, [])

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      const type = e.dataTransfer.getData("application/reactflow-type")
      const label = e.dataTransfer.getData("application/reactflow-label")
      if (!type || !rfInstance.current || !reactFlowWrapper.current) return

      const bounds = reactFlowWrapper.current.getBoundingClientRect()
      const position = rfInstance.current.screenToFlowPosition({
        x: e.clientX - bounds.left,
        y: e.clientY - bounds.top,
      })

      workflow.addNode(type, label, position)
    },
    [workflow]
  )

  const handleNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      onNodeDoubleClick?.(node.id)
    },
    [onNodeDoubleClick]
  )

  const nodesWithPersonas = useMemo(() => {
    if (Object.keys(customPersonas).length === 0) return workflow.nodes
    return workflow.nodes.map((n) => {
      const cp = customPersonas[n.id]
      if (!cp) return n
      return { ...n, data: { ...n.data, customPersona: cp } }
    })
  }, [workflow.nodes, customPersonas])

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow
        nodes={nodesWithPersonas}
        edges={workflow.edges}
        onNodesChange={workflow.onNodesChange}
        onEdgesChange={workflow.onEdgesChange}
        onConnect={workflow.onConnect}
        onNodeClick={workflow.onNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        onPaneClick={() => workflow.setSelectedNode(null)}
        onInit={(instance) => { rfInstance.current = instance }}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        deleteKeyCode={["Backspace", "Delete"]}
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
          nodeColor={(n) => {
            const persona = PERSONAS[n.type || ""]
            return persona?.color || "#6d5dfc"
          }}
        />
      </ReactFlow>
    </div>
  )
}
