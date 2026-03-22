import { useCallback, useRef, type DragEvent } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { AgentPersonaNode } from "./nodes/AgentPersonaNode"
import { AnimatedEdge } from "./AnimatedEdge"
import { PERSONAS } from "./personas"
import type { useWorkflow } from "@/hooks/useWorkflow"

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
  group: AgentPersonaNode,
}

const edgeTypes = {
  default: AnimatedEdge,
}

type WorkflowHook = ReturnType<typeof useWorkflow>

interface FlowCanvasProps {
  workflow: WorkflowHook
}

export function FlowCanvas({ workflow }: FlowCanvasProps) {
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

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow
        nodes={workflow.nodes}
        edges={workflow.edges}
        onNodesChange={workflow.onNodesChange}
        onEdgesChange={workflow.onEdgesChange}
        onConnect={workflow.onConnect}
        onNodeClick={workflow.onNodeClick}
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
