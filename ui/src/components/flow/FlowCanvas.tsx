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

import { DataSourceNode } from "./nodes/DataSourceNode"
import { ModelNode } from "./nodes/ModelNode"
import { TrainingNode } from "./nodes/TrainingNode"
import { EvalNode } from "./nodes/EvalNode"
import { ExportNode } from "./nodes/ExportNode"
import { AgentNode } from "./nodes/AgentNode"
import { PromptNode } from "./nodes/PromptNode"
import { ConditionalNode } from "./nodes/ConditionalNode"
import type { useWorkflow } from "@/hooks/useWorkflow"

const nodeTypes = {
  dataSource: DataSourceNode,
  model: ModelNode,
  training: TrainingNode,
  eval: EvalNode,
  export: ExportNode,
  agent: AgentNode,
  prompt: PromptNode,
  conditional: ConditionalNode,
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
        fitView
        deleteKeyCode={["Backspace", "Delete"]}
        className="bg-background"
        defaultEdgeOptions={{ animated: true }}
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
            const colors: Record<string, string> = {
              dataSource: "#22c55e",
              model: "#3b82f6",
              training: "#6d5dfc",
              eval: "#eab308",
              export: "#ef4444",
              agent: "#8b5cf6",
              prompt: "#06b6d4",
              conditional: "#f97316",
            }
            return colors[n.type || ""] || "#6d5dfc"
          }}
        />
      </ReactFlow>
    </div>
  )
}
