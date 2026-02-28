import { useState, useCallback } from "react"
import {
  useNodesState,
  useEdgesState,
  addEdge,
  type Node,
  type Edge,
  type Connection,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react"
import { api } from "@/api/client"

export interface WorkflowMeta {
  id: string
  name: string
  created_at: string
  updated_at: string
  run_count: number
  last_run: string | null
}

export function useWorkflow() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [workflowId, setWorkflowId] = useState<string | null>(null)
  const [workflowName, setWorkflowName] = useState("Untitled Workflow")
  const [savedWorkflows, setSavedWorkflows] = useState<WorkflowMeta[]>([])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const [saving, setSaving] = useState(false)
  const [running, setRunning] = useState(false)

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, animated: true }, eds))
    },
    [setEdges]
  )

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => setSelectedNode(node),
    []
  )

  const updateNodeData = useCallback(
    (nodeId: string, data: Record<string, unknown>) => {
      setNodes((nds) =>
        nds.map((n) => (n.id === nodeId ? { ...n, data } : n))
      )
      setSelectedNode((prev) => (prev?.id === nodeId ? { ...prev, data } : prev))
    },
    [setNodes]
  )

  const addNode = useCallback(
    (type: string, label: string, position: { x: number; y: number }) => {
      const id = `${type}_${Date.now()}`
      const newNode: Node = {
        id,
        type,
        position,
        data: { label, config: {}, status: "idle" },
      }
      setNodes((nds) => [...nds, newNode])
    },
    [setNodes]
  )

  const clearCanvas = useCallback(() => {
    setNodes([])
    setEdges([])
    setWorkflowId(null)
    setWorkflowName("Untitled Workflow")
    setSelectedNode(null)
  }, [setNodes, setEdges])

  const save = useCallback(async () => {
    setSaving(true)
    try {
      const result = await api.saveWorkflow({
        name: workflowName,
        nodes: nodes.map((n) => ({
          id: n.id,
          type: n.type,
          position: n.position,
          data: n.data,
        })),
        edges: edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle,
          targetHandle: e.targetHandle,
        })),
        workflow_id: workflowId ?? undefined,
      })
      setWorkflowId(result.id as string)
      return result
    } finally {
      setSaving(false)
    }
  }, [workflowName, nodes, edges, workflowId])

  const load = useCallback(
    async (id: string) => {
      const wf = await api.getWorkflow(id) as {
        id: string; name: string;
        nodes: Array<Record<string, unknown>>;
        edges: Array<Record<string, unknown>>;
      }
      setWorkflowId(wf.id)
      setWorkflowName(wf.name)
      setNodes(
        wf.nodes.map((n: Record<string, unknown>) => ({
          id: n.id as string,
          type: n.type as string,
          position: n.position as { x: number; y: number },
          data: n.data as Record<string, unknown>,
        }))
      )
      setEdges(
        wf.edges.map((e: Record<string, unknown>) => ({
          id: e.id as string,
          source: e.source as string,
          target: e.target as string,
          sourceHandle: e.sourceHandle as string | undefined,
          targetHandle: e.targetHandle as string | undefined,
          animated: true,
        }))
      )
      setSelectedNode(null)
    },
    [setNodes, setEdges]
  )

  const loadList = useCallback(async () => {
    const list = await api.listWorkflows()
    setSavedWorkflows(list as unknown as WorkflowMeta[])
    return list
  }, [])

  const run = useCallback(async () => {
    if (!workflowId) {
      const saved = await save()
      if (!saved) return null
    }
    setRunning(true)
    try {
      return await api.runWorkflow(workflowId!)
    } finally {
      setRunning(false)
    }
  }, [workflowId, save])

  const deleteWorkflow = useCallback(
    async (id: string) => {
      await api.deleteWorkflow(id)
      if (workflowId === id) clearCanvas()
      await loadList()
    },
    [workflowId, clearCanvas, loadList]
  )

  return {
    // State
    nodes,
    edges,
    workflowId,
    workflowName,
    savedWorkflows,
    selectedNode,
    saving,
    running,
    // Setters
    setWorkflowName,
    setSelectedNode,
    // Node/edge handlers
    onNodesChange: onNodesChange as OnNodesChange,
    onEdgesChange: onEdgesChange as OnEdgesChange,
    onConnect,
    onNodeClick,
    updateNodeData,
    addNode,
    // Workflow operations
    clearCanvas,
    save,
    load,
    loadList,
    run,
    deleteWorkflow,
  }
}
