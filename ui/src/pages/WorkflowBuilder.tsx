import { useEffect, useState, useMemo } from "react"
import { ReactFlowProvider } from "@xyflow/react"
import { AnimatePresence } from "framer-motion"
import {
  Play,
  Save,
  Trash2,
  FolderOpen,
  Plus,
  Loader2,
  X,
  Landmark,
  History,
} from "lucide-react"
import { NodePalette } from "@/components/flow/NodePalette"
import { FlowCanvas } from "@/components/flow/FlowCanvas"
import { PropertiesPanel } from "@/components/flow/PropertiesPanel"
import { AgentOffice } from "@/components/flow/office/AgentOffice"
import { ViewToggle } from "@/components/flow/ViewToggle"
import { PersonaEditor } from "@/components/flow/PersonaEditor"
import { TraceReplay } from "@/components/flow/TraceReplay"
import { useWorkflow, type WorkflowMeta } from "@/hooks/useWorkflow"
import { useTraceReplay } from "@/hooks/useTraceReplay"

type ViewMode = "dag" | "office" | "split"

function WorkflowToolbar({
  workflow,
  onOpenLoad,
  viewMode,
  onViewModeChange,
  onReplay,
  isReplayMode,
}: {
  workflow: ReturnType<typeof useWorkflow>
  onOpenLoad: () => void
  viewMode: ViewMode
  onViewModeChange: (v: ViewMode) => void
  onReplay: () => void
  isReplayMode: boolean
}) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-card">
      <input
        type="text"
        value={workflow.workflowName}
        onChange={(e) => workflow.setWorkflowName(e.target.value)}
        className="bg-transparent text-sm font-medium focus:outline-none focus:border-b focus:border-primary w-48"
        placeholder="Workflow name..."
      />
      <ViewToggle view={viewMode} onChange={onViewModeChange} />
      <div className="flex-1" />
      <button
        onClick={() => workflow.loadBankingTemplate()}
        disabled={workflow.templateLoading}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-secondary hover:bg-secondary/80 transition-colors disabled:opacity-50"
      >
        {workflow.templateLoading ? <Loader2 size={14} className="animate-spin" /> : <Landmark size={14} />}
        Banking Template
      </button>
      <button
        onClick={onOpenLoad}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-secondary hover:bg-secondary/80 transition-colors"
      >
        <FolderOpen size={14} />
        Load
      </button>
      <button
        onClick={() => workflow.save()}
        disabled={workflow.saving}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-secondary hover:bg-secondary/80 transition-colors disabled:opacity-50"
      >
        {workflow.saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
        Save
      </button>
      <button
        onClick={onReplay}
        disabled={workflow.nodes.length === 0 || isReplayMode}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-secondary hover:bg-secondary/80 transition-colors disabled:opacity-50"
      >
        <History size={14} />
        Replay
      </button>
      <button
        onClick={() => workflow.run()}
        disabled={workflow.running || workflow.nodes.length === 0 || isReplayMode}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {workflow.running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
        Run
      </button>
      <button
        onClick={workflow.clearCanvas}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded text-destructive hover:bg-destructive/10 transition-colors"
      >
        <Trash2 size={14} />
      </button>
    </div>
  )
}

function LoadModal({
  workflows,
  onLoad,
  onDelete,
  onClose,
  onNew,
}: {
  workflows: WorkflowMeta[]
  onLoad: (id: string) => void
  onDelete: (id: string) => void
  onClose: () => void
  onNew: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-card border border-border rounded-lg w-[440px] max-h-[500px] flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h2 className="text-sm font-semibold">Saved Workflows</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {workflows.length === 0 ? (
            <div className="text-center py-8 text-sm text-muted-foreground">No saved workflows yet</div>
          ) : (
            <div className="space-y-1">
              {workflows.map((wf) => (
                <div
                  key={wf.id}
                  className="flex items-center gap-2 px-3 py-2 rounded hover:bg-secondary group cursor-pointer"
                  onClick={() => {
                    onLoad(wf.id)
                    onClose()
                  }}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{wf.name}</div>
                    <div className="text-[10px] text-muted-foreground">
                      Updated {new Date(wf.updated_at).toLocaleDateString()}
                      {wf.run_count > 0 && ` · ${wf.run_count} runs`}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(wf.id)
                    }}
                    className="opacity-0 group-hover:opacity-100 text-destructive hover:text-destructive/80 transition-opacity"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="border-t border-border px-4 py-3">
          <button
            onClick={() => {
              onNew()
              onClose()
            }}
            className="flex items-center gap-1.5 text-xs text-primary hover:text-primary/80"
          >
            <Plus size={14} />
            New Workflow
          </button>
        </div>
      </div>
    </div>
  )
}

export function WorkflowBuilder() {
  const workflow = useWorkflow()
  const [showLoad, setShowLoad] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>("dag")
  const [personaEditorNodeId, setPersonaEditorNodeId] = useState<string | null>(null)
  const replay = useTraceReplay(workflow.nodes, workflow.edges)

  useEffect(() => {
    workflow.loadList().then((list) => {
      const items = list as Array<{ id: string }>
      if (items.length > 0 && workflow.nodes.length === 0) {
        workflow.load(items[0].id)
      }
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleOpenLoad = async () => {
    await workflow.loadList()
    setShowLoad(true)
  }

  const handleOfficeSelectNode = (id: string) => {
    const node = workflow.nodes.find((n) => n.id === id) ?? null
    workflow.setSelectedNode(node)
  }

  const handleNodeDoubleClick = (nodeId: string) => {
    setPersonaEditorNodeId(nodeId)
  }

  const personaEditorNode = personaEditorNodeId
    ? workflow.nodes.find((n) => n.id === personaEditorNodeId)
    : null

  const showDag = viewMode === "dag" || viewMode === "split"
  const showOffice = viewMode === "office" || viewMode === "split"

  // When replay is active, override node statuses with replay data
  const replayWorkflow = useMemo(() => {
    if (!replay.isReplayMode || Object.keys(replay.nodeStatuses).length === 0) {
      return workflow
    }
    // Create a modified workflow with replay statuses injected into node data
    const modifiedNodes = workflow.nodes.map((n) => {
      const rs = replay.nodeStatuses[n.id]
      if (!rs) return n
      return {
        ...n,
        data: {
          ...n.data,
          replayStatus: rs,
        },
      }
    })
    return { ...workflow, nodes: modifiedNodes }
  }, [workflow.nodes, replay.isReplayMode, replay.nodeStatuses])

  return (
    <div className="h-[calc(100vh-1rem)] flex flex-col -m-6">
      <WorkflowToolbar
        workflow={workflow}
        onOpenLoad={handleOpenLoad}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        onReplay={replay.enterReplay}
        isReplayMode={replay.isReplayMode}
      />
      {workflow.runError && (
        <div className="mx-4 mt-3 px-3 py-2 rounded border border-destructive/30 bg-destructive/10 text-destructive text-xs">
          {workflow.runError}
        </div>
      )}
      <div className="flex flex-1 min-h-0 relative">
        <NodePalette />
        <div className={`flex flex-1 min-h-0 ${viewMode === "split" ? "gap-1" : ""}`}>
          {showDag && (
            <div className={`${viewMode === "split" ? "w-1/2" : "flex-1"} min-h-0 relative`}>
              <ReactFlowProvider>
                <FlowCanvas
                  workflow={replayWorkflow}
                  onNodeDoubleClick={handleNodeDoubleClick}
                  customPersonas={workflow.customPersonas}
                />
              </ReactFlowProvider>
            </div>
          )}
          {showOffice && (
            <div className={`${viewMode === "split" ? "w-1/2" : "flex-1"} min-h-0 relative`}>
              <AgentOffice
                nodes={workflow.nodes}
                selectedNodeId={workflow.selectedNode?.id ?? null}
                onSelectNode={handleOfficeSelectNode}
                onDoubleClickNode={handleNodeDoubleClick}
                customPersonas={workflow.customPersonas}
                environment={workflow.officeEnvironment}
                onEnvironmentChange={workflow.setOfficeEnvironment}
                replayNodeStatuses={replay.isReplayMode ? replay.nodeStatuses : undefined}
              />
            </div>
          )}
        </div>
        {workflow.selectedNode && (
          <PropertiesPanel
            node={workflow.selectedNode}
            onClose={() => workflow.setSelectedNode(null)}
            onUpdate={workflow.updateNodeData}
          />
        )}
        <AnimatePresence>
          {replay.isReplayMode && (
            <TraceReplay
              trace={replay.trace}
              isPlaying={replay.isPlaying}
              currentIndex={replay.currentIndex}
              speed={replay.speed}
              onPlay={replay.play}
              onPause={replay.pause}
              onStepForward={replay.stepForward}
              onSeek={replay.seek}
              onSpeedChange={replay.setSpeed}
              onClose={replay.exitReplay}
              currentTime={replay.currentTime}
              totalTime={replay.totalTime}
            />
          )}
        </AnimatePresence>
      </div>

      {showLoad && (
        <LoadModal
          workflows={workflow.savedWorkflows}
          onLoad={(id) => workflow.load(id)}
          onDelete={(id) => workflow.deleteWorkflow(id)}
          onClose={() => setShowLoad(false)}
          onNew={workflow.clearCanvas}
        />
      )}

      {personaEditorNode && (
        <PersonaEditor
          nodeId={personaEditorNode.id}
          currentType={personaEditorNode.type || ""}
          customPersona={workflow.customPersonas[personaEditorNode.id]}
          onSave={(nodeId, persona) => {
            if (Object.keys(persona).length === 0) {
              workflow.clearCustomPersona(nodeId)
            } else {
              workflow.setCustomPersona(nodeId, persona)
            }
            setPersonaEditorNodeId(null)
          }}
          onClose={() => setPersonaEditorNodeId(null)}
        />
      )}
    </div>
  )
}
