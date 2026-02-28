import { useState, useEffect, useCallback } from "react"
import {
  Plus,
  Trash2,
  Tag,
  History,
  Play,
  Save,
  ArrowLeft,
  ChevronRight,
  GitCompare,
  X,
} from "lucide-react"
import { api } from "@/api/client"

interface PromptVersion {
  version: number
  system_prompt: string
  variables: string[]
  model: string
  parameters: Record<string, unknown>
  created_at: string
  metrics: Record<string, unknown> | null
}

interface PromptItem {
  id: string
  name: string
  description: string
  current_version: number
  versions: PromptVersion[]
  tags: string[]
  created_at: string
  updated_at: string
}

/* ── Prompt List View ─────────────────────────────────────────── */

function PromptCard({
  prompt,
  onSelect,
  onDelete,
}: {
  prompt: PromptItem
  onSelect: () => void
  onDelete: () => void
}) {
  return (
    <div
      onClick={onSelect}
      className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors cursor-pointer group"
    >
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h3 className="font-medium text-sm truncate">{prompt.name}</h3>
          {prompt.description && (
            <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
              {prompt.description}
            </p>
          )}
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="opacity-0 group-hover:opacity-100 text-destructive hover:text-destructive/80 transition-opacity ml-2"
        >
          <Trash2 size={14} />
        </button>
      </div>
      <div className="flex items-center gap-2 mt-3">
        <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded">
          v{prompt.current_version}
        </span>
        {prompt.tags.map((t) => (
          <span
            key={t}
            className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded"
          >
            {t}
          </span>
        ))}
        <span className="flex-1" />
        <span className="text-[10px] text-muted-foreground">
          {new Date(prompt.updated_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  )
}

/* ── Create/New Modal ──────────────────────────────────────────── */

function CreatePromptModal({
  onClose,
  onCreate,
}: {
  onClose: () => void
  onCreate: (data: { name: string; system_prompt: string; description: string; tags: string[] }) => void
}) {
  const [name, setName] = useState("")
  const [desc, setDesc] = useState("")
  const [text, setText] = useState("")
  const [tags, setTags] = useState("")

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-card border border-border rounded-lg w-[520px] shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h2 className="text-sm font-semibold">New Prompt</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X size={16} />
          </button>
        </div>
        <div className="p-4 space-y-3">
          <div>
            <label className="text-[10px] font-medium text-muted-foreground uppercase">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Agent Prompt"
              className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div>
            <label className="text-[10px] font-medium text-muted-foreground uppercase">Description</label>
            <input
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="Optional description..."
              className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
          <div>
            <label className="text-[10px] font-medium text-muted-foreground uppercase">
              System Prompt
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={"You are {{role}}. Help the user with {{task}}."}
              rows={6}
              className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
            <p className="text-[10px] text-muted-foreground mt-1">
              Use {"{{variable}}"} for template variables
            </p>
          </div>
          <div>
            <label className="text-[10px] font-medium text-muted-foreground uppercase">
              Tags (comma-separated)
            </label>
            <input
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="agent, production"
              className="w-full mt-1 px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-border">
          <button onClick={onClose} className="px-3 py-1.5 text-xs rounded hover:bg-secondary">
            Cancel
          </button>
          <button
            disabled={!name || !text}
            onClick={() => onCreate({
              name,
              system_prompt: text,
              description: desc,
              tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
            })}
            className="px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </div>
    </div>
  )
}

/* ── Prompt Detail / Editor ────────────────────────────────────── */

function PromptDetail({
  prompt,
  onBack,
  onRefresh,
}: {
  prompt: PromptItem
  onBack: () => void
  onRefresh: () => void
}) {
  const latest = prompt.versions[prompt.versions.length - 1]
  const [editText, setEditText] = useState(latest.system_prompt)
  const [testVars, setTestVars] = useState<Record<string, string>>({})
  const [testResult, setTestResult] = useState<string | null>(null)
  const [showDiff, setShowDiff] = useState(false)
  const [diffV1, setDiffV1] = useState(Math.max(1, prompt.current_version - 1))
  const [diffV2, setDiffV2] = useState(prompt.current_version)
  const [diffText, setDiffText] = useState("")
  const [saving, setSaving] = useState(false)

  const handleSaveVersion = async () => {
    if (editText === latest.system_prompt) return
    setSaving(true)
    try {
      await api.addPromptVersion(prompt.id, { system_prompt: editText })
      onRefresh()
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    const resp = await api.testPrompt(prompt.id, { variables: testVars }) as {
      rendered: string; variables_missing: string[]
    }
    setTestResult(resp.rendered)
  }

  const handleDiff = async () => {
    const d = await api.diffPromptVersions(prompt.id, diffV1, diffV2) as { diff: string }
    setDiffText(d.diff)
    setShowDiff(true)
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="text-muted-foreground hover:text-foreground">
          <ArrowLeft size={18} />
        </button>
        <div className="flex-1">
          <h2 className="text-lg font-bold">{prompt.name}</h2>
          {prompt.description && (
            <p className="text-xs text-muted-foreground">{prompt.description}</p>
          )}
        </div>
        <div className="flex items-center gap-1">
          {prompt.tags.map((t) => (
            <span
              key={t}
              className="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded"
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Editor */}
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium">Prompt Editor</span>
              <span className="text-[10px] text-muted-foreground">v{prompt.current_version}</span>
            </div>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={12}
              className="w-full px-3 py-2 bg-secondary border border-border rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary resize-none leading-relaxed"
            />
            <div className="flex items-center justify-between mt-2">
              <div className="text-[10px] text-muted-foreground">
                Variables: {latest.variables.length > 0
                  ? latest.variables.map((v) => `{{${v}}}`).join(", ")
                  : "none"}
              </div>
              <button
                onClick={handleSaveVersion}
                disabled={saving || editText === latest.system_prompt}
                className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Save size={12} />
                Save as v{prompt.current_version + 1}
              </button>
            </div>
          </div>

          {/* Test panel */}
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium">Test Prompt</span>
              <button
                onClick={handleTest}
                className="flex items-center gap-1 px-2 py-1 text-[10px] rounded bg-success/10 text-success hover:bg-success/20"
              >
                <Play size={10} />
                Render
              </button>
            </div>
            <div className="space-y-2">
              {latest.variables.map((v) => (
                <div key={v} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground w-20 shrink-0">
                    {`{{${v}}}`}
                  </span>
                  <input
                    value={testVars[v] || ""}
                    onChange={(e) => setTestVars((prev) => ({ ...prev, [v]: e.target.value }))}
                    placeholder={`Value for ${v}`}
                    className="flex-1 px-2 py-1 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
              ))}
              {latest.variables.length === 0 && (
                <p className="text-[10px] text-muted-foreground">No variables to fill</p>
              )}
            </div>
            {testResult && (
              <div className="mt-3 p-2 bg-secondary rounded text-xs font-mono whitespace-pre-wrap">
                {testResult}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar — version history + diff */}
        <div className="space-y-3">
          {/* Version history */}
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-1.5 mb-3">
              <History size={14} className="text-muted-foreground" />
              <span className="text-xs font-medium">Version History</span>
            </div>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {[...prompt.versions].reverse().map((v) => (
                <div
                  key={v.version}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer hover:bg-secondary ${
                    v.version === prompt.current_version ? "bg-primary/10 text-primary" : ""
                  }`}
                  onClick={() => setEditText(v.system_prompt)}
                >
                  <span className="font-mono font-medium">v{v.version}</span>
                  <span className="flex-1 text-muted-foreground truncate">
                    {v.system_prompt.slice(0, 30)}...
                  </span>
                  <ChevronRight size={12} className="text-muted-foreground" />
                </div>
              ))}
            </div>
          </div>

          {/* Diff */}
          {prompt.versions.length > 1 && (
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <GitCompare size={14} className="text-muted-foreground" />
                <span className="text-xs font-medium">Compare Versions</span>
              </div>
              <div className="flex items-center gap-2 mb-2">
                <select
                  value={diffV1}
                  onChange={(e) => setDiffV1(Number(e.target.value))}
                  className="flex-1 px-2 py-1 bg-secondary border border-border rounded text-xs"
                >
                  {prompt.versions.map((v) => (
                    <option key={v.version} value={v.version}>v{v.version}</option>
                  ))}
                </select>
                <span className="text-xs text-muted-foreground">vs</span>
                <select
                  value={diffV2}
                  onChange={(e) => setDiffV2(Number(e.target.value))}
                  className="flex-1 px-2 py-1 bg-secondary border border-border rounded text-xs"
                >
                  {prompt.versions.map((v) => (
                    <option key={v.version} value={v.version}>v{v.version}</option>
                  ))}
                </select>
                <button
                  onClick={handleDiff}
                  className="px-2 py-1 text-[10px] rounded bg-secondary hover:bg-secondary/80"
                >
                  Diff
                </button>
              </div>
              {showDiff && (
                <pre className="text-[10px] font-mono bg-secondary rounded p-2 overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
                  {diffText || "No differences"}
                </pre>
              )}
            </div>
          )}

          {/* Info */}
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="text-[10px] text-muted-foreground space-y-1">
              <div>ID: {prompt.id}</div>
              <div>Created: {new Date(prompt.created_at).toLocaleString()}</div>
              <div>Updated: {new Date(prompt.updated_at).toLocaleString()}</div>
              {latest.model && <div>Model: {latest.model}</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Main Page ─────────────────────────────────────────────────── */

export function PromptLab() {
  const [prompts, setPrompts] = useState<PromptItem[]>([])
  const [selected, setSelected] = useState<PromptItem | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [tagFilter, setTagFilter] = useState<string>("")

  const loadPrompts = useCallback(async () => {
    const list = await api.listPrompts(tagFilter || undefined) as unknown as PromptItem[]
    setPrompts(list)
  }, [tagFilter])

  useEffect(() => {
    loadPrompts()
  }, [loadPrompts])

  const handleCreate = async (data: {
    name: string; system_prompt: string; description: string; tags: string[]
  }) => {
    await api.createPrompt(data)
    setShowCreate(false)
    await loadPrompts()
  }

  const handleDelete = async (id: string) => {
    await api.deletePrompt(id)
    if (selected?.id === id) setSelected(null)
    await loadPrompts()
  }

  const handleSelect = async (id: string) => {
    const p = await api.getPrompt(id) as unknown as PromptItem
    setSelected(p)
  }

  const handleRefresh = async () => {
    if (!selected) return
    const p = await api.getPrompt(selected.id) as unknown as PromptItem
    setSelected(p)
    await loadPrompts()
  }

  // Collect all unique tags
  const allTags = [...new Set(prompts.flatMap((p) => p.tags))]

  if (selected) {
    return (
      <PromptDetail
        prompt={selected}
        onBack={() => setSelected(null)}
        onRefresh={handleRefresh}
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Prompt Lab</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Version, test, and compare your prompts
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 px-3 py-2 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <Plus size={14} />
          New Prompt
        </button>
      </div>

      {/* Tag filter */}
      {allTags.length > 0 && (
        <div className="flex items-center gap-2">
          <Tag size={14} className="text-muted-foreground" />
          <button
            onClick={() => setTagFilter("")}
            className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
              !tagFilter ? "bg-primary/10 text-primary" : "bg-secondary hover:bg-secondary/80"
            }`}
          >
            All
          </button>
          {allTags.map((t) => (
            <button
              key={t}
              onClick={() => setTagFilter(t)}
              className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
                tagFilter === t ? "bg-primary/10 text-primary" : "bg-secondary hover:bg-secondary/80"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      )}

      {/* Prompt grid */}
      {prompts.length === 0 ? (
        <div className="text-center py-16 text-sm text-muted-foreground">
          No prompts yet. Create your first one to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {prompts.map((p) => (
            <PromptCard
              key={p.id}
              prompt={p}
              onSelect={() => handleSelect(p.id)}
              onDelete={() => handleDelete(p.id)}
            />
          ))}
        </div>
      )}

      {/* Create modal */}
      {showCreate && (
        <CreatePromptModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}
    </div>
  )
}
