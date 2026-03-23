import { useEffect, useState, useCallback } from "react"
import { api, setApiKey } from "@/api/client"
import {
  Cpu,
  CheckCircle,
  XCircle,
  Shield,
  Key,
  Server,
  Copy,
  Trash2,
  Plus,
  Users,
  FileCheck,
  ScrollText,
  Settings as SettingsIcon,
  UserPlus,
  ChevronDown,
  Search,
  Calendar,
  Globe,
} from "lucide-react"
import { AnimatedPage, FadeIn } from "@/components/ui/AnimatedPage"
import { motion, AnimatePresence } from "framer-motion"

// ─── Types ───────────────────────────────────────────────────────────────────

interface ServerSettings {
  version: string
  auth_enabled: boolean
  stand_mode: string
  env_profile: string
  cors_origins: string[]
  data_dir: string
}

interface Workspace {
  id: string
  name: string
  created_at: string
  member_count?: number
}

interface WorkspaceMember {
  user_id: string
  name: string
  email: string
  role: string
}

interface Approval {
  id: string
  title: string
  requester: string
  status: "pending" | "approved" | "rejected"
  created_at: string
  resource_type: string
  reviewed_by?: string
  reviewed_at?: string
}

interface AuditEntry {
  id: string
  timestamp: string
  user: string
  action: string
  resource: string
  ip: string
}

type TabKey = "general" | "workspaces" | "approvals" | "audit"

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "general", label: "General", icon: <SettingsIcon size={16} /> },
  { key: "workspaces", label: "Workspaces", icon: <Users size={16} /> },
  { key: "approvals", label: "Approvals", icon: <FileCheck size={16} /> },
  { key: "audit", label: "Audit Log", icon: <ScrollText size={16} /> },
]

// ─── Main Component ──────────────────────────────────────────────────────────

export function Settings() {
  const [activeTab, setActiveTab] = useState<TabKey>("general")

  return (
    <AnimatedPage>
      <div className="max-w-4xl space-y-6">
        <div>
          <h2 className="text-2xl font-bold">Settings</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Server configuration, security, governance, and hardware info
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 border-b border-border">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`
                flex items-center gap-2 px-4 py-2.5 text-sm font-medium
                border-b-2 transition-colors
                ${activeTab === tab.key
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
                }
              `}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            {activeTab === "general" && <GeneralTab />}
            {activeTab === "workspaces" && <WorkspacesTab />}
            {activeTab === "approvals" && <ApprovalsTab />}
            {activeTab === "audit" && <AuditLogTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </AnimatedPage>
  )
}

// ─── General Tab (original Settings content) ─────────────────────────────────

function GeneralTab() {
  const [hardware, setHardware] = useState<Record<string, unknown> | null>(null)
  const [health, setHealth] = useState<boolean | null>(null)
  const [settings, setSettings] = useState<ServerSettings | null>(null)
  const [apiKeys, setApiKeys] = useState<Array<{ name: string }>>([])
  const [newKeyName, setNewKeyName] = useState("")
  const [generatedKey, setGeneratedKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    api.getHardware().then(setHardware).catch(() => setHardware(null))
    api.health().then(() => setHealth(true)).catch(() => setHealth(false))
    api.getSettings().then(setSettings).catch(() => {})
    api.listApiKeys().then(setApiKeys).catch(() => {})
  }, [])

  const handleGenerateKey = async () => {
    try {
      const name = newKeyName.trim() || "default"
      const result = await api.generateApiKey(name)
      setGeneratedKey(result.key)
      setApiKey(result.key)
      setNewKeyName("")
      api.listApiKeys().then(setApiKeys).catch(() => {})
    } catch {
      // silently handle
    }
  }

  const handleRevokeKey = async (name: string) => {
    try {
      await api.revokeApiKey(name)
      api.listApiKeys().then(setApiKeys).catch(() => {})
    } catch {
      // silently handle
    }
  }

  const handleCopyKey = () => {
    if (generatedKey) {
      navigator.clipboard.writeText(generatedKey)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <FadeIn delay={0}>
        <div className="bg-card border border-border rounded-lg p-4 flex items-center gap-3">
          {health === true ? (
            <CheckCircle className="text-success" size={20} />
          ) : health === false ? (
            <XCircle className="text-destructive" size={20} />
          ) : (
            <div className="w-5 h-5 rounded-full bg-muted animate-pulse" />
          )}
          <div>
            <div className="text-sm font-medium">
              API Server{" "}
              {health === true
                ? "Online"
                : health === false
                  ? "Offline"
                  : "Checking..."}
            </div>
            <div className="text-xs text-muted-foreground">
              {health === false && "Start with: pulsar ui"}
            </div>
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        <div className="bg-card border border-border rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Server size={18} className="text-primary" />
            <h3 className="font-semibold">Server Info</h3>
          </div>
          {settings ? (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <InfoRow label="Version" value={settings.version} />
              <InfoRow
                label="Auth"
                value={settings.auth_enabled ? "Enabled" : "Disabled"}
              />
              <InfoRow label="Stand Mode" value={settings.stand_mode} />
              <InfoRow label="Env Profile" value={settings.env_profile} />
              <div className="col-span-2">
                <div className="text-xs text-muted-foreground">
                  CORS Origins
                </div>
                <div className="font-mono text-xs mt-0.5">
                  {settings.cors_origins.join(", ")}
                </div>
              </div>
              <div className="col-span-2">
                <div className="text-xs text-muted-foreground">
                  Data Directory
                </div>
                <div className="font-mono text-xs mt-0.5 break-all">
                  {settings.data_dir}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Loading...</p>
          )}
        </div>
      </FadeIn>

      <FadeIn delay={0.1}>
        <div className="bg-card border border-border rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Shield size={18} className="text-primary" />
            <h3 className="font-semibold">API Keys</h3>
            {settings && !settings.auth_enabled && (
              <span className="text-[10px] bg-muted text-muted-foreground px-2 py-0.5 rounded ml-auto">
                Auth disabled - keys stored but not enforced
              </span>
            )}
          </div>

          {generatedKey && (
            <div className="bg-success/10 border border-success/30 rounded-lg p-3 space-y-2">
              <div className="text-xs font-medium text-success">
                New key generated - copy it now, it will not be shown again
              </div>
              <div className="flex items-center gap-2">
                <code className="flex-1 text-xs font-mono bg-secondary p-2 rounded break-all">
                  {generatedKey}
                </code>
                <button
                  onClick={handleCopyKey}
                  className="p-2 hover:bg-secondary rounded transition-colors"
                  title="Copy"
                >
                  <Copy size={14} />
                </button>
              </div>
              {copied && (
                <span className="text-[10px] text-success">Copied!</span>
              )}
            </div>
          )}

          {apiKeys.length > 0 ? (
            <div className="space-y-1">
              {apiKeys.map((k, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-secondary/30"
                >
                  <div className="flex items-center gap-2">
                    <Key size={12} className="text-muted-foreground" />
                    <span className="text-sm">{k.name}</span>
                  </div>
                  <button
                    onClick={() => handleRevokeKey(k.name)}
                    className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                    title="Revoke"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No API keys generated
            </p>
          )}

          <div className="flex items-center gap-2 pt-2 border-t border-border">
            <input
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="Key name (optional)"
              className="flex-1 px-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
            />
            <button
              onClick={handleGenerateKey}
              className="flex items-center gap-1 px-3 py-1.5 bg-primary text-primary-foreground rounded text-xs hover:bg-primary/90 transition-colors"
            >
              <Plus size={12} />
              Generate
            </button>
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.15}>
        <div className="bg-card border border-border rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Cpu size={18} className="text-primary" />
            <h3 className="font-semibold">Hardware</h3>
          </div>
          {hardware ? (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <InfoRow
                label="GPU"
                value={String(hardware.gpu_name || "N/A")}
              />
              <InfoRow label="GPUs" value={String(hardware.num_gpus || 0)} />
              <InfoRow
                label="VRAM per GPU"
                value={`${hardware.vram_per_gpu_gb} GB`}
              />
              <InfoRow
                label="Total VRAM"
                value={`${hardware.total_vram_gb} GB`}
              />
              <InfoRow
                label="BF16 Supported"
                value={hardware.bf16_supported ? "Yes" : "No"}
              />
              <InfoRow
                label="Recommended Strategy"
                value={String(hardware.strategy || "-")}
              />
              <InfoRow
                label="Batch Size"
                value={String(hardware.recommended_batch_size || "-")}
              />
              <InfoRow
                label="Grad Accum"
                value={String(
                  hardware.recommended_gradient_accumulation || "-"
                )}
              />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Loading hardware info...
            </p>
          )}
        </div>
      </FadeIn>
    </div>
  )
}

// ─── Workspaces Tab ──────────────────────────────────────────────────────────

function WorkspacesTab() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [loading, setLoading] = useState(true)
  const [newName, setNewName] = useState("")
  const [creating, setCreating] = useState(false)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [members, setMembers] = useState<Record<string, WorkspaceMember[]>>({})
  const [addMemberForm, setAddMemberForm] = useState<{
    workspaceId: string
    userId: string
    role: string
  } | null>(null)

  const fetchWorkspaces = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/governance/workspaces", {
        headers: authHeaders(),
      })
      const data = await res.json()
      setWorkspaces(data.workspaces || [])
    } catch {
      setWorkspaces([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchWorkspaces()
  }, [fetchWorkspaces])

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await fetch("/api/v1/governance/workspaces", {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName.trim() }),
      })
      setNewName("")
      await fetchWorkspaces()
    } finally {
      setCreating(false)
    }
  }

  const fetchMembers = async (workspaceId: string) => {
    try {
      const res = await fetch(
        `/api/v1/governance/workspaces/${workspaceId}/members`,
        { headers: authHeaders() }
      )
      const data = await res.json()
      setMembers((prev) => ({ ...prev, [workspaceId]: data.members || [] }))
    } catch {
      setMembers((prev) => ({ ...prev, [workspaceId]: [] }))
    }
  }

  const toggleExpand = (id: string) => {
    if (expandedId === id) {
      setExpandedId(null)
    } else {
      setExpandedId(id)
      if (!members[id]) fetchMembers(id)
    }
  }

  const handleAddMember = async () => {
    if (!addMemberForm) return
    const { workspaceId, userId, role } = addMemberForm
    if (!userId.trim()) return
    try {
      await fetch(
        `/api/v1/governance/workspaces/${workspaceId}/members`,
        {
          method: "POST",
          headers: { ...authHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId.trim(), role: role || "viewer" }),
        }
      )
      setAddMemberForm(null)
      await fetchMembers(workspaceId)
    } catch {
      // silently handle
    }
  }

  const handleRemoveMember = async (workspaceId: string, uid: string) => {
    try {
      await fetch(
        `/api/v1/governance/workspaces/${workspaceId}/members/${uid}`,
        { method: "DELETE", headers: authHeaders() }
      )
      await fetchMembers(workspaceId)
    } catch {
      // silently handle
    }
  }

  const handleUpdateRole = async (
    workspaceId: string,
    uid: string,
    role: string
  ) => {
    try {
      await fetch(
        `/api/v1/governance/workspaces/${workspaceId}/members/${uid}`,
        {
          method: "PUT",
          headers: { ...authHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ role }),
        }
      )
      await fetchMembers(workspaceId)
    } catch {
      // silently handle
    }
  }

  if (loading) {
    return <LoadingPlaceholder text="Loading workspaces..." />
  }

  return (
    <div className="space-y-4">
      {/* Create workspace */}
      <FadeIn delay={0}>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Users size={18} className="text-primary" />
            <h3 className="font-semibold">Workspaces</h3>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="New workspace name"
              className="flex-1 px-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <button
              onClick={handleCreate}
              disabled={creating || !newName.trim()}
              className="flex items-center gap-1 px-3 py-1.5 bg-primary text-primary-foreground rounded text-xs hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              <Plus size={12} />
              Create
            </button>
          </div>
        </div>
      </FadeIn>

      {/* Workspace list */}
      {workspaces.length === 0 ? (
        <FadeIn delay={0.05}>
          <div className="text-center py-12 text-muted-foreground text-sm">
            No workspaces yet. Create one above to get started.
          </div>
        </FadeIn>
      ) : (
        workspaces.map((ws, i) => (
          <FadeIn key={ws.id} delay={0.05 * (i + 1)}>
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleExpand(ws.id)}
                className="w-full flex items-center justify-between p-4 hover:bg-secondary/30 transition-colors text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center">
                    <Users size={16} className="text-primary" />
                  </div>
                  <div>
                    <div className="text-sm font-medium">{ws.name}</div>
                    <div className="text-xs text-muted-foreground">
                      Created {new Date(ws.created_at).toLocaleDateString()}
                      {ws.member_count !== undefined &&
                        ` \u00B7 ${ws.member_count} members`}
                    </div>
                  </div>
                </div>
                <ChevronDown
                  size={16}
                  className={`text-muted-foreground transition-transform ${
                    expandedId === ws.id ? "rotate-180" : ""
                  }`}
                />
              </button>

              <AnimatePresence>
                {expandedId === ws.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="border-t border-border p-4 space-y-3">
                      {/* Members */}
                      {members[ws.id] ? (
                        members[ws.id].length > 0 ? (
                          <div className="space-y-1">
                            {members[ws.id].map((m) => (
                              <div
                                key={m.user_id}
                                className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-secondary/30"
                              >
                                <div className="flex items-center gap-2">
                                  <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center text-[10px] font-medium text-primary">
                                    {m.name?.[0]?.toUpperCase() || "U"}
                                  </div>
                                  <div>
                                    <div className="text-sm">{m.name}</div>
                                    <div className="text-[10px] text-muted-foreground">
                                      {m.email}
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <RoleBadge
                                    role={m.role}
                                    onChange={(role) =>
                                      handleUpdateRole(ws.id, m.user_id, role)
                                    }
                                  />
                                  <button
                                    onClick={() =>
                                      handleRemoveMember(ws.id, m.user_id)
                                    }
                                    className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                                    title="Remove member"
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            No members in this workspace
                          </p>
                        )
                      ) : (
                        <p className="text-xs text-muted-foreground animate-pulse">
                          Loading members...
                        </p>
                      )}

                      {/* Add member form */}
                      {addMemberForm?.workspaceId === ws.id ? (
                        <div className="flex items-center gap-2 pt-2 border-t border-border">
                          <input
                            value={addMemberForm.userId}
                            onChange={(e) =>
                              setAddMemberForm({
                                ...addMemberForm,
                                userId: e.target.value,
                              })
                            }
                            placeholder="User ID or email"
                            className="flex-1 px-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                          />
                          <select
                            value={addMemberForm.role}
                            onChange={(e) =>
                              setAddMemberForm({
                                ...addMemberForm,
                                role: e.target.value,
                              })
                            }
                            className="px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                          >
                            <option value="viewer">Viewer</option>
                            <option value="editor">Editor</option>
                            <option value="admin">Admin</option>
                          </select>
                          <button
                            onClick={handleAddMember}
                            className="px-3 py-1.5 bg-primary text-primary-foreground rounded text-xs hover:bg-primary/90 transition-colors"
                          >
                            Add
                          </button>
                          <button
                            onClick={() => setAddMemberForm(null)}
                            className="px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() =>
                            setAddMemberForm({
                              workspaceId: ws.id,
                              userId: "",
                              role: "viewer",
                            })
                          }
                          className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 transition-colors pt-2 border-t border-border"
                        >
                          <UserPlus size={12} />
                          Add member
                        </button>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </FadeIn>
        ))
      )}
    </div>
  )
}

// ─── Approvals Tab ───────────────────────────────────────────────────────────

function ApprovalsTab() {
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [pendingCount, setPendingCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [reviewNote, setReviewNote] = useState<Record<string, string>>({})

  const fetchApprovals = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/governance/approvals", {
        headers: authHeaders(),
      })
      const data = await res.json()
      setApprovals(data.approvals || [])
      setPendingCount((data.approvals || []).filter((a: Approval) => a.status === "pending").length)
    } catch {
      setApprovals([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchApprovals()
  }, [fetchApprovals])

  const handleReview = async (
    id: string,
    decision: "approved" | "rejected"
  ) => {
    try {
      await fetch(`/api/v1/governance/approvals/${id}/review`, {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({
          status: decision,
          reason: reviewNote[id] || "",
        }),
      })
      setReviewNote((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
      await fetchApprovals()
    } catch {
      // silently handle
    }
  }

  if (loading) {
    return <LoadingPlaceholder text="Loading approvals..." />
  }

  return (
    <div className="space-y-4">
      <FadeIn delay={0}>
        <div className="bg-card border border-border rounded-lg p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileCheck size={18} className="text-primary" />
            <h3 className="font-semibold">Approval Requests</h3>
          </div>
          {pendingCount > 0 && (
            <span className="px-2 py-0.5 text-xs font-medium bg-yellow-500/20 text-yellow-400 rounded-full">
              {pendingCount} pending
            </span>
          )}
        </div>
      </FadeIn>

      {approvals.length === 0 ? (
        <FadeIn delay={0.05}>
          <div className="text-center py-12 text-muted-foreground text-sm">
            No approval requests found.
          </div>
        </FadeIn>
      ) : (
        approvals.map((a, i) => (
          <FadeIn key={a.id} delay={0.05 * (i + 1)}>
            <div className="bg-card border border-border rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium">{a.title}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    by {a.requester} &middot;{" "}
                    {new Date(a.created_at).toLocaleString()} &middot;{" "}
                    {a.resource_type}
                  </div>
                </div>
                <StatusBadge status={a.status} />
              </div>

              {a.status === "pending" && (
                <div className="space-y-2 pt-2 border-t border-border">
                  <input
                    value={reviewNote[a.id] || ""}
                    onChange={(e) =>
                      setReviewNote((prev) => ({
                        ...prev,
                        [a.id]: e.target.value,
                      }))
                    }
                    placeholder="Review note (optional)"
                    className="w-full px-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleReview(a.id, "approved")}
                      className="flex items-center gap-1 px-3 py-1.5 bg-emerald-600 text-white rounded text-xs hover:bg-emerald-500 transition-colors"
                    >
                      <CheckCircle size={12} />
                      Approve
                    </button>
                    <button
                      onClick={() => handleReview(a.id, "rejected")}
                      className="flex items-center gap-1 px-3 py-1.5 bg-red-600 text-white rounded text-xs hover:bg-red-500 transition-colors"
                    >
                      <XCircle size={12} />
                      Reject
                    </button>
                  </div>
                </div>
              )}

              {a.status !== "pending" && a.reviewed_by && (
                <div className="text-[10px] text-muted-foreground pt-1 border-t border-border">
                  Reviewed by {a.reviewed_by}
                  {a.reviewed_at &&
                    ` on ${new Date(a.reviewed_at).toLocaleString()}`}
                </div>
              )}
            </div>
          </FadeIn>
        ))
      )}
    </div>
  )
}

// ─── Audit Log Tab ───────────────────────────────────────────────────────────

function AuditLogTab() {
  const [logs, setLogs] = useState<AuditEntry[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [actionFilter, setActionFilter] = useState("")
  const [dateFilter, setDateFilter] = useState("")

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (actionFilter) params.set("action", actionFilter)
      if (dateFilter) params.set("date", dateFilter)
      const qs = params.toString()
      const res = await fetch(
        `/api/v1/governance/audit${qs ? `?${qs}` : ""}`,
        { headers: authHeaders() }
      )
      const data = await res.json()
      setLogs(data.logs || [])
      setTotal(data.total || 0)
    } catch {
      setLogs([])
    } finally {
      setLoading(false)
    }
  }, [actionFilter, dateFilter])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  return (
    <div className="space-y-4">
      <FadeIn delay={0}>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <ScrollText size={18} className="text-primary" />
              <h3 className="font-semibold">Audit Log</h3>
              {total > 0 && (
                <span className="text-xs text-muted-foreground">
                  ({total} entries)
                </span>
              )}
            </div>
          </div>
          {/* Filters */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search
                size={12}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
              />
              <input
                value={actionFilter}
                onChange={(e) => setActionFilter(e.target.value)}
                placeholder="Filter by action..."
                className="w-full pl-7 pr-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <div className="relative">
              <Calendar
                size={12}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none"
              />
              <input
                type="date"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className="pl-7 pr-3 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.05}>
        {loading ? (
          <LoadingPlaceholder text="Loading audit log..." />
        ) : logs.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground text-sm">
            No audit entries found.
          </div>
        ) : (
          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                      Timestamp
                    </th>
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                      User
                    </th>
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                      Action
                    </th>
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                      Resource
                    </th>
                    <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Globe size={11} />
                        IP
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((entry, i) => (
                    <motion.tr
                      key={entry.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className="border-b border-border last:border-0 hover:bg-secondary/30 transition-colors"
                    >
                      <td className="px-4 py-2.5 text-muted-foreground font-mono whitespace-nowrap">
                        {new Date(entry.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5">{entry.user}</td>
                      <td className="px-4 py-2.5">
                        <span className="px-1.5 py-0.5 bg-primary/10 text-primary rounded text-[10px] font-medium">
                          {entry.action}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 font-mono text-muted-foreground">
                        {entry.resource}
                      </td>
                      <td className="px-4 py-2.5 font-mono text-muted-foreground">
                        {entry.ip}
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </FadeIn>
    </div>
  )
}

// ─── Shared Components ───────────────────────────────────────────────────────

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  )
}

function StatusBadge({ status }: { status: "pending" | "approved" | "rejected" }) {
  const styles = {
    pending: "bg-yellow-500/20 text-yellow-400",
    approved: "bg-emerald-500/20 text-emerald-400",
    rejected: "bg-red-500/20 text-red-400",
  }
  return (
    <span
      className={`px-2 py-0.5 text-[10px] font-medium rounded-full capitalize ${styles[status]}`}
    >
      {status}
    </span>
  )
}

function RoleBadge({
  role,
  onChange,
}: {
  role: string
  onChange: (role: string) => void
}) {
  const colors: Record<string, string> = {
    admin: "bg-purple-500/20 text-purple-400",
    editor: "bg-blue-500/20 text-blue-400",
    viewer: "bg-zinc-500/20 text-zinc-400",
  }
  return (
    <select
      value={role}
      onChange={(e) => onChange(e.target.value)}
      className={`
        px-2 py-0.5 text-[10px] font-medium rounded-full border-0
        cursor-pointer appearance-none text-center
        ${colors[role] || colors.viewer}
        bg-opacity-20 focus:outline-none focus:ring-1 focus:ring-primary
      `}
    >
      <option value="viewer">viewer</option>
      <option value="editor">editor</option>
      <option value="admin">admin</option>
    </select>
  )
}

function LoadingPlaceholder({ text }: { text: string }) {
  return (
    <div className="bg-card border border-border rounded-lg p-8 text-center">
      <div className="w-6 h-6 rounded-full border-2 border-primary border-t-transparent animate-spin mx-auto mb-3" />
      <p className="text-sm text-muted-foreground">{text}</p>
    </div>
  )
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function authHeaders(): Record<string, string> {
  const key = localStorage.getItem("pulsar_api_key")
  if (key) return { Authorization: `Bearer ${key}` }
  return {}
}
