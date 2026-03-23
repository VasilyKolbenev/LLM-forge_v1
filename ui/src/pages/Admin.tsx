import { useEffect, useState, useCallback } from "react"
import { api, AdminUser, SystemHealth } from "@/api/client"
import { useAuth } from "@/lib/auth"
import { useNavigate } from "react-router-dom"
import {
  Shield,
  Users,
  Heart,
  Settings2,
  RefreshCw,
  UserX,
  UserCheck,
  Key,
  LogOut,
  Search,
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Database,
  HardDrive,
  MemoryStick,
  Server,
} from "lucide-react"
import { AnimatedPage } from "@/components/ui/AnimatedPage"
import { motion, AnimatePresence } from "framer-motion"

type TabKey = "users" | "health" | "config"

const TABS: { key: TabKey; label: string; icon: React.ReactNode }[] = [
  { key: "users", label: "Users", icon: <Users size={16} /> },
  { key: "health", label: "System Health", icon: <Heart size={16} /> },
  { key: "config", label: "Configuration", icon: <Settings2 size={16} /> },
]

export function Admin() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<TabKey>("users")

  useEffect(() => {
    if (user && user.role !== "admin") {
      navigate("/dashboard")
    }
  }, [user, navigate])

  if (!user || user.role !== "admin") {
    return null
  }

  return (
    <AnimatedPage>
      <div className="max-w-5xl space-y-6">
        <div className="flex items-center gap-3">
          <Shield size={28} className="text-primary" />
          <div>
            <h2 className="text-2xl font-bold">Admin Panel</h2>
            <p className="text-muted-foreground text-sm mt-1">
              User management, system health, and configuration
            </p>
          </div>
        </div>

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

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            {activeTab === "users" && <UsersTab />}
            {activeTab === "health" && <HealthTab />}
            {activeTab === "config" && <ConfigTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </AnimatedPage>
  )
}

// ─── Users Tab ────────────────────────────────────────────────────────────────

function UsersTab() {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [tempPassword, setTempPassword] = useState<{ userId: string; password: string } | null>(null)

  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await api.getAdminUsers()
      setUsers(data.users)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load users")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const handleRoleChange = async (userId: string, role: string) => {
    setActionLoading(userId)
    try {
      await api.updateAdminUser(userId, { role })
      setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, role } : u)))
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update role")
    } finally {
      setActionLoading(null)
    }
  }

  const handleToggleActive = async (user: AdminUser) => {
    setActionLoading(user.id)
    try {
      if (user.is_active) {
        await api.deactivateUser(user.id)
      } else {
        await api.activateUser(user.id)
      }
      setUsers((prev) =>
        prev.map((u) => (u.id === user.id ? { ...u, is_active: !u.is_active } : u))
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to toggle user status")
    } finally {
      setActionLoading(null)
    }
  }

  const handleResetPassword = async (userId: string) => {
    setActionLoading(userId)
    try {
      const result = await api.resetUserPassword(userId)
      setTempPassword({ userId, password: result.temporary_password })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset password")
    } finally {
      setActionLoading(null)
    }
  }

  const handleForceLogout = async (userId: string) => {
    setActionLoading(userId)
    try {
      await api.forceLogoutUser(userId)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to force logout")
    } finally {
      setActionLoading(null)
    }
  }

  const filtered = users.filter(
    (u) =>
      u.email.toLowerCase().includes(search.toLowerCase()) ||
      u.name.toLowerCase().includes(search.toLowerCase())
  )

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground">
        <RefreshCw size={18} className="animate-spin mr-2" />
        Loading users...
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-lg text-sm flex items-center gap-2">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {tempPassword && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 text-yellow-200 px-4 py-3 rounded-lg text-sm">
          <p className="font-medium mb-1">Temporary password generated</p>
          <code className="bg-black/30 px-2 py-1 rounded text-xs font-mono">
            {tempPassword.password}
          </code>
          <button
            onClick={() => setTempPassword(null)}
            className="ml-3 text-xs underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search by email or name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-secondary border border-border rounded-lg text-sm
              focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        <div className="text-xs text-muted-foreground bg-secondary px-3 py-2 rounded-lg">
          Users self-register via /login
        </div>
      </div>

      <div className="border border-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-secondary/50 border-b border-border">
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Email</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Name</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Role</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Status</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Last Login</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-8 text-muted-foreground">
                  No users found
                </td>
              </tr>
            ) : (
              filtered.map((u) => (
                <tr key={u.id} className="border-b border-border last:border-0 hover:bg-secondary/30">
                  <td className="px-4 py-3 font-mono text-xs">{u.email}</td>
                  <td className="px-4 py-3">{u.name || "—"}</td>
                  <td className="px-4 py-3">
                    <select
                      value={u.role}
                      onChange={(e) => handleRoleChange(u.id, e.target.value)}
                      disabled={actionLoading === u.id}
                      className="bg-secondary border border-border rounded px-2 py-1 text-xs
                        focus:outline-none focus:ring-1 focus:ring-primary"
                    >
                      <option value="user">user</option>
                      <option value="admin">admin</option>
                      <option value="viewer">viewer</option>
                    </select>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                        u.is_active
                          ? "bg-green-500/10 text-green-400"
                          : "bg-red-500/10 text-red-400"
                      }`}
                    >
                      {u.is_active ? (
                        <CheckCircle size={12} />
                      ) : (
                        <XCircle size={12} />
                      )}
                      {u.is_active ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {u.last_login
                      ? new Date(u.last_login).toLocaleString()
                      : "Never"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => handleToggleActive(u)}
                        disabled={actionLoading === u.id}
                        title={u.is_active ? "Deactivate" : "Activate"}
                        className="p-1.5 rounded hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground disabled:opacity-50"
                      >
                        {u.is_active ? <UserX size={14} /> : <UserCheck size={14} />}
                      </button>
                      <button
                        onClick={() => handleResetPassword(u.id)}
                        disabled={actionLoading === u.id}
                        title="Reset Password"
                        className="p-1.5 rounded hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground disabled:opacity-50"
                      >
                        <Key size={14} />
                      </button>
                      <button
                        onClick={() => handleForceLogout(u.id)}
                        disabled={actionLoading === u.id}
                        title="Force Logout"
                        className="p-1.5 rounded hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground disabled:opacity-50"
                      >
                        <LogOut size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Health Tab ───────────────────────────────────────────────────────────────

function HealthTab() {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [cleanupResult, setCleanupResult] = useState<Record<string, number> | null>(null)
  const [cleanupLoading, setCleanupLoading] = useState(false)

  const fetchHealth = useCallback(async () => {
    try {
      setError(null)
      const data = await api.getSystemHealth()
      setHealth(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load health")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30_000)
    return () => clearInterval(interval)
  }, [fetchHealth])

  const handleCleanup = async () => {
    setCleanupLoading(true)
    try {
      const result = await api.runSystemCleanup()
      setCleanupResult(result.cleaned)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cleanup failed")
    } finally {
      setCleanupLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground">
        <RefreshCw size={18} className="animate-spin mr-2" />
        Loading system health...
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-lg text-sm flex items-center gap-2">
        <AlertTriangle size={16} />
        {error}
      </div>
    )
  }

  if (!health) return null

  const diskPercent = health.disk.total_gb > 0
    ? ((health.disk.used_gb / health.disk.total_gb) * 100).toFixed(1)
    : "0"
  const memPercent = health.memory.total_gb > 0
    ? ((health.memory.used_gb / health.memory.total_gb) * 100).toFixed(1)
    : "0"

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Auto-refreshes every 30s</p>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchHealth}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-secondary hover:bg-secondary/80
              border border-border rounded-lg transition-colors"
          >
            <RefreshCw size={12} />
            Refresh
          </button>
          <button
            onClick={handleCleanup}
            disabled={cleanupLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-primary/10 hover:bg-primary/20
              text-primary border border-primary/30 rounded-lg transition-colors disabled:opacity-50"
          >
            <Trash2 size={12} />
            {cleanupLoading ? "Cleaning..." : "Run Cleanup"}
          </button>
        </div>
      </div>

      {cleanupResult && (
        <div className="bg-green-500/10 border border-green-500/30 text-green-300 px-4 py-3 rounded-lg text-sm">
          <p className="font-medium mb-1">Cleanup completed</p>
          <div className="flex gap-4 text-xs">
            {Object.entries(cleanupResult).map(([key, val]) => (
              <span key={key}>
                {key}: <span className="font-mono">{val}</span>
              </span>
            ))}
          </div>
          <button
            onClick={() => setCleanupResult(null)}
            className="mt-1 text-xs underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <HealthCard
          icon={<Database size={20} />}
          title="Database"
          ok={health.database.connected}
          items={[
            { label: "Status", value: health.database.connected ? "Connected" : "Disconnected" },
            { label: "Type", value: health.database.type },
          ]}
        />
        <HealthCard
          icon={<Server size={20} />}
          title="Redis"
          ok={health.redis.connected}
          items={[
            { label: "Status", value: health.redis.connected ? "Connected" : "Disconnected" },
            { label: "Latency", value: `${health.redis.latency_ms}ms` },
          ]}
        />
        <HealthCard
          icon={<HardDrive size={20} />}
          title="S3 Storage"
          ok={health.s3.configured}
          items={[
            { label: "Status", value: health.s3.configured ? "Configured" : "Not configured" },
            { label: "Bucket", value: health.s3.bucket || "—" },
          ]}
        />
        <HealthCard
          icon={<HardDrive size={20} />}
          title="Disk Space"
          ok={parseFloat(diskPercent) < 90}
          items={[
            { label: "Used", value: `${health.disk.used_gb.toFixed(1)} / ${health.disk.total_gb.toFixed(1)} GB` },
            { label: "Free", value: `${health.disk.free_gb.toFixed(1)} GB (${diskPercent}% used)` },
          ]}
        />
        <HealthCard
          icon={<MemoryStick size={20} />}
          title="Memory"
          ok={parseFloat(memPercent) < 90}
          items={[
            { label: "Used", value: `${health.memory.used_gb.toFixed(1)} / ${health.memory.total_gb.toFixed(1)} GB` },
            { label: "Free", value: `${health.memory.free_gb.toFixed(1)} GB (${memPercent}% used)` },
          ]}
        />
      </div>
    </div>
  )
}

function HealthCard({
  icon,
  title,
  ok,
  items,
}: {
  icon: React.ReactNode
  title: string
  ok: boolean
  items: { label: string; value: string }[]
}) {
  return (
    <div className="border border-border rounded-lg p-4 bg-card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          {icon}
          {title}
        </div>
        <span
          className={`w-2.5 h-2.5 rounded-full ${ok ? "bg-green-500" : "bg-red-500"}`}
          title={ok ? "Healthy" : "Unhealthy"}
        />
      </div>
      <div className="space-y-1.5">
        {items.map((item) => (
          <div key={item.label} className="flex justify-between text-xs">
            <span className="text-muted-foreground">{item.label}</span>
            <span className="font-mono">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Config Tab ───────────────────────────────────────────────────────────────

function ConfigTab() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .getSystemConfig()
      .then((data) => setConfig(data))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load config"))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground">
        <RefreshCw size={18} className="animate-spin mr-2" />
        Loading configuration...
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-lg text-sm flex items-center gap-2">
        <AlertTriangle size={16} />
        {error}
      </div>
    )
  }

  if (!config) return null

  const configSections = [
    {
      title: "Environment",
      items: extractConfigItems(config, [
        "environment",
        "env_profile",
        "debug",
        "version",
      ]),
    },
    {
      title: "Database",
      items: extractConfigItems(config, [
        "database_type",
        "database_url",
        "database_name",
      ]),
    },
    {
      title: "Authentication",
      items: extractConfigItems(config, [
        "auth_enabled",
        "auth_mode",
        "jwt_algorithm",
        "token_expiry",
      ]),
    },
    {
      title: "Redis",
      items: extractConfigItems(config, [
        "redis_enabled",
        "redis_host",
        "redis_port",
      ]),
    },
    {
      title: "S3 / Storage",
      items: extractConfigItems(config, [
        "s3_enabled",
        "s3_bucket",
        "s3_region",
        "data_dir",
      ]),
    },
  ]

  return (
    <div className="space-y-4">
      <div className="bg-yellow-500/10 border border-yellow-500/30 text-yellow-200 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
        <AlertTriangle size={16} />
        Read-only view. Configuration changes require server restart.
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {configSections.map((section) => (
          <div
            key={section.title}
            className="border border-border rounded-lg p-4 bg-card"
          >
            <h3 className="text-sm font-medium mb-3">{section.title}</h3>
            <div className="space-y-2">
              {section.items.length === 0 ? (
                <p className="text-xs text-muted-foreground">No data available</p>
              ) : (
                section.items.map(({ key, value }) => (
                  <div key={key} className="flex justify-between text-xs gap-2">
                    <span className="text-muted-foreground shrink-0">{key}</span>
                    <span className="font-mono text-right truncate" title={String(value)}>
                      {formatConfigValue(value)}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="border border-border rounded-lg p-4 bg-card">
        <h3 className="text-sm font-medium mb-3">Raw Configuration</h3>
        <pre className="text-xs font-mono bg-secondary/50 rounded-lg p-4 overflow-auto max-h-64 text-muted-foreground">
          {JSON.stringify(config, null, 2)}
        </pre>
      </div>
    </div>
  )
}

function extractConfigItems(
  config: Record<string, unknown>,
  keys: string[]
): { key: string; value: unknown }[] {
  return keys
    .filter((k) => k in config)
    .map((k) => ({ key: k, value: config[k] }))
}

function formatConfigValue(value: unknown): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "boolean") return value ? "Yes" : "No"
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}
