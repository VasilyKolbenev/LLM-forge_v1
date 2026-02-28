import { useEffect, useState } from "react"
import { api } from "@/api/client"
import { Cpu, CheckCircle, XCircle } from "lucide-react"

export function Settings() {
  const [hardware, setHardware] = useState<Record<string, unknown> | null>(null)
  const [health, setHealth] = useState<boolean | null>(null)

  useEffect(() => {
    api.getHardware().then(setHardware).catch(() => setHardware(null))
    api.health().then(() => setHealth(true)).catch(() => setHealth(false))
  }, [])

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Settings</h2>
        <p className="text-muted-foreground text-sm mt-1">System info and hardware</p>
      </div>

      {/* Server status */}
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
            API Server {health === true ? "Online" : health === false ? "Offline" : "Checking..."}
          </div>
          <div className="text-xs text-muted-foreground">
            {health === false && "Start with: forge ui"}
          </div>
        </div>
      </div>

      {/* Hardware info */}
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Cpu size={18} className="text-primary" />
          <h3 className="font-semibold">Hardware</h3>
        </div>
        {hardware ? (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <InfoRow label="GPU" value={String(hardware.gpu_name || "N/A")} />
            <InfoRow label="GPUs" value={String(hardware.num_gpus || 0)} />
            <InfoRow label="VRAM per GPU" value={`${hardware.vram_per_gpu_gb} GB`} />
            <InfoRow label="Total VRAM" value={`${hardware.total_vram_gb} GB`} />
            <InfoRow label="BF16 Supported" value={hardware.bf16_supported ? "Yes" : "No"} />
            <InfoRow label="Recommended Strategy" value={String(hardware.strategy || "—")} />
            <InfoRow label="Batch Size" value={String(hardware.recommended_batch_size || "—")} />
            <InfoRow label="Grad Accum" value={String(hardware.recommended_gradient_accumulation || "—")} />
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">Loading hardware info...</p>
        )}
      </div>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  )
}
