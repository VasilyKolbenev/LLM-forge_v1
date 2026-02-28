import { useNavigate } from "react-router-dom"
import { Cpu, HardDrive, MonitorDot } from "lucide-react"
import type { SystemMetrics } from "@/hooks/useMetrics"

interface Props {
  metrics: SystemMetrics | null
  connected: boolean
}

function UtilBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{
          width: `${pct}%`,
          backgroundColor: pct > 90 ? "var(--color-destructive)" : pct > 70 ? "var(--color-warning)" : color,
        }}
      />
    </div>
  )
}

export function ResourceBar({ metrics, connected }: Props) {
  const navigate = useNavigate()

  if (!connected || !metrics) {
    return (
      <button
        onClick={() => navigate("/monitoring")}
        className="mx-2 mb-2 p-2 rounded-lg bg-secondary/50 text-[10px] text-muted-foreground hover:bg-secondary transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground animate-pulse" />
          Connecting...
        </div>
      </button>
    )
  }

  const hasGpu = metrics.gpus.length > 0
  const gpu = hasGpu ? metrics.gpus[0] : null

  return (
    <button
      onClick={() => navigate("/monitoring")}
      className="mx-2 mb-2 p-2.5 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors space-y-2"
    >
      {/* GPU */}
      {gpu && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px]">
            <span className="flex items-center gap-1 text-muted-foreground">
              <MonitorDot size={10} />
              GPU
            </span>
            <span className="text-foreground font-medium">
              {Math.round(gpu.utilization_percent)}%
            </span>
          </div>
          <UtilBar value={gpu.utilization_percent} max={100} color="var(--color-primary)" />
          <div className="flex justify-between text-[9px] text-muted-foreground">
            <span>VRAM {gpu.memory_used_gb.toFixed(1)}/{gpu.memory_total_gb.toFixed(1)} GB</span>
            <span>{gpu.temperature_c}&deg;C</span>
          </div>
        </div>
      )}

      {/* CPU */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px]">
          <span className="flex items-center gap-1 text-muted-foreground">
            <Cpu size={10} />
            CPU
          </span>
          <span className="text-foreground font-medium">
            {Math.round(metrics.cpu_percent)}%
          </span>
        </div>
        <UtilBar value={metrics.cpu_percent} max={100} color="var(--color-success)" />
      </div>

      {/* RAM */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px]">
          <span className="flex items-center gap-1 text-muted-foreground">
            <HardDrive size={10} />
            RAM
          </span>
          <span className="text-foreground font-medium">
            {metrics.ram_used_gb.toFixed(1)}/{metrics.ram_total_gb.toFixed(1)} GB
          </span>
        </div>
        <UtilBar value={metrics.ram_used_gb} max={metrics.ram_total_gb} color="#3b82f6" />
      </div>

      {/* Multi-GPU indicator */}
      {metrics.gpus.length > 1 && (
        <div className="text-[9px] text-muted-foreground text-center">
          {metrics.gpus.length} GPUs connected
        </div>
      )}
    </button>
  )
}
