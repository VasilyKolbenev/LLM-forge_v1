import { motion } from "framer-motion"
import {
  Database,
  FlaskConical,
  Box,
  Globe,
  Activity,
  ThumbsUp,
} from "lucide-react"

export interface TimelineEvent {
  timestamp: string
  event_type: string
  title: string
  description?: string
  entity_id?: string
  entity_type?: string
}

interface TimelineProps {
  events: TimelineEvent[]
  onEventClick?: (event: TimelineEvent) => void
}

const ENTITY_COLOR: Record<string, string> = {
  dataset: "#14b8a6",
  experiment: "#6d5dfc",
  model: "#3b82f6",
  deployment: "#ef4444",
  traces: "#6366f1",
  feedback: "#0ea5e9",
}

const ENTITY_ICON: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  dataset: Database,
  experiment: FlaskConical,
  model: Box,
  deployment: Globe,
  traces: Activity,
  feedback: ThumbsUp,
}

function formatTimestamp(ts: string): string {
  const d = new Date(ts)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMin = Math.floor(diffMs / 60_000)
  const diffHr = Math.floor(diffMin / 60)
  const diffDays = Math.floor(diffHr / 24)

  if (diffMin < 1) return "just now"
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return d.toLocaleDateString()
}

export function Timeline({ events, onEventClick }: TimelineProps) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        No events yet
      </div>
    )
  }

  return (
    <div className="relative h-full overflow-y-auto pr-1">
      {/* Vertical line */}
      <div className="absolute left-3 top-0 bottom-0 w-px bg-border" />

      <div className="space-y-1 py-2">
        {events.map((event, i) => {
          const color = ENTITY_COLOR[event.entity_type ?? ""] ?? "#52525b"
          const Icon = ENTITY_ICON[event.entity_type ?? ""]

          return (
            <motion.button
              key={`${event.timestamp}-${i}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03, duration: 0.2 }}
              onClick={() => onEventClick?.(event)}
              className="relative flex items-start gap-3 w-full text-left pl-1 pr-2 py-2 rounded-md hover:bg-secondary/50 transition-colors group"
            >
              {/* Dot */}
              <div
                className="relative z-10 w-6 h-6 rounded-full flex items-center justify-center shrink-0 border-2"
                style={{
                  backgroundColor: `${color}20`,
                  borderColor: `${color}60`,
                  color,
                }}
              >
                {Icon ? <Icon size={12} /> : (
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0 -mt-0.5">
                <div className="text-xs font-medium truncate group-hover:text-foreground transition-colors">
                  {event.title}
                </div>
                {event.description && (
                  <div className="text-[10px] text-muted-foreground truncate mt-0.5">
                    {event.description}
                  </div>
                )}
                <div className="text-[10px] text-muted-foreground/60 mt-0.5">
                  {formatTimestamp(event.timestamp)}
                </div>
              </div>
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}
