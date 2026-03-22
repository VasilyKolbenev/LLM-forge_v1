import { useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  X,
} from "lucide-react"
import type { TraceEvent } from "@/hooks/useTraceReplay"

interface TraceReplayProps {
  trace: TraceEvent[]
  isPlaying: boolean
  currentIndex: number
  speed: number
  onPlay: () => void
  onPause: () => void
  onStepForward: () => void
  onSeek: (index: number) => void
  onSpeedChange: (speed: number) => void
  onClose: () => void
  currentTime: number
  totalTime: number
}

const SPEEDS = [1, 2, 4]

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, "0")}`
}

function EventMarker({
  event,
  position,
  isCurrent,
}: {
  event: TraceEvent
  position: number
  isCurrent: boolean
}) {
  const colorClass =
    event.type === "complete"
      ? "bg-emerald-500"
      : event.type === "error"
        ? "bg-red-500"
        : event.type === "start"
          ? "bg-blue-500"
          : "bg-blue-400"

  return (
    <div
      className="absolute top-1/2 -translate-y-1/2"
      style={{ left: `${position}%` }}
      title={event.message ?? event.type}
    >
      <div
        className={`w-1.5 h-1.5 rounded-full ${colorClass} ${
          isCurrent ? "ring-2 ring-white/60 scale-150" : ""
        } transition-transform`}
      />
    </div>
  )
}

function ActiveEvents({
  trace,
  currentIndex,
}: {
  trace: TraceEvent[]
  currentIndex: number
}) {
  const recentEvents = useMemo(() => {
    if (currentIndex < 0) return []
    const seen = new Set<string>()
    const events: TraceEvent[] = []
    for (let i = currentIndex; i >= 0 && events.length < 4; i--) {
      const ev = trace[i]
      if (seen.has(ev.nodeId)) continue
      seen.add(ev.nodeId)
      events.unshift(ev)
    }
    return events
  }, [trace, currentIndex])

  return (
    <div className="flex items-center gap-2 overflow-x-auto scrollbar-none px-1 min-h-[20px]">
      <AnimatePresence mode="popLayout">
        {recentEvents.map((ev) => {
          const dotColor =
            ev.type === "complete"
              ? "bg-emerald-400"
              : ev.type === "error"
                ? "bg-red-400"
                : "bg-blue-400"
          return (
            <motion.div
              key={`${ev.nodeId}-${ev.timestamp}`}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
              transition={{ duration: 0.15 }}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-white/5 text-[10px] text-muted-foreground whitespace-nowrap shrink-0"
            >
              <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
              {ev.message ?? ev.type}
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}

export function TraceReplay({
  trace,
  isPlaying,
  currentIndex,
  speed,
  onPlay,
  onPause,
  onStepForward,
  onSeek,
  onSpeedChange,
  onClose,
  currentTime,
  totalTime,
}: TraceReplayProps) {
  const progress = totalTime > 0 ? (currentTime / totalTime) * 100 : 0

  const handleSliderClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const targetTime = ratio * totalTime
    // Find the closest event index
    let closest = 0
    let minDist = Infinity
    for (let i = 0; i < trace.length; i++) {
      const dist = Math.abs(trace[i].timestamp - targetTime)
      if (dist < minDist) {
        minDist = dist
        closest = i
      }
    }
    onSeek(closest)
  }

  const handleStepBack = () => {
    if (currentIndex > 0) {
      onSeek(currentIndex - 1)
    }
  }

  return (
    <motion.div
      initial={{ y: 80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 80, opacity: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 300 }}
      className="absolute bottom-0 left-0 right-0 z-40 bg-[#111113] border-t border-border/60 backdrop-blur-sm"
    >
      <div className="px-4 py-2 space-y-1.5">
        {/* Controls row */}
        <div className="flex items-center gap-3">
          {/* Playback controls */}
          <div className="flex items-center gap-1">
            <button
              onClick={handleStepBack}
              disabled={currentIndex <= 0}
              className="p-1 rounded hover:bg-white/10 text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              title="Step back"
            >
              <SkipBack size={14} />
            </button>
            <button
              onClick={isPlaying ? onPause : onPlay}
              className="p-1.5 rounded hover:bg-white/10 text-foreground transition-colors"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button
              onClick={onStepForward}
              disabled={currentIndex >= trace.length - 1}
              className="p-1 rounded hover:bg-white/10 text-muted-foreground hover:text-foreground disabled:opacity-30 transition-colors"
              title="Step forward"
            >
              <SkipForward size={14} />
            </button>
          </div>

          {/* Divider */}
          <div className="w-px h-5 bg-border/50" />

          {/* Speed selector */}
          <div className="flex items-center gap-0.5">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => onSpeedChange(s)}
                className={`px-1.5 py-0.5 text-[10px] rounded font-medium transition-colors ${
                  speed === s
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/10"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          {/* Divider */}
          <div className="w-px h-5 bg-border/50" />

          {/* Time display */}
          <div className="text-[11px] text-muted-foreground font-mono tabular-nums">
            {formatTime(currentTime)} / {formatTime(totalTime)}
          </div>

          <div className="flex-1" />

          {/* Close button */}
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-white/10 text-muted-foreground hover:text-foreground transition-colors"
            title="Exit replay"
          >
            <X size={14} />
          </button>
        </div>

        {/* Timeline slider */}
        <div
          className="relative h-3 cursor-pointer group"
          onClick={handleSliderClick}
        >
          {/* Track background */}
          <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 bg-white/10 rounded-full" />

          {/* Progress fill */}
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 left-0 h-1 bg-primary/70 rounded-full"
            style={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />

          {/* Event markers */}
          {trace.map((event, i) => {
            const pos = totalTime > 0 ? (event.timestamp / totalTime) * 100 : 0
            return (
              <EventMarker
                key={`${event.nodeId}-${event.timestamp}-${i}`}
                event={event}
                position={pos}
                isCurrent={i === currentIndex}
              />
            )
          })}

          {/* Playhead */}
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-primary rounded-full shadow-md border border-primary-foreground/30 -ml-[5px] group-hover:scale-125 transition-transform"
            style={{ left: `${progress}%` }}
          />
        </div>

        {/* Active event labels */}
        <ActiveEvents trace={trace} currentIndex={currentIndex} />
      </div>
    </motion.div>
  )
}
