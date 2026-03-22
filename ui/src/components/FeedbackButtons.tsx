import { useState } from "react"
import { ThumbsUp, ThumbsDown, Check } from "lucide-react"

interface FeedbackButtonsProps {
  traceId: string
  onFeedback?: (traceId: string, rating: number, reason?: string) => void
}

const REASONS = [
  { value: "hallucination", label: "Hallucination" },
  { value: "wrong_tool", label: "Wrong tool" },
  { value: "unsafe", label: "Unsafe" },
  { value: "slow", label: "Too slow" },
  { value: "other", label: "Other" },
]

export function FeedbackButtons({ traceId, onFeedback }: FeedbackButtonsProps) {
  const [saved, setSaved] = useState(false)
  const [showReasons, setShowReasons] = useState(false)
  const [sending, setSending] = useState(false)

  const submit = async (rating: number, reason?: string) => {
    setSending(true)
    try {
      const res = await fetch(`/api/v1/traces/${traceId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feedback_type: "thumbs",
          rating,
          reason: reason || "",
        }),
      })
      if (res.ok) {
        setSaved(true)
        setShowReasons(false)
        onFeedback?.(traceId, rating, reason)
      }
    } catch {
      // silently ignore feedback errors
    } finally {
      setSending(false)
    }
  }

  if (saved) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground mt-1">
        <Check size={12} className="text-green-500" /> Saved
      </span>
    )
  }

  return (
    <div className="mt-1">
      <div className="inline-flex items-center gap-1">
        <button
          onClick={() => submit(1)}
          disabled={sending}
          className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          title="Good response"
        >
          <ThumbsUp size={14} />
        </button>
        <button
          onClick={() => setShowReasons(true)}
          disabled={sending}
          className="p-1 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          title="Bad response"
        >
          <ThumbsDown size={14} />
        </button>
      </div>
      {showReasons && (
        <div className="flex flex-wrap gap-1 mt-1">
          {REASONS.map((r) => (
            <button
              key={r.value}
              onClick={() => submit(0, r.value)}
              disabled={sending}
              className="text-xs px-2 py-0.5 rounded bg-secondary text-muted-foreground hover:text-foreground hover:bg-secondary/80 transition-colors disabled:opacity-50"
            >
              {r.label}
            </button>
          ))}
          <button
            onClick={() => setShowReasons(false)}
            className="text-xs px-2 py-0.5 text-muted-foreground hover:text-foreground"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}
