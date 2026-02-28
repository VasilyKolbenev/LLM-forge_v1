import { useEffect, useRef, useState, useCallback } from "react"
import { sseUrl } from "@/api/client"

export interface SSEMetrics {
  step: number
  epoch: number
  loss: number | null
  learning_rate: number | null
  gpu_mem_gb: number | null
}

interface SSEEvent {
  event: string
  data: Record<string, unknown>
}

export function useSSE(jobId: string | null) {
  const [metrics, setMetrics] = useState<SSEMetrics[]>([])
  const [status, setStatus] = useState<"idle" | "streaming" | "completed" | "error">("idle")
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const stop = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
  }, [])

  useEffect(() => {
    if (!jobId) {
      setStatus("idle")
      return
    }

    setMetrics([])
    setStatus("streaming")
    setError(null)

    const es = new EventSource(sseUrl(jobId))
    esRef.current = es

    es.onmessage = (e) => {
      try {
        const event: SSEEvent = JSON.parse(e.data)
        if (event.event === "metrics") {
          setMetrics((prev) => [...prev, event.data as unknown as SSEMetrics])
        } else if (event.event === "completed") {
          setStatus("completed")
          es.close()
        } else if (event.event === "error") {
          setStatus("error")
          setError((event.data as { error?: string }).error || "Unknown error")
          es.close()
        }
      } catch {
        // ignore parse errors on keepalive
      }
    }

    es.onerror = () => {
      setStatus("error")
      setError("Connection lost")
      es.close()
    }

    return () => {
      es.close()
      esRef.current = null
    }
  }, [jobId])

  return { metrics, status, error, stop }
}
