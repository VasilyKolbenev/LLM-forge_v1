import { useState, useEffect, useRef, useCallback } from "react"

export interface GPUMetrics {
  index: number
  name: string
  utilization_percent: number
  memory_used_gb: number
  memory_total_gb: number
  temperature_c: number
  power_watts: number
}

export interface SystemMetrics {
  timestamp: number
  cpu_percent: number
  cpu_count: number
  ram_used_gb: number
  ram_total_gb: number
  ram_percent: number
  disk_used_gb: number
  disk_total_gb: number
  gpus: GPUMetrics[]
}

const MAX_HISTORY = 900 // 30 min at 2s interval

export function useMetrics() {
  const [current, setCurrent] = useState<SystemMetrics | null>(null)
  const [history, setHistory] = useState<SystemMetrics[]>([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  const connect = useCallback(() => {
    if (esRef.current) return

    const es = new EventSource("/api/v1/metrics/live")
    esRef.current = es

    es.onmessage = (e) => {
      try {
        const data: SystemMetrics = JSON.parse(e.data)
        setCurrent(data)
        setHistory((prev) => {
          const next = [...prev, data]
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next
        })
        setConnected(true)
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      setConnected(false)
      es.close()
      esRef.current = null
      // Reconnect after 5s
      setTimeout(connect, 5000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      esRef.current?.close()
      esRef.current = null
    }
  }, [connect])

  return { current, history, connected }
}
