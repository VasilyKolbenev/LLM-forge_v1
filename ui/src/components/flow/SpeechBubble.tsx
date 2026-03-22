import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

interface SpeechBubbleProps {
  messages: string[]
  status: string
  visible: boolean
}

export function SpeechBubble({ messages, status, visible }: SpeechBubbleProps) {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (status !== "running" || messages.length <= 1) return
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % messages.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [status, messages.length])

  useEffect(() => {
    setIndex(0)
  }, [status])

  const text = messages[index] ?? messages[0] ?? ""

  return (
    <AnimatePresence>
      {visible && text && (
        <motion.div
          initial={{ opacity: 0, y: 6, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 6, scale: 0.9 }}
          transition={{ duration: 0.2 }}
          className="absolute -top-9 left-1/2 -translate-x-1/2 z-10 pointer-events-none"
        >
          <div className="relative bg-card border border-border rounded-md px-2 py-1 shadow-lg whitespace-nowrap">
            <span className="text-[10px] text-muted-foreground">{text}</span>
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 bg-card border-r border-b border-border rotate-45" />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
