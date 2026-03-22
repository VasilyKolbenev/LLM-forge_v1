import { useState, useRef, useEffect } from "react"
import { ChevronDown } from "lucide-react"

export type OfficeEnvironment =
  | "modern-office"
  | "lab"
  | "server-room"
  | "command-center"
  | "open-space"

interface EnvironmentPickerProps {
  environment: OfficeEnvironment
  onChange: (env: OfficeEnvironment) => void
}

interface EnvironmentPreset {
  id: OfficeEnvironment
  name: string
  dot: string
}

const ENVIRONMENTS: EnvironmentPreset[] = [
  { id: "modern-office", name: "Modern Office", dot: "#6d5dfc" },
  { id: "lab", name: "Research Lab", dot: "#0ea5e9" },
  { id: "server-room", name: "Server Room", dot: "#ef4444" },
  { id: "command-center", name: "Command Center", dot: "#3b82f6" },
  { id: "open-space", name: "Open Space", dot: "#84cc16" },
]

export function EnvironmentPicker({ environment, onChange }: EnvironmentPickerProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const current = ENVIRONMENTS.find((e) => e.id === environment) ?? ENVIRONMENTS[0]

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as HTMLElement)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded bg-card/80 border border-border hover:bg-secondary transition-colors"
      >
        <span
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: current.dot }}
        />
        <span className="text-foreground">{current.name}</span>
        <ChevronDown size={12} className="text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-48 bg-card border border-border rounded-lg shadow-xl z-10 py-1">
          {ENVIRONMENTS.map((env) => (
            <button
              key={env.id}
              onClick={() => {
                onChange(env.id)
                setOpen(false)
              }}
              className={`flex items-center gap-2 w-full px-3 py-2 text-xs hover:bg-secondary transition-colors ${
                env.id === environment ? "text-foreground bg-secondary/50" : "text-muted-foreground"
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: env.dot }}
              />
              {env.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
