import { useState, useMemo } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, RotateCcw } from "lucide-react"
import { PERSONAS } from "./personas"

export interface CustomPersona {
  name?: string
  role?: string
  avatarColor?: string
  avatarEmoji?: string
  idleMessage?: string
  workingMessages?: string[]
  doneMessage?: string
  errorMessage?: string
}

interface PersonaEditorProps {
  nodeId: string
  currentType: string
  customPersona?: CustomPersona
  onSave: (nodeId: string, persona: CustomPersona) => void
  onClose: () => void
}

const PRESET_COLORS = [
  "#6d5dfc", "#eab308", "#22c55e", "#3b82f6",
  "#ef4444", "#ec4899", "#8b5cf6", "#06b6d4",
  "#f97316", "#14b8a6", "#0ea5e9", "#84cc16",
]

export function PersonaEditor({
  nodeId,
  currentType,
  customPersona,
  onSave,
  onClose,
}: PersonaEditorProps) {
  const defaults = PERSONAS[currentType]
  const fallback = {
    name: "Node",
    role: "Generic",
    color: "#6d5dfc",
    idleMessage: "Idle...",
    workingMessages: ["Working..."],
    doneMessage: "Done!",
    errorMessage: "Error...",
  }

  const baseName = defaults?.name ?? fallback.name
  const baseRole = defaults?.role ?? fallback.role
  const baseColor = defaults?.color ?? fallback.color
  const baseIdleMessage = defaults?.idleMessage ?? fallback.idleMessage
  const baseWorkingMessages = defaults?.workingMessages ?? fallback.workingMessages
  const baseDoneMessage = defaults?.doneMessage ?? fallback.doneMessage
  const baseErrorMessage = defaults?.errorMessage ?? fallback.errorMessage

  const [name, setName] = useState(customPersona?.name ?? baseName)
  const [role, setRole] = useState(customPersona?.role ?? baseRole)
  const [avatarColor, setAvatarColor] = useState(customPersona?.avatarColor ?? baseColor)
  const [avatarEmoji, setAvatarEmoji] = useState(customPersona?.avatarEmoji ?? "")
  const [idleMessage, setIdleMessage] = useState(customPersona?.idleMessage ?? baseIdleMessage)
  const [workingMessages, setWorkingMessages] = useState<string[]>(
    customPersona?.workingMessages ?? [...baseWorkingMessages]
  )
  const [doneMessage, setDoneMessage] = useState(customPersona?.doneMessage ?? baseDoneMessage)
  const [errorMessage, setErrorMessage] = useState(customPersona?.errorMessage ?? baseErrorMessage)

  const avatarDisplay = useMemo(() => {
    if (avatarEmoji) return avatarEmoji
    return name.charAt(0).toUpperCase() || "?"
  }, [avatarEmoji, name])

  const handleAddWorkingMessage = () => {
    setWorkingMessages((prev) => [...prev, ""])
  }

  const handleUpdateWorkingMessage = (index: number, value: string) => {
    setWorkingMessages((prev) => prev.map((m, i) => (i === index ? value : m)))
  }

  const handleRemoveWorkingMessage = (index: number) => {
    if (workingMessages.length <= 1) return
    setWorkingMessages((prev) => prev.filter((_, i) => i !== index))
  }

  const handleReset = () => {
    setName(baseName)
    setRole(baseRole)
    setAvatarColor(baseColor)
    setAvatarEmoji("")
    setIdleMessage(baseIdleMessage)
    setWorkingMessages([...baseWorkingMessages])
    setDoneMessage(baseDoneMessage)
    setErrorMessage(baseErrorMessage)
  }

  const handleSave = () => {
    const persona: CustomPersona = {}
    if (name !== baseName) persona.name = name
    if (role !== baseRole) persona.role = role
    if (avatarColor !== baseColor) persona.avatarColor = avatarColor
    if (avatarEmoji) persona.avatarEmoji = avatarEmoji
    if (idleMessage !== baseIdleMessage) persona.idleMessage = idleMessage
    if (JSON.stringify(workingMessages) !== JSON.stringify(baseWorkingMessages)) {
      persona.workingMessages = workingMessages.filter((m) => m.trim() !== "")
    }
    if (doneMessage !== baseDoneMessage) persona.doneMessage = doneMessage
    if (errorMessage !== baseErrorMessage) persona.errorMessage = errorMessage
    onSave(nodeId, persona)
  }

  const inputClasses =
    "w-full px-2 py-1.5 bg-secondary border border-border rounded text-xs focus:outline-none focus:ring-1 focus:ring-primary"

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
        <motion.div
          className="bg-card border border-border rounded-lg w-[420px] max-h-[80vh] flex flex-col shadow-2xl"
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h2 className="text-sm font-semibold">Customize Agent</h2>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X size={16} />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Avatar Preview */}
            <div className="flex justify-center">
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold transition-colors"
                style={{
                  backgroundColor: `${avatarColor}25`,
                  color: avatarColor,
                  border: `3px solid ${avatarColor}`,
                }}
              >
                {avatarDisplay}
              </div>
            </div>

            {/* Name */}
            <div>
              <label className="text-[10px] font-medium text-muted-foreground uppercase">
                Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={inputClasses}
                placeholder="Agent name..."
              />
            </div>

            {/* Role */}
            <div>
              <label className="text-[10px] font-medium text-muted-foreground uppercase">
                Role
              </label>
              <input
                type="text"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className={inputClasses}
                placeholder="Agent role..."
              />
            </div>

            {/* Color Picker */}
            <div>
              <label className="text-[10px] font-medium text-muted-foreground uppercase">
                Color
              </label>
              <div className="flex flex-wrap gap-2 mt-1.5">
                {PRESET_COLORS.map((c) => (
                  <button
                    key={c}
                    onClick={() => setAvatarColor(c)}
                    className="w-7 h-7 rounded-full transition-transform hover:scale-110"
                    style={{
                      backgroundColor: c,
                      outline: avatarColor === c ? "2px solid white" : "2px solid transparent",
                      outlineOffset: "2px",
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Emoji */}
            <div>
              <label className="text-[10px] font-medium text-muted-foreground uppercase">
                Emoji (optional, replaces letter)
              </label>
              <input
                type="text"
                value={avatarEmoji}
                onChange={(e) => setAvatarEmoji(e.target.value.slice(0, 2))}
                className={inputClasses}
                placeholder="e.g. \uD83E\uDDD1\u200D\uD83D\uDD2C"
              />
            </div>

            {/* Speech Patterns */}
            <div className="border-t border-border pt-3">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                Speech Patterns
              </div>

              {/* Idle */}
              <div className="mb-3">
                <label className="text-[10px] font-medium text-muted-foreground uppercase">
                  Idle
                </label>
                <input
                  type="text"
                  value={idleMessage}
                  onChange={(e) => setIdleMessage(e.target.value)}
                  className={inputClasses}
                  placeholder="Idle message..."
                />
              </div>

              {/* Working Messages */}
              <div className="mb-3">
                <label className="text-[10px] font-medium text-muted-foreground uppercase">
                  Working
                </label>
                <div className="space-y-1.5 mt-1">
                  {workingMessages.map((msg, i) => (
                    <div key={i} className="flex gap-1">
                      <input
                        type="text"
                        value={msg}
                        onChange={(e) => handleUpdateWorkingMessage(i, e.target.value)}
                        className={inputClasses}
                        placeholder="Working message..."
                      />
                      {workingMessages.length > 1 && (
                        <button
                          onClick={() => handleRemoveWorkingMessage(i)}
                          className="text-muted-foreground hover:text-destructive transition-colors shrink-0 px-1"
                        >
                          <X size={12} />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                <button
                  onClick={handleAddWorkingMessage}
                  className="mt-1.5 text-[10px] text-primary hover:text-primary/80 transition-colors"
                >
                  + Add message
                </button>
              </div>

              {/* Done */}
              <div className="mb-3">
                <label className="text-[10px] font-medium text-muted-foreground uppercase">
                  Done
                </label>
                <input
                  type="text"
                  value={doneMessage}
                  onChange={(e) => setDoneMessage(e.target.value)}
                  className={inputClasses}
                  placeholder="Done message..."
                />
              </div>

              {/* Error */}
              <div>
                <label className="text-[10px] font-medium text-muted-foreground uppercase">
                  Error
                </label>
                <input
                  type="text"
                  value={errorMessage}
                  onChange={(e) => setErrorMessage(e.target.value)}
                  className={inputClasses}
                  placeholder="Error message..."
                />
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-4 py-3 border-t border-border">
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <RotateCcw size={12} />
              Reset to Default
            </button>
            <div className="flex gap-2">
              <button
                onClick={onClose}
                className="px-3 py-1.5 text-xs rounded bg-secondary hover:bg-secondary/80 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
