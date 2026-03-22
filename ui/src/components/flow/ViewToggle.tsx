import { Grid3X3, Building2, Columns2 } from "lucide-react"

type ViewMode = "dag" | "office" | "split"

interface ViewToggleProps {
  view: ViewMode
  onChange: (view: ViewMode) => void
}

const VIEW_OPTIONS: { value: ViewMode; icon: typeof Grid3X3; label: string }[] = [
  { value: "dag", icon: Grid3X3, label: "DAG View" },
  { value: "office", icon: Building2, label: "Office View" },
  { value: "split", icon: Columns2, label: "Split View" },
]

export function ViewToggle({ view, onChange }: ViewToggleProps) {
  return (
    <div className="flex items-center rounded border border-border overflow-hidden">
      {VIEW_OPTIONS.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          onClick={() => onChange(value)}
          title={label}
          className={`flex items-center justify-center px-2 py-1.5 text-xs transition-colors ${
            view === value
              ? "bg-primary text-primary-foreground"
              : "bg-card text-muted-foreground hover:text-foreground hover:bg-secondary"
          }`}
        >
          <Icon size={14} />
        </button>
      ))}
    </div>
  )
}
