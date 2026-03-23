import type { OfficeEnvironment } from "../../EnvironmentPicker"

export interface EnvTheme3D {
  accent: string
  accentHex: number
  secondary: string
  secondaryHex: number
  fogColor: string
  fogHex: number
  floorTint: string
  floorTintHex: number
  ambientIntensity: number
}

export const THEMES_3D: Record<OfficeEnvironment, EnvTheme3D> = {
  "modern-office": {
    accent: "#06b6d4", accentHex: 0x06b6d4,
    secondary: "#8b5cf6", secondaryHex: 0x8b5cf6,
    fogColor: "#050810", fogHex: 0x050810,
    floorTint: "#08090c", floorTintHex: 0x08090c,
    ambientIntensity: 0.15,
  },
  lab: {
    accent: "#22c55e", accentHex: 0x22c55e,
    secondary: "#0ea5e9", secondaryHex: 0x0ea5e9,
    fogColor: "#040a06", fogHex: 0x040a06,
    floorTint: "#060c08", floorTintHex: 0x060c08,
    ambientIntensity: 0.12,
  },
  "command-center": {
    accent: "#8b5cf6", accentHex: 0x8b5cf6,
    secondary: "#ec4899", secondaryHex: 0xec4899,
    fogColor: "#080610", fogHex: 0x080610,
    floorTint: "#0a0810", floorTintHex: 0x0a0810,
    ambientIntensity: 0.1,
  },
  "server-room": {
    accent: "#ef4444", accentHex: 0xef4444,
    secondary: "#f97316", secondaryHex: 0xf97316,
    fogColor: "#0a0606", fogHex: 0x0a0606,
    floorTint: "#0c0808", floorTintHex: 0x0c0808,
    ambientIntensity: 0.1,
  },
  "open-space": {
    accent: "#f59e0b", accentHex: 0xf59e0b,
    secondary: "#84cc16", secondaryHex: 0x84cc16,
    fogColor: "#0a0806", fogHex: 0x0a0806,
    floorTint: "#0c0a06", floorTintHex: 0x0c0a06,
    ambientIntensity: 0.18,
  },
}
