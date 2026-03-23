export type CharacterState = "idle" | "working" | "walking" | "delivering" | "celebrating" | "error"

export interface AgentData3D {
  id: string
  index: number
  worldX: number
  worldY: number
  worldZ: number
  name: string
  color: string
  colorHex: number
  category: string
  characterState: CharacterState
  selected: boolean
}

export interface WalkerData3D {
  id: string
  agentIdx: number
  targetIdx: number
  name: string
  color: string
  colorHex: number
  category: string
  state: CharacterState
  x: number
  y: number
  z: number
  phase: "going" | "delivering" | "returning"
}
