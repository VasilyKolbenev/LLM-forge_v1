// Character traits for 3D rendering — same data as SimsCharacter.tsx SVG system

export interface CharTraits {
  skinTone: string
  hairStyle: string
  hairColor: string
  glasses: boolean
  glassesStyle?: "round" | "square" | "thin"
  clothing: string
  accessory?: string
  bodyBuild: "slim" | "medium" | "stocky"
  facialHair?: string
}

export const CHARACTER_TRAITS: Record<string, CharTraits> = {
  Flux:     { skinTone: "#d4a574", hairStyle: "short", hairColor: "#2d1b0e", glasses: false, clothing: "tshirt", bodyBuild: "medium", accessory: "watch" },
  Core:     { skinTone: "#f0c8a0", hairStyle: "styled", hairColor: "#1a1a2e", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim" },
  Lyra:     { skinTone: "#f5d6b8", hairStyle: "ponytail", hairColor: "#8b4513", glasses: false, clothing: "hoodie", bodyBuild: "slim", accessory: "earring" },
  Prism:    { skinTone: "#c68642", hairStyle: "curly", hairColor: "#1a0e00", glasses: true, glassesStyle: "square", clothing: "vest", bodyBuild: "medium" },
  Atlas:    { skinTone: "#f0c8a0", hairStyle: "messy", hairColor: "#4a3728", glasses: true, glassesStyle: "thin", clothing: "labcoat", bodyBuild: "stocky", facialHair: "beard" },
  Sage:     { skinTone: "#e8b88a", hairStyle: "bun", hairColor: "#2d1b0e", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim", accessory: "badge" },
  Rune:     { skinTone: "#d4a574", hairStyle: "mohawk", hairColor: "#ff4500", glasses: false, clothing: "jacket", bodyBuild: "medium", accessory: "earring" },
  Themis:   { skinTone: "#f5d6b8", hairStyle: "long", hairColor: "#1a1a2e", glasses: true, glassesStyle: "square", clothing: "shirt", bodyBuild: "slim" },
  Delta:    { skinTone: "#c68642", hairStyle: "buzz", hairColor: "#0a0a0a", glasses: false, clothing: "hoodie", bodyBuild: "stocky", accessory: "headset" },
  Forge:    { skinTone: "#d4a574", hairStyle: "cap", hairColor: "#333", glasses: false, clothing: "vest", bodyBuild: "stocky", accessory: "watch" },
  Beacon:   { skinTone: "#f0c8a0", hairStyle: "short", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "shirt", bodyBuild: "medium" },
  Nexus:    { skinTone: "#e8b88a", hairStyle: "styled", hairColor: "#1a1a2e", glasses: false, clothing: "jacket", bodyBuild: "medium" },
  Switch:   { skinTone: "#f5d6b8", hairStyle: "short", hairColor: "#8b0000", glasses: true, glassesStyle: "square", clothing: "tshirt", bodyBuild: "slim" },
  Duo:      { skinTone: "#c68642", hairStyle: "curly", hairColor: "#1a0e00", glasses: false, clothing: "hoodie", bodyBuild: "medium", accessory: "headset" },
  Bastion:  { skinTone: "#d4a574", hairStyle: "buzz", hairColor: "#2d2d2d", glasses: false, clothing: "vest", bodyBuild: "stocky", facialHair: "goatee", accessory: "badge" },
  Warden:   { skinTone: "#f0c8a0", hairStyle: "bald", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "jacket", bodyBuild: "stocky", facialHair: "mustache" },
  Cipher:   { skinTone: "#e8b88a", hairStyle: "ponytail", hairColor: "#1a1a2e", glasses: false, clothing: "hoodie", bodyBuild: "slim", accessory: "scarf" },
  Vault:    { skinTone: "#d4a574", hairStyle: "messy", hairColor: "#8b4513", glasses: true, glassesStyle: "round", clothing: "tshirt", bodyBuild: "medium" },
  Finch:    { skinTone: "#f5d6b8", hairStyle: "styled", hairColor: "#2d1b0e", glasses: false, clothing: "shirt", bodyBuild: "slim", accessory: "watch" },
  Vera:     { skinTone: "#c68642", hairStyle: "long", hairColor: "#1a0e00", glasses: false, clothing: "labcoat", bodyBuild: "slim", accessory: "badge" },
  Scope:    { skinTone: "#f0c8a0", hairStyle: "cap", hairColor: "#333", glasses: true, glassesStyle: "square", clothing: "hoodie", bodyBuild: "medium", accessory: "headset" },
  Archivist:{ skinTone: "#e8b88a", hairStyle: "bun", hairColor: "#4a3728", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim", facialHair: "goatee" },
  Curator:  { skinTone: "#d4a574", hairStyle: "curly", hairColor: "#2d1b0e", glasses: false, clothing: "vest", bodyBuild: "medium" },
  Auditor:  { skinTone: "#f5d6b8", hairStyle: "styled", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "jacket", bodyBuild: "stocky", accessory: "badge" },
  Claw:     { skinTone: "#c68642", hairStyle: "mohawk", hairColor: "#dc2626", glasses: false, clothing: "jacket", bodyBuild: "stocky", accessory: "earring" },
  Shield:   { skinTone: "#f0c8a0", hairStyle: "buzz", hairColor: "#1a1a2e", glasses: true, glassesStyle: "square", clothing: "vest", bodyBuild: "stocky", facialHair: "beard" },
  Echo:     { skinTone: "#e8b88a", hairStyle: "short", hairColor: "#333", glasses: false, clothing: "tshirt", bodyBuild: "slim" },
  Pulse:    { skinTone: "#d4a574", hairStyle: "messy", hairColor: "#8b4513", glasses: true, glassesStyle: "round", clothing: "hoodie", bodyBuild: "medium" },
  Synth:    { skinTone: "#f5d6b8", hairStyle: "ponytail", hairColor: "#1a0e00", glasses: false, clothing: "labcoat", bodyBuild: "slim", accessory: "earring" },
  Relay:    { skinTone: "#c68642", hairStyle: "styled", hairColor: "#2d1b0e", glasses: true, glassesStyle: "thin", clothing: "shirt", bodyBuild: "medium" },
  Shell:    { skinTone: "#f0c8a0", hairStyle: "cap", hairColor: "#555", glasses: false, clothing: "tshirt", bodyBuild: "medium" },
}

export const DEFAULT_TRAITS: CharTraits = {
  skinTone: "#f0c8a0", hairStyle: "short", hairColor: "#333",
  glasses: false, clothing: "tshirt", bodyBuild: "medium",
}
