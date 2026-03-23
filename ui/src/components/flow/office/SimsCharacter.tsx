import { memo, useEffect, useMemo, useState } from "react"
import { motion, useAnimation, type Variants } from "framer-motion"

type CharacterState = "idle" | "working" | "walking" | "delivering" | "celebrating" | "error"

interface SimsCharacterProps {
  x: number
  y: number
  name: string
  color: string
  category: string
  state: CharacterState
  sitting?: boolean
  targetX?: number
  targetY?: number
  onArrived?: () => void
}

// Per-character visual traits keyed by persona name
interface CharTraits {
  skinTone: string
  hairStyle: "messy" | "short" | "cap" | "styled" | "long" | "buzz" | "ponytail" | "bun" | "mohawk" | "curly" | "bald"
  hairColor: string
  glasses: boolean
  glassesStyle?: "round" | "square" | "thin"
  clothing: "tshirt" | "shirt" | "hoodie" | "labcoat" | "vest" | "jacket"
  clothingDetail?: string       // collar, buttons, zip line color
  accessory?: "earring" | "badge" | "headset" | "scarf" | "watch"
  bodyBuild: "slim" | "medium" | "stocky"
  facialHair?: "beard" | "goatee" | "mustache"
}

const CHARACTER_TRAITS: Record<string, CharTraits> = {
  // DATA
  Flux:     { skinTone: "#d4a574", hairStyle: "short", hairColor: "#2d1b0e", glasses: false, clothing: "tshirt", bodyBuild: "medium", accessory: "watch" },
  Core:     { skinTone: "#f0c8a0", hairStyle: "styled", hairColor: "#1a1a2e", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim" },
  Lyra:     { skinTone: "#f5d6b8", hairStyle: "ponytail", hairColor: "#8b4513", glasses: false, clothing: "hoodie", bodyBuild: "slim", accessory: "earring" },
  Prism:    { skinTone: "#c68642", hairStyle: "curly", hairColor: "#1a0e00", glasses: true, glassesStyle: "square", clothing: "vest", bodyBuild: "medium" },
  // TRAINING
  Atlas:    { skinTone: "#f0c8a0", hairStyle: "messy", hairColor: "#4a3728", glasses: true, glassesStyle: "thin", clothing: "labcoat", bodyBuild: "stocky", facialHair: "beard" },
  // EVALUATION
  Sage:     { skinTone: "#e8b88a", hairStyle: "bun", hairColor: "#2d1b0e", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim", accessory: "badge" },
  Rune:     { skinTone: "#d4a574", hairStyle: "mohawk", hairColor: "#ff4500", glasses: false, clothing: "jacket", bodyBuild: "medium", accessory: "earring" },
  Themis:   { skinTone: "#f5d6b8", hairStyle: "long", hairColor: "#1a1a2e", glasses: true, glassesStyle: "square", clothing: "shirt", bodyBuild: "slim" },
  Delta:    { skinTone: "#c68642", hairStyle: "buzz", hairColor: "#0a0a0a", glasses: false, clothing: "hoodie", bodyBuild: "stocky", accessory: "headset" },
  // EXPORT
  Forge:    { skinTone: "#d4a574", hairStyle: "cap", hairColor: "#333", glasses: false, clothing: "vest", bodyBuild: "stocky", accessory: "watch" },
  Beacon:   { skinTone: "#f0c8a0", hairStyle: "short", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "shirt", bodyBuild: "medium" },
  // AGENTS
  Nexus:    { skinTone: "#e8b88a", hairStyle: "styled", hairColor: "#1a1a2e", glasses: false, clothing: "jacket", bodyBuild: "medium" },
  Switch:   { skinTone: "#f5d6b8", hairStyle: "short", hairColor: "#8b0000", glasses: true, glassesStyle: "square", clothing: "tshirt", bodyBuild: "slim" },
  Duo:      { skinTone: "#c68642", hairStyle: "curly", hairColor: "#1a0e00", glasses: false, clothing: "hoodie", bodyBuild: "medium", accessory: "headset" },
  Bastion:  { skinTone: "#d4a574", hairStyle: "buzz", hairColor: "#2d2d2d", glasses: false, clothing: "vest", bodyBuild: "stocky", facialHair: "goatee", accessory: "badge" },
  // SAFETY
  Warden:   { skinTone: "#f0c8a0", hairStyle: "bald", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "jacket", bodyBuild: "stocky", facialHair: "mustache" },
  Cipher:   { skinTone: "#e8b88a", hairStyle: "ponytail", hairColor: "#1a1a2e", glasses: false, clothing: "hoodie", bodyBuild: "slim", accessory: "scarf" },
  // OPS
  Vault:    { skinTone: "#d4a574", hairStyle: "messy", hairColor: "#8b4513", glasses: true, glassesStyle: "round", clothing: "tshirt", bodyBuild: "medium" },
  Finch:    { skinTone: "#f5d6b8", hairStyle: "styled", hairColor: "#2d1b0e", glasses: false, clothing: "shirt", bodyBuild: "slim", accessory: "watch" },
  Vera:     { skinTone: "#c68642", hairStyle: "long", hairColor: "#1a0e00", glasses: false, clothing: "labcoat", bodyBuild: "slim", accessory: "badge" },
  Scope:    { skinTone: "#f0c8a0", hairStyle: "cap", hairColor: "#333", glasses: true, glassesStyle: "square", clothing: "hoodie", bodyBuild: "medium", accessory: "headset" },
  // CLOSED LOOP
  Archivist:{ skinTone: "#e8b88a", hairStyle: "bun", hairColor: "#4a3728", glasses: true, glassesStyle: "round", clothing: "shirt", bodyBuild: "slim", facialHair: "goatee" },
  Curator:  { skinTone: "#d4a574", hairStyle: "curly", hairColor: "#2d1b0e", glasses: false, clothing: "vest", bodyBuild: "medium" },
  Auditor:  { skinTone: "#f5d6b8", hairStyle: "styled", hairColor: "#555", glasses: true, glassesStyle: "thin", clothing: "jacket", bodyBuild: "stocky", accessory: "badge" },
  // OPENCLAW
  Claw:     { skinTone: "#c68642", hairStyle: "mohawk", hairColor: "#dc2626", glasses: false, clothing: "jacket", bodyBuild: "stocky", accessory: "earring" },
  Shield:   { skinTone: "#f0c8a0", hairStyle: "buzz", hairColor: "#1a1a2e", glasses: true, glassesStyle: "square", clothing: "vest", bodyBuild: "stocky", facialHair: "beard" },
  // OTHER
  Echo:     { skinTone: "#e8b88a", hairStyle: "short", hairColor: "#333", glasses: false, clothing: "tshirt", bodyBuild: "slim" },
  Pulse:    { skinTone: "#d4a574", hairStyle: "messy", hairColor: "#8b4513", glasses: true, glassesStyle: "round", clothing: "hoodie", bodyBuild: "medium" },
  Synth:    { skinTone: "#f5d6b8", hairStyle: "ponytail", hairColor: "#1a0e00", glasses: false, clothing: "labcoat", bodyBuild: "slim", accessory: "earring" },
  Relay:    { skinTone: "#c68642", hairStyle: "styled", hairColor: "#2d1b0e", glasses: true, glassesStyle: "thin", clothing: "shirt", bodyBuild: "medium" },
  Shell:    { skinTone: "#f0c8a0", hairStyle: "cap", hairColor: "#555", glasses: false, clothing: "tshirt", bodyBuild: "medium" },
}

const DEFAULT_TRAITS: CharTraits = {
  skinTone: "#f0c8a0", hairStyle: "short", hairColor: "#333",
  glasses: false, clothing: "tshirt", bodyBuild: "medium",
}

// Stable hash from name to get consistent variation
function nameHash(name: string): number {
  let h = 0
  for (let i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0
  return Math.abs(h)
}

const BUILD_SIZES = { slim: { bodyW: 7, legLen: 15 }, medium: { bodyW: 9, legLen: 14 }, stocky: { bodyW: 11, legLen: 12 } }

// Animation variants
const idleVariants: Variants = {
  idle: { y: [0, -2, 0], transition: { duration: 2.5, repeat: Infinity, ease: "easeInOut" } },
}
// Body sway while walking
const walkBodySway: Variants = {
  walking: { rotate: [-2, 2, -2], y: [0, -3, 0], transition: { duration: 0.4, repeat: Infinity, ease: "easeInOut" } },
}
const typingLeft: Variants = {
  working: { rotate: [0, -8, 0, -5, 0], transition: { duration: 0.6, repeat: Infinity, ease: "easeInOut" } },
}
const typingRight: Variants = {
  working: { rotate: [0, 8, 0, 5, 0], transition: { duration: 0.5, repeat: Infinity, ease: "easeInOut", delay: 0.15 } },
}
const walkLeft: Variants = {
  walking: { rotate: [15, -15, 15], transition: { duration: 0.4, repeat: Infinity, ease: "easeInOut" } },
}
const walkRight: Variants = {
  walking: { rotate: [-15, 15, -15], transition: { duration: 0.4, repeat: Infinity, ease: "easeInOut" } },
}
const celebrateBody: Variants = {
  celebrating: { y: [0, -8, 0, -5, 0], transition: { duration: 0.8, repeat: 3, ease: "easeOut" } },
}

// --- Hair renderers ---
function HairMessy({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR + 2} ${-headR - 16} Q ${-headR + 5} ${-headR - 24} 0 ${-headR - 22} Q ${headR - 5} ${-headR - 24} ${headR - 2} ${-headR - 16}`}
        fill={color} stroke={`${color}cc`} strokeWidth={0.5} />
      <path d={`M ${-headR + 4} ${-headR - 20} L ${-headR + 1} ${-headR - 26}`} fill="none" stroke={color} strokeWidth={1.5} />
      <path d={`M ${headR - 4} ${-headR - 20} L ${headR - 1} ${-headR - 26}`} fill="none" stroke={color} strokeWidth={1.5} />
    </g>
  )
}
function HairShort({ headR, color }: { headR: number; color: string }) {
  return <path d={`M ${-headR + 1} ${-headR - 12} Q 0 ${-headR - 22} ${headR - 1} ${-headR - 12}`} fill={color} />
}
function HairCap({ headR, color, capColor }: { headR: number; color: string; capColor: string }) {
  return (
    <g>
      <path d={`M ${-headR + 1} ${-headR - 12} Q 0 ${-headR - 20} ${headR - 1} ${-headR - 12}`} fill={color} />
      <rect x={-headR - 1} y={-2 * headR - 5} width={headR * 2 + 2} height={headR * 0.7} rx={2} fill={capColor} />
      <rect x={headR - 2} y={-2 * headR + headR * 0.7 - 6} width={8} height={3} rx={1} fill={`${capColor}cc`} />
    </g>
  )
}
function HairStyled({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR} ${-headR - 14} Q ${-headR + 3} ${-headR - 25} 0 ${-headR - 23} Q ${headR - 3} ${-headR - 25} ${headR} ${-headR - 14}`}
        fill={color} stroke={`${color}aa`} strokeWidth={0.8} />
      <path d={`M ${-headR + 3} ${-headR - 20} Q 0 ${-headR - 27} ${headR - 3} ${-headR - 20}`}
        fill="none" stroke={`${color}dd`} strokeWidth={1} />
    </g>
  )
}
function HairLong({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR - 1} ${-headR - 10} Q ${-headR - 3} ${-headR - 22} 0 ${-headR - 20} Q ${headR + 3} ${-headR - 22} ${headR + 1} ${-headR - 10}`}
        fill={color} />
      <path d={`M ${-headR - 1} ${-headR - 10} Q ${-headR - 4} ${-headR + 8} ${-headR - 2} ${-headR + 16}`}
        fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" />
      <path d={`M ${headR + 1} ${-headR - 10} Q ${headR + 4} ${-headR + 8} ${headR + 2} ${-headR + 16}`}
        fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" />
    </g>
  )
}
function HairBuzz({ headR, color }: { headR: number; color: string }) {
  return <path d={`M ${-headR + 1} ${-headR - 10} Q 0 ${-headR - 19} ${headR - 1} ${-headR - 10}`} fill={`${color}80`} />
}
function HairPonytail({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR + 1} ${-headR - 12} Q 0 ${-headR - 22} ${headR - 1} ${-headR - 12}`} fill={color} />
      <path d={`M ${headR - 2} ${-headR - 12} Q ${headR + 10} ${-headR - 8} ${headR + 8} ${-headR + 10}`}
        fill="none" stroke={color} strokeWidth={3} strokeLinecap="round" />
      <circle cx={headR + 8} cy={-headR + 12} r={3} fill={color} />
    </g>
  )
}
function HairBun({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR + 1} ${-headR - 12} Q 0 ${-headR - 22} ${headR - 1} ${-headR - 12}`} fill={color} />
      <circle cx={0} cy={-2 * headR - 6} r={6} fill={color} stroke={`${color}cc`} strokeWidth={0.5} />
    </g>
  )
}
function HairMohawk({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      <path d={`M ${-headR + 3} ${-headR - 12} L ${-headR + 3} ${-headR - 12}`} fill={`${color}40`} />
      {[-6, -2, 2, 6].map((dx, i) => (
        <rect key={i} x={dx - 2} y={-2 * headR - 8 - i * 2} width={4} height={10 + i} rx={1} fill={color} opacity={1 - i * 0.1} />
      ))}
    </g>
  )
}
function HairCurly({ headR, color }: { headR: number; color: string }) {
  return (
    <g>
      {[-8, -3, 2, 7].map((dx, i) => (
        <circle key={i} cx={dx} cy={-2 * headR + 2 - i % 2 * 3} r={5 + (i % 2)} fill={color} opacity={0.9} />
      ))}
      <circle cx={-headR + 1} cy={-headR - 6} r={4} fill={color} />
      <circle cx={headR - 1} cy={-headR - 6} r={4} fill={color} />
    </g>
  )
}
function HairBald() { return null }

const HAIR_RENDERERS: Record<string, React.FC<{ headR: number; color: string; capColor?: string }>> = {
  messy: HairMessy, short: HairShort, cap: HairCap, styled: HairStyled,
  long: HairLong, buzz: HairBuzz, ponytail: HairPonytail, bun: HairBun,
  mohawk: HairMohawk, curly: HairCurly, bald: HairBald,
}

// --- Glasses renderers ---
function GlassesRound({ headR }: { headR: number }) {
  return (
    <g>
      <circle cx={-4} cy={-headR - 9} r={4} fill="none" stroke="#888" strokeWidth={0.8} />
      <circle cx={4} cy={-headR - 9} r={4} fill="none" stroke="#888" strokeWidth={0.8} />
      <line x1={0} y1={-headR - 9} x2={0} y2={-headR - 9} stroke="#888" strokeWidth={0.5} />
      <line x1={-8} y1={-headR - 9} x2={-headR + 1} y2={-headR - 8} stroke="#888" strokeWidth={0.5} />
      <line x1={8} y1={-headR - 9} x2={headR - 1} y2={-headR - 8} stroke="#888" strokeWidth={0.5} />
      {/* Lens reflection */}
      <circle cx={-3} cy={-headR - 10} r={1} fill="white" opacity={0.2} />
      <circle cx={5} cy={-headR - 10} r={1} fill="white" opacity={0.2} />
    </g>
  )
}
function GlassesSquare({ headR }: { headR: number }) {
  return (
    <g>
      <rect x={-7.5} y={-headR - 12} width={7} height={6} rx={1} fill="none" stroke="#666" strokeWidth={0.8} />
      <rect x={0.5} y={-headR - 12} width={7} height={6} rx={1} fill="none" stroke="#666" strokeWidth={0.8} />
      <line x1={-0.5} y1={-headR - 9} x2={0.5} y2={-headR - 9} stroke="#666" strokeWidth={0.5} />
      <line x1={-7.5} y1={-headR - 9} x2={-headR + 1} y2={-headR - 8} stroke="#666" strokeWidth={0.5} />
      <line x1={7.5} y1={-headR - 9} x2={headR - 1} y2={-headR - 8} stroke="#666" strokeWidth={0.5} />
    </g>
  )
}
function GlassesThin({ headR }: { headR: number }) {
  return (
    <g>
      <ellipse cx={-4} cy={-headR - 9} rx={5} ry={2.5} fill="none" stroke="#aaa" strokeWidth={0.6} />
      <ellipse cx={4} cy={-headR - 9} rx={5} ry={2.5} fill="none" stroke="#aaa" strokeWidth={0.6} />
      <line x1={-9} y1={-headR - 9} x2={-headR} y2={-headR - 8} stroke="#aaa" strokeWidth={0.4} />
      <line x1={9} y1={-headR - 9} x2={headR} y2={-headR - 8} stroke="#aaa" strokeWidth={0.4} />
    </g>
  )
}

// --- Clothing renderers ---
function ClothingTshirt({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      <rect x={-bodyW} y={-4} width={bodyW * 2} height={18} rx={3} fill={color} />
      {/* Collar V */}
      <path d={`M ${-3} -4 L 0 2 L 3 -4`} fill="none" stroke={`${color}80`} strokeWidth={0.8} />
    </g>
  )
}
function ClothingShirt({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      <rect x={-bodyW} y={-4} width={bodyW * 2} height={18} rx={2} fill={color} />
      {/* Collar */}
      <path d={`M ${-4} -4 L ${-2} 2 L 0 -1 L 2 2 L 4 -4`} fill={`${color}dd`} stroke={`${color}60`} strokeWidth={0.5} />
      {/* Buttons */}
      <circle cx={0} cy={4} r={1} fill={`${color}60`} stroke="white" strokeWidth={0.3} />
      <circle cx={0} cy={8} r={1} fill={`${color}60`} stroke="white" strokeWidth={0.3} />
      <circle cx={0} cy={12} r={1} fill={`${color}60`} stroke="white" strokeWidth={0.3} />
    </g>
  )
}
function ClothingHoodie({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      <rect x={-bodyW} y={-4} width={bodyW * 2} height={18} rx={4} fill={color} />
      {/* Hood outline */}
      <path d={`M ${-bodyW + 2} -4 Q ${-bodyW + 2} -10 0 -8 Q ${bodyW - 2} -10 ${bodyW - 2} -4`}
        fill={`${color}cc`} stroke={`${color}80`} strokeWidth={0.5} />
      {/* Pocket */}
      <rect x={-bodyW + 3} y={8} width={bodyW * 2 - 6} height={6} rx={2} fill={`${color}bb`} stroke={`${color}60`} strokeWidth={0.3} />
      {/* Drawstrings */}
      <line x1={-2} y1={-2} x2={-2} y2={5} stroke={`${color}60`} strokeWidth={0.5} />
      <line x1={2} y1={-2} x2={2} y2={5} stroke={`${color}60`} strokeWidth={0.5} />
    </g>
  )
}
function ClothingLabcoat({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      {/* Inner shirt */}
      <rect x={-bodyW + 2} y={-2} width={bodyW * 2 - 4} height={16} rx={2} fill={color} />
      {/* White coat */}
      <rect x={-bodyW - 1} y={-4} width={bodyW * 2 + 2} height={20} rx={2} fill="#e8e8e8" fillOpacity={0.9} stroke="#ccc" strokeWidth={0.5} />
      {/* Lapels */}
      <path d={`M 0 -4 L -4 4`} fill="none" stroke="#bbb" strokeWidth={0.5} />
      <path d={`M 0 -4 L 4 4`} fill="none" stroke="#bbb" strokeWidth={0.5} />
      {/* Pocket with pen */}
      <rect x={-bodyW + 1} y={4} width={6} height={5} rx={1} fill="#ddd" stroke="#bbb" strokeWidth={0.3} />
      <line x1={-bodyW + 3} y1={3} x2={-bodyW + 3} y2={7} stroke="#3b82f6" strokeWidth={0.8} />
    </g>
  )
}
function ClothingVest({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      {/* Under-tshirt */}
      <rect x={-bodyW} y={-4} width={bodyW * 2} height={18} rx={3} fill="#333" />
      {/* Vest */}
      <rect x={-bodyW + 1} y={-3} width={bodyW - 2} height={17} rx={2} fill={color} />
      <rect x={1} y={-3} width={bodyW - 2} height={17} rx={2} fill={color} />
      {/* Zipper */}
      <line x1={0} y1={-2} x2={0} y2={14} stroke="#666" strokeWidth={0.5} strokeDasharray="2 1" />
    </g>
  )
}
function ClothingJacket({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      <rect x={-bodyW} y={-4} width={bodyW * 2} height={18} rx={2} fill={color} />
      {/* Collar high */}
      <rect x={-bodyW + 1} y={-6} width={bodyW * 2 - 2} height={4} rx={1.5} fill={`${color}dd`} />
      {/* Zipper */}
      <line x1={0} y1={-4} x2={0} y2={14} stroke="#888" strokeWidth={0.8} />
      {/* Zipper pull */}
      <rect x={-1} y={3} width={2} height={3} rx={0.5} fill="#aaa" />
      {/* Pockets */}
      <line x1={-bodyW + 2} y1={8} x2={-3} y2={8} stroke={`${color}80`} strokeWidth={0.5} />
      <line x1={3} y1={8} x2={bodyW - 2} y2={8} stroke={`${color}80`} strokeWidth={0.5} />
    </g>
  )
}

const CLOTHING_RENDERERS: Record<string, React.FC<{ bodyW: number; color: string }>> = {
  tshirt: ClothingTshirt, shirt: ClothingShirt, hoodie: ClothingHoodie,
  labcoat: ClothingLabcoat, vest: ClothingVest, jacket: ClothingJacket,
}

// --- Accessory renderers ---
function AccessoryBadge({ bodyW }: { bodyW: number }) {
  return (
    <g>
      <rect x={bodyW - 5} y={-2} width={5} height={7} rx={1} fill="#fbbf24" stroke="#d97706" strokeWidth={0.3} />
      <circle cx={bodyW - 2.5} cy={1} r={1.5} fill="#d97706" />
    </g>
  )
}
function AccessoryHeadset({ headR }: { headR: number }) {
  return (
    <g>
      <path d={`M ${-headR - 2} ${-headR - 6} Q ${-headR - 4} ${-headR - 16} 0 ${-headR - 18} Q ${headR + 4} ${-headR - 16} ${headR + 2} ${-headR - 6}`}
        fill="none" stroke="#555" strokeWidth={1.5} />
      <rect x={-headR - 4} y={-headR - 8} width={4} height={6} rx={2} fill="#444" />
      <rect x={headR} y={-headR - 8} width={4} height={6} rx={2} fill="#444" />
      <path d={`M ${-headR - 2} ${-headR - 3} Q ${-headR - 5} ${-headR + 2} ${-headR - 3} ${-headR + 5}`}
        fill="none" stroke="#555" strokeWidth={1} />
      <circle cx={-headR - 3} cy={-headR + 6} r={2} fill="#444" />
    </g>
  )
}
function AccessoryScarf({ bodyW, color }: { bodyW: number; color: string }) {
  return (
    <g>
      <path d={`M ${-bodyW - 1} -4 Q 0 0 ${bodyW + 1} -4`} fill={color} stroke={`${color}80`} strokeWidth={0.5} />
      <path d={`M ${bodyW} -3 Q ${bodyW + 3} 4 ${bodyW + 1} 10`} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" />
    </g>
  )
}
function AccessoryEarring({ headR }: { headR: number }) {
  return <circle cx={-headR - 1} cy={-headR - 5} r={1.5} fill="#fbbf24" stroke="#d97706" strokeWidth={0.3} />
}
function AccessoryWatch({ bodyW }: { bodyW: number }) {
  return (
    <g>
      <rect x={-bodyW - 10} y={11} width={5} height={3} rx={1} fill="#555" stroke="#777" strokeWidth={0.3} />
      <rect x={-bodyW - 9} y={11.5} width={3} height={2} rx={0.5} fill="#1a1a2e" />
    </g>
  )
}

// --- Facial hair ---
function BeardFull({ headR }: { headR: number }) {
  return (
    <path d={`M ${-headR + 3} ${-headR - 2} Q ${-headR + 2} ${-headR + 8} 0 ${-headR + 10} Q ${headR - 2} ${-headR + 8} ${headR - 3} ${-headR - 2}`}
      fill="#3d2b1f" opacity={0.6} />
  )
}
function Goatee({ headR }: { headR: number }) {
  return (
    <path d={`M -3 ${-headR - 2} Q -4 ${-headR + 4} 0 ${-headR + 6} Q 4 ${-headR + 4} 3 ${-headR - 2}`}
      fill="#3d2b1f" opacity={0.5} />
  )
}
function Mustache({ headR }: { headR: number }) {
  return (
    <path d={`M -5 ${-headR - 4} Q -3 ${-headR - 2} 0 ${-headR - 3} Q 3 ${-headR - 2} 5 ${-headR - 4}`}
      fill="#3d2b1f" opacity={0.6} strokeWidth={0.5} />
  )
}

function SimsCharacterInner({
  x, y, name, color, category, state, sitting = false, targetX, targetY, onArrived,
}: SimsCharacterProps) {
  const traits = useMemo(() => CHARACTER_TRAITS[name] ?? DEFAULT_TRAITS, [name])
  const controls = useAnimation()
  const [isWalking, setIsWalking] = useState(false)

  const build = BUILD_SIZES[traits.bodyBuild]
  const headR = 12

  // Walk to target
  useEffect(() => {
    if (state === "walking" && targetX !== undefined && targetY !== undefined) {
      setIsWalking(true)
      const dist = Math.sqrt((targetX - x) ** 2 + (targetY - y) ** 2)
      controls.start({ x: targetX, y: targetY, transition: { duration: Math.max(dist / 80, 0.8), ease: "easeInOut" } })
        .then(() => { setIsWalking(false); onArrived?.() })
    }
  }, [state, targetX, targetY])

  const Clothing = CLOTHING_RENDERERS[traits.clothing] ?? ClothingTshirt
  const HairRenderer = HAIR_RENDERERS[traits.hairStyle] ?? HairShort

  const thoughtContent = state === "working" ? "..." : state === "error" ? "!" : state === "celebrating" ? "✓" : null

  return (
    <motion.g
      initial={{ x, y, opacity: 0 }}
      animate={isWalking ? controls : { x, y, opacity: 1 }}
      transition={{ opacity: { duration: 0.5 } }}
      style={{ cursor: "pointer" }}
    >
      {/* Neon ground ring */}
      <motion.ellipse cx={0} cy={sitting ? 26 : build.legLen + 20} rx={16} ry={6}
        fill="none" stroke={color} strokeWidth={0.8}
        opacity={state === "working" ? 0.4 : state === "celebrating" ? 0.6 : 0.12}
        animate={state === "working" ? { opacity: [0.2, 0.5, 0.2] } : state === "celebrating" ? { rx: [16, 25, 16], ry: [6, 10, 6], opacity: [0.6, 0.1, 0.6] } : undefined}
        transition={{ duration: state === "celebrating" ? 0.8 : 1.5, repeat: state === "celebrating" ? 3 : Infinity }}
      />
      {/* Neon shadow */}
      <ellipse cx={0} cy={sitting ? 24 : build.legLen + 18} rx={12} ry={4} fill={color} opacity={0.06} />

      <motion.g
        variants={
          state === "celebrating" ? celebrateBody
          : (state === "walking" || isWalking) ? walkBodySway
          : idleVariants
        }
        animate={
          state === "celebrating" ? "celebrating"
          : (state === "walking" || isWalking) ? "walking"
          : "idle"
        }
      >
        {sitting ? (
          <>
            {/* Sitting legs — thighs horizontal, lower legs vertical */}
            {/* Left thigh */}
            <line x1={-5} y1={12} x2={-12} y2={16} stroke="#334" strokeWidth={4} strokeLinecap="round" />
            {/* Left lower leg (hanging down) */}
            <line x1={-12} y1={16} x2={-12} y2={26} stroke="#334" strokeWidth={3.5} strokeLinecap="round" />
            {/* Right thigh */}
            <line x1={5} y1={12} x2={12} y2={16} stroke="#334" strokeWidth={4} strokeLinecap="round" />
            {/* Right lower leg */}
            <line x1={12} y1={16} x2={12} y2={26} stroke="#334" strokeWidth={3.5} strokeLinecap="round" />
            {/* Shoes */}
            <ellipse cx={-12} cy={27} rx={4.5} ry={2} fill="#222" />
            <ellipse cx={12} cy={27} rx={4.5} ry={2} fill="#222" />
          </>
        ) : (
          <>
            {/* Standing legs — animate when walking */}
            <motion.line x1={-4} y1={12} x2={-4} y2={12 + build.legLen}
              stroke="#334" strokeWidth={4} strokeLinecap="round"
              variants={(isWalking || state === "walking") ? walkLeft : undefined}
              animate={(isWalking || state === "walking") ? "walking" : undefined}
              style={{ transformOrigin: "-4px 12px" }} />
            <motion.line x1={4} y1={12} x2={4} y2={12 + build.legLen}
              stroke="#334" strokeWidth={4} strokeLinecap="round"
              variants={(isWalking || state === "walking") ? walkRight : undefined}
              animate={(isWalking || state === "walking") ? "walking" : undefined}
              style={{ transformOrigin: "4px 12px" }} />
            {/* Shoes */}
            <ellipse cx={-5} cy={12 + build.legLen + 2} rx={5} ry={2.5} fill="#222" />
            <ellipse cx={5} cy={12 + build.legLen + 2} rx={5} ry={2.5} fill="#222" />
            <ellipse cx={-5} cy={12 + build.legLen + 1} rx={4} ry={1.5} fill="#333" />
            <ellipse cx={5} cy={12 + build.legLen + 1} rx={4} ry={1.5} fill="#333" />
          </>
        )}

        {/* Clothing / Torso */}
        <Clothing bodyW={build.bodyW} color={color} />

        {/* === CYBERPUNK: TRON lines on body === */}
        <line x1={-build.bodyW + 1} y1={-2} x2={-build.bodyW + 1} y2={14}
          stroke={color} strokeWidth={0.6} opacity={0.4} />
        <line x1={build.bodyW - 1} y1={-2} x2={build.bodyW - 1} y2={14}
          stroke={color} strokeWidth={0.6} opacity={0.4} />
        <line x1={-build.bodyW + 1} y1={6} x2={build.bodyW - 1} y2={6}
          stroke={color} strokeWidth={0.3} opacity={0.2} />

        {/* Badge accessory */}
        {traits.accessory === "badge" && <AccessoryBadge bodyW={build.bodyW} />}

        {/* Left arm */}
        <motion.g
          variants={state === "working" ? typingLeft : undefined}
          animate={state === "working" ? "working" : undefined}
          style={{ transformOrigin: `${-build.bodyW - 1}px 0px` }}
        >
          <line x1={-build.bodyW - 1} y1={0} x2={-build.bodyW - 8} y2={12}
            stroke={traits.skinTone} strokeWidth={3.5} strokeLinecap="round" />
          <circle cx={-build.bodyW - 8} cy={13} r={2.5} fill={traits.skinTone} />
          {traits.accessory === "watch" && <AccessoryWatch bodyW={build.bodyW} />}
        </motion.g>

        {/* Right arm */}
        <motion.g
          variants={state === "working" ? typingRight : undefined}
          animate={state === "working" ? "working" : undefined}
          style={{ transformOrigin: `${build.bodyW + 1}px 0px` }}
        >
          <line x1={build.bodyW + 1} y1={0} x2={build.bodyW + 8} y2={12}
            stroke={traits.skinTone} strokeWidth={3.5} strokeLinecap="round" />
          {state === "delivering" ? (
            <g>
              {/* DATA CUBE — glowing holographic cube */}
              <motion.g animate={{ y: [0, -2, 0], rotate: [0, 10, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}>
                <polygon points={`${build.bodyW + 10},4 ${build.bodyW + 16},7 ${build.bodyW + 10},10 ${build.bodyW + 4},7`}
                  fill={`${color}40`} stroke={color} strokeWidth={0.6} />
                <polygon points={`${build.bodyW + 4},7 ${build.bodyW + 10},10 ${build.bodyW + 10},16 ${build.bodyW + 4},13`}
                  fill={`${color}20`} stroke={color} strokeWidth={0.4} />
                <polygon points={`${build.bodyW + 10},10 ${build.bodyW + 16},7 ${build.bodyW + 16},13 ${build.bodyW + 10},16`}
                  fill={`${color}30`} stroke={color} strokeWidth={0.4} />
              </motion.g>
              <circle cx={build.bodyW + 10} cy={10} r={8} fill={color} opacity={0.08} />
            </g>
          ) : (
            <circle cx={build.bodyW + 8} cy={13} r={2.5} fill={traits.skinTone} />
          )}
        </motion.g>

        {/* Scarf accessory (over clothing) */}
        {traits.accessory === "scarf" && <AccessoryScarf bodyW={build.bodyW} color="#ef4444" />}

        {/* Neck */}
        <rect x={-2.5} y={-8} width={5} height={5} fill={traits.skinTone} />

        {/* Head */}
        <circle cx={0} cy={-headR - 8} r={headR} fill={traits.skinTone} />

        {/* Facial hair */}
        {traits.facialHair === "beard" && <BeardFull headR={headR} />}
        {traits.facialHair === "goatee" && <Goatee headR={headR} />}
        {traits.facialHair === "mustache" && <Mustache headR={headR} />}

        {/* Eyes — with NEON GLOW when working */}
        <motion.g
          animate={state === "working" ? { y: [0, 1, 0] } : undefined}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <ellipse cx={-4} cy={-headR - 9} rx={1.8} ry={2} fill={state === "working" ? color : "#1a1a2e"} />
          <ellipse cx={4} cy={-headR - 9} rx={1.8} ry={2} fill={state === "working" ? color : "#1a1a2e"} />
          {state === "working" ? (
            <>
              {/* Eye glow */}
              <circle cx={-4} cy={-headR - 9} r={4} fill={color} opacity={0.15} />
              <circle cx={4} cy={-headR - 9} r={4} fill={color} opacity={0.15} />
            </>
          ) : (
            <>
              <circle cx={-3.3} cy={-headR - 9.5} r={0.8} fill="white" />
              <circle cx={4.7} cy={-headR - 9.5} r={0.8} fill="white" />
            </>
          )}
        </motion.g>

        {/* Glasses / VR visor */}
        {state === "working" ? (
          /* Cyber VR visor — glowing strip across eyes */
          <g>
            <rect x={-headR + 2} y={-headR - 12} width={headR * 2 - 4} height={5} rx={2}
              fill={`${color}30`} stroke={color} strokeWidth={0.5} />
            <motion.rect x={-headR + 3} y={-headR - 11} width={headR * 2 - 6} height={3} rx={1}
              fill={color} opacity={0.2}
              animate={{ opacity: [0.1, 0.3, 0.1] }}
              transition={{ duration: 1.5, repeat: Infinity }} />
          </g>
        ) : (
          <>
            {traits.glasses && traits.glassesStyle === "round" && <GlassesRound headR={headR} />}
            {traits.glasses && traits.glassesStyle === "square" && <GlassesSquare headR={headR} />}
            {traits.glasses && traits.glassesStyle === "thin" && <GlassesThin headR={headR} />}
          </>
        )}

        {/* Cyber implant — glowing dot on temple */}
        <motion.circle cx={headR - 1} cy={-headR - 7} r={1.5}
          fill={color} opacity={0.5}
          animate={{ opacity: [0.3, 0.8, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
        />

        {/* Mouth */}
        {state === "celebrating" ? (
          <path d={`M -3 ${-headR - 4} Q 0 ${-headR - 1} 3 ${-headR - 4}`} fill="none" stroke="#1a1a2e" strokeWidth={1} />
        ) : state === "error" ? (
          <path d={`M -3 ${-headR - 3} Q 0 ${-headR - 5} 3 ${-headR - 3}`} fill="none" stroke="#1a1a2e" strokeWidth={1} />
        ) : (
          <line x1={-2} y1={-headR - 4} x2={2} y2={-headR - 4} stroke="#1a1a2e" strokeWidth={1} strokeLinecap="round" />
        )}

        {/* Earring accessory */}
        {traits.accessory === "earring" && <AccessoryEarring headR={headR} />}

        {/* Hair */}
        <HairRenderer headR={headR} color={traits.hairColor} capColor={color} />

        {/* Headset accessory (over hair) */}
        {traits.accessory === "headset" && <AccessoryHeadset headR={headR} />}

        {/* === HOLOGRAM THOUGHT BUBBLE === */}
        {state === "working" && (
          <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
            {/* Hologram code display above head */}
            <g transform={`translate(${headR + 14}, ${-2 * headR - 25})`}>
              <rect x={-14} y={-10} width={28} height={18} rx={2}
                fill={`${color}08`} stroke={color} strokeWidth={0.5} opacity={0.7} />
              {/* Running code lines */}
              {[0, 1, 2, 3].map((i) => (
                <motion.line key={i}
                  x1={-10} y1={-6 + i * 4} x2={-10 + 8 + i * 2} y2={-6 + i * 4}
                  stroke={color} strokeWidth={0.7} opacity={0.5}
                  animate={{ x2: [-10, -10 + 8 + i * 2, -10], opacity: [0.2, 0.6, 0.2] }}
                  transition={{ duration: 1 + i * 0.3, repeat: Infinity, delay: i * 0.2 }}
                />
              ))}
              {/* Connection line to head */}
              <line x1={-14} y1={8} x2={-headR - 4} y2={15}
                stroke={color} strokeWidth={0.3} opacity={0.3} strokeDasharray="2 2" />
            </g>
          </motion.g>
        )}

        {state === "error" && (
          <motion.g
            animate={{ x: [-1, 1, -1, 0] }}
            transition={{ duration: 0.15, repeat: 5 }}
          >
            {/* Glitch overlay */}
            <rect x={-headR} y={-2 * headR - 5} width={headR * 2} height={headR * 2 + 20}
              fill="#ef4444" opacity={0.05} />
            {/* Error symbol */}
            <g transform={`translate(${headR + 14}, ${-2 * headR - 18})`}>
              <motion.polygon points="0,-8 9,6 -9,6" fill="none" stroke="#ef4444" strokeWidth={1}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 0.5, repeat: Infinity }} />
              <text x={0} y={3} textAnchor="middle" fill="#ef4444" fontSize={10} fontWeight="bold">!</text>
            </g>
          </motion.g>
        )}

        {state === "celebrating" && (
          <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {/* Particle burst */}
            {[0, 60, 120, 180, 240, 300].map((angle, i) => (
              <motion.circle key={i}
                cx={0} cy={-headR - 8} r={2}
                fill={color}
                animate={{
                  cx: [0, Math.cos(angle * Math.PI / 180) * 25],
                  cy: [-headR - 8, -headR - 8 + Math.sin(angle * Math.PI / 180) * 25],
                  opacity: [0.8, 0],
                  r: [2, 0.5],
                }}
                transition={{ duration: 0.8, repeat: 3, delay: i * 0.05 }}
              />
            ))}
            <motion.text x={0} y={-2 * headR - 15} textAnchor="middle" fill={color} fontSize={10}
              animate={{ y: [-2 * headR - 15, -2 * headR - 25], opacity: [1, 0] }}
              transition={{ duration: 1.5, repeat: 2 }}>
              ✓
            </motion.text>
          </motion.g>
        )}
      </motion.g>

      {/* Neon name tag */}
      {name && (
        <g style={{ pointerEvents: "none", userSelect: "none" }}>
          <text x={0} y={sitting ? 38 : build.legLen + 32} textAnchor="middle"
            fill={color} fontSize={8} fontWeight="600" opacity={0.9}
            filter="url(#neonGlow)">
            {name}
          </text>
          <text x={0} y={sitting ? 38 : build.legLen + 32} textAnchor="middle"
            fill="white" fontSize={8} fontWeight="600" opacity={0.7}>
            {name}
          </text>
        </g>
      )}
    </motion.g>
  )
}

export const SimsCharacter = memo(SimsCharacterInner)
