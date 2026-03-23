import { useMemo } from "react"
import * as THREE from "three"

// ===== HEAD =====
export function Head({ skinTone, scale = 1 }: { skinTone: string; scale?: number }) {
  return (
    <mesh position={[0, 0, 0]} scale={scale}>
      <sphereGeometry args={[0.22, 12, 10]} />
      <meshStandardMaterial color={skinTone} roughness={0.8} />
    </mesh>
  )
}

// ===== EYES =====
export function Eyes({ glowColor, glowIntensity = 0 }: { glowColor: number; glowIntensity: number }) {
  const eyeColor = glowIntensity > 0.1 ? glowColor : 0x1a1a2e
  return (
    <group position={[0, 0.02, 0.18]}>
      <mesh position={[-0.07, 0, 0]}>
        <sphereGeometry args={[0.04, 6, 6]} />
        <meshBasicMaterial color={eyeColor} />
      </mesh>
      <mesh position={[0.07, 0, 0]}>
        <sphereGeometry args={[0.04, 6, 6]} />
        <meshBasicMaterial color={eyeColor} />
      </mesh>
      {glowIntensity > 0.1 && (
        <>
          <pointLight position={[-0.07, 0, 0.05]} color={glowColor} intensity={glowIntensity * 0.5} distance={0.5} />
          <pointLight position={[0.07, 0, 0.05]} color={glowColor} intensity={glowIntensity * 0.5} distance={0.5} />
        </>
      )}
    </group>
  )
}

// ===== VR VISOR =====
export function VRVisor({ color, opacity }: { color: number; opacity: number }) {
  if (opacity < 0.05) return null
  return (
    <mesh position={[0, 0.02, 0.19]}>
      <boxGeometry args={[0.35, 0.1, 0.06]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  )
}

// ===== CYBER IMPLANT =====
export function CyberImplant({ color }: { color: number }) {
  return (
    <mesh position={[0.2, 0.05, 0.1]}>
      <sphereGeometry args={[0.025, 6, 6]} />
      <meshBasicMaterial color={color} />
    </mesh>
  )
}

// ===== BODY =====
export function Body({
  color, tronColor, bodyWidth = 0.18,
}: {
  color: string; tronColor: number; bodyWidth?: number
}) {
  return (
    <group>
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[bodyWidth * 2, 0.4, 0.2]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>
      {/* TRON lines on sides */}
      <mesh position={[-bodyWidth + 0.01, 0, 0.101]}>
        <boxGeometry args={[0.02, 0.38, 0.005]} />
        <meshBasicMaterial color={tronColor} transparent opacity={0.6} />
      </mesh>
      <mesh position={[bodyWidth - 0.01, 0, 0.101]}>
        <boxGeometry args={[0.02, 0.38, 0.005]} />
        <meshBasicMaterial color={tronColor} transparent opacity={0.6} />
      </mesh>
      {/* Horizontal TRON line */}
      <mesh position={[0, 0, 0.101]}>
        <boxGeometry args={[bodyWidth * 2 - 0.04, 0.01, 0.005]} />
        <meshBasicMaterial color={tronColor} transparent opacity={0.3} />
      </mesh>
    </group>
  )
}

// ===== ARM =====
export function Arm({ skinTone, side }: { skinTone: string; side: "left" | "right" }) {
  const x = side === "left" ? -0.22 : 0.22
  return (
    <group position={[x, 0.1, 0]}>
      {/* Upper arm */}
      <mesh position={[0, -0.12, 0]}>
        <cylinderGeometry args={[0.04, 0.035, 0.25, 6]} />
        <meshStandardMaterial color={skinTone} roughness={0.8} />
      </mesh>
      {/* Hand */}
      <mesh position={[0, -0.27, 0]}>
        <sphereGeometry args={[0.04, 6, 6]} />
        <meshStandardMaterial color={skinTone} roughness={0.8} />
      </mesh>
    </group>
  )
}

// ===== LEG =====
export function Leg({ side }: { side: "left" | "right" }) {
  const x = side === "left" ? -0.08 : 0.08
  return (
    <group position={[x, 0, 0]}>
      <mesh position={[0, -0.2, 0]}>
        <cylinderGeometry args={[0.05, 0.04, 0.35, 6]} />
        <meshStandardMaterial color="#2a2a3a" roughness={0.8} />
      </mesh>
      {/* Shoe */}
      <mesh position={[0, -0.4, 0.02]}>
        <boxGeometry args={[0.1, 0.06, 0.14]} />
        <meshStandardMaterial color="#1a1a24" roughness={0.9} />
      </mesh>
    </group>
  )
}

// ===== SITTING LEGS =====
export function SittingLegs() {
  return (
    <group>
      {/* Left thigh (horizontal) + lower leg (vertical) */}
      <group position={[-0.08, 0, 0]}>
        <mesh position={[-0.08, -0.05, 0.1]} rotation={[Math.PI / 2.2, 0, 0]}>
          <cylinderGeometry args={[0.05, 0.04, 0.25, 6]} />
          <meshStandardMaterial color="#2a2a3a" roughness={0.8} />
        </mesh>
        <mesh position={[-0.08, -0.2, 0.2]}>
          <cylinderGeometry args={[0.04, 0.04, 0.25, 6]} />
          <meshStandardMaterial color="#2a2a3a" roughness={0.8} />
        </mesh>
        <mesh position={[-0.08, -0.35, 0.22]}>
          <boxGeometry args={[0.1, 0.06, 0.14]} />
          <meshStandardMaterial color="#1a1a24" roughness={0.9} />
        </mesh>
      </group>
      {/* Right leg — mirror */}
      <group position={[0.08, 0, 0]}>
        <mesh position={[0.08, -0.05, 0.1]} rotation={[Math.PI / 2.2, 0, 0]}>
          <cylinderGeometry args={[0.05, 0.04, 0.25, 6]} />
          <meshStandardMaterial color="#2a2a3a" roughness={0.8} />
        </mesh>
        <mesh position={[0.08, -0.2, 0.2]}>
          <cylinderGeometry args={[0.04, 0.04, 0.25, 6]} />
          <meshStandardMaterial color="#2a2a3a" roughness={0.8} />
        </mesh>
        <mesh position={[0.08, -0.35, 0.22]}>
          <boxGeometry args={[0.1, 0.06, 0.14]} />
          <meshStandardMaterial color="#1a1a24" roughness={0.9} />
        </mesh>
      </group>
    </group>
  )
}

// ===== HAIR STYLES =====
export function Hair({ style, color }: { style: string; color: string }) {
  switch (style) {
    case "short":
      return (
        <mesh position={[0, 0.12, 0]}>
          <sphereGeometry args={[0.2, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color={color} roughness={0.9} />
        </mesh>
      )
    case "messy":
      return (
        <group>
          <mesh position={[0, 0.14, 0]}>
            <sphereGeometry args={[0.21, 8, 6, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.9} />
          </mesh>
          {/* Sticking out bits */}
          <mesh position={[-0.15, 0.18, 0]} rotation={[0, 0, 0.5]}>
            <boxGeometry args={[0.06, 0.12, 0.06]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <mesh position={[0.12, 0.2, 0.05]} rotation={[0, 0, -0.3]}>
            <boxGeometry args={[0.05, 0.1, 0.05]} />
            <meshStandardMaterial color={color} />
          </mesh>
        </group>
      )
    case "mohawk":
      return (
        <group>
          {[-0.06, -0.02, 0.02, 0.06].map((z, i) => (
            <mesh key={i} position={[0, 0.22 + i * 0.02, z]}>
              <boxGeometry args={[0.06, 0.12 + i * 0.02, 0.06]} />
              <meshStandardMaterial color={color} />
            </mesh>
          ))}
        </group>
      )
    case "ponytail":
      return (
        <group>
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.2, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.9} />
          </mesh>
          <mesh position={[0, 0.08, -0.22]}>
            <sphereGeometry args={[0.08, 8, 8]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <mesh position={[0, 0.1, -0.15]} rotation={[0.5, 0, 0]}>
            <cylinderGeometry args={[0.03, 0.03, 0.12, 6]} />
            <meshStandardMaterial color={color} />
          </mesh>
        </group>
      )
    case "bun":
      return (
        <group>
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.2, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.9} />
          </mesh>
          <mesh position={[0, 0.28, 0]}>
            <sphereGeometry args={[0.1, 8, 8]} />
            <meshStandardMaterial color={color} />
          </mesh>
        </group>
      )
    case "curly":
      return (
        <group>
          {[[-0.14, 0.1], [-0.05, 0.15], [0.05, 0.15], [0.14, 0.1]].map(([x, y], i) => (
            <mesh key={i} position={[x, y, 0]}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshStandardMaterial color={color} roughness={0.9} />
            </mesh>
          ))}
        </group>
      )
    case "long":
      return (
        <group>
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.21, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.9} />
          </mesh>
          {/* Side strands */}
          <mesh position={[-0.18, -0.1, 0]}>
            <cylinderGeometry args={[0.04, 0.03, 0.3, 6]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <mesh position={[0.18, -0.1, 0]}>
            <cylinderGeometry args={[0.04, 0.03, 0.3, 6]} />
            <meshStandardMaterial color={color} />
          </mesh>
        </group>
      )
    case "cap":
      return (
        <group>
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.2, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.9} />
          </mesh>
          {/* Cap */}
          <mesh position={[0, 0.16, 0]}>
            <cylinderGeometry args={[0.24, 0.24, 0.1, 10]} />
            <meshStandardMaterial color="#444466" roughness={0.8} />
          </mesh>
          {/* Brim */}
          <mesh position={[0, 0.12, 0.12]}>
            <boxGeometry args={[0.2, 0.03, 0.15]} />
            <meshStandardMaterial color="#444466" roughness={0.8} />
          </mesh>
        </group>
      )
    case "styled":
      return (
        <group>
          <mesh position={[0, 0.14, 0]}>
            <sphereGeometry args={[0.22, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
            <meshStandardMaterial color={color} roughness={0.8} />
          </mesh>
          {/* Volume on top */}
          <mesh position={[0, 0.2, 0.05]}>
            <boxGeometry args={[0.2, 0.08, 0.15]} />
            <meshStandardMaterial color={color} roughness={0.8} />
          </mesh>
        </group>
      )
    case "buzz":
      return (
        <mesh position={[0, 0.1, 0]}>
          <sphereGeometry args={[0.19, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color={color} roughness={1} transparent opacity={0.6} />
        </mesh>
      )
    case "bald":
    default:
      return null
  }
}

// ===== HOLOGRAM BUBBLE (above head when working) =====
export function HologramBubble({ color, opacity }: { color: number; opacity: number }) {
  if (opacity < 0.05) return null
  return (
    <group position={[0.4, 0.5, 0]}>
      {/* Frame */}
      <mesh>
        <boxGeometry args={[0.5, 0.35, 0.02]} />
        <meshBasicMaterial color={color} transparent opacity={opacity * 0.15} />
      </mesh>
      {/* Code lines */}
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} position={[-0.1 + i * 0.02, 0.08 - i * 0.06, 0.015]}>
          <boxGeometry args={[0.15 + i * 0.03, 0.02, 0.005]} />
          <meshBasicMaterial color={color} transparent opacity={opacity * 0.5} />
        </mesh>
      ))}
      {/* Connection line to head */}
      <mesh position={[-0.25, -0.3, 0]} rotation={[0, 0, 0.5]}>
        <cylinderGeometry args={[0.005, 0.005, 0.3, 4]} />
        <meshBasicMaterial color={color} transparent opacity={opacity * 0.3} />
      </mesh>
    </group>
  )
}
