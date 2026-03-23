import { useRef, useMemo } from "react"
import { useFrame } from "@react-three/fiber"
import { Text, Billboard } from "@react-three/drei"
import * as THREE from "three"
import type { CharacterState } from "./types"
import { useCharacterAnimation } from "./hooks/useCharacterAnimation"
import { Head, Eyes, VRVisor, CyberImplant, Body, Arm, Leg, SittingLegs, Hair, HologramBubble } from "./CharacterParts"

// Import character traits from SVG system
import { CHARACTER_TRAITS, DEFAULT_TRAITS } from "./characterTraits"

interface AgentCharacterProps {
  position: [number, number, number]
  name: string
  color: string
  colorHex: number
  state: CharacterState
  sitting?: boolean
  onClick?: () => void
  onDoubleClick?: () => void
}

const BUILD_SIZES = {
  slim: { bodyW: 0.15, legLen: 0.35 },
  medium: { bodyW: 0.18, legLen: 0.32 },
  stocky: { bodyW: 0.22, legLen: 0.28 },
}

export function AgentCharacter({
  position, name, color, colorHex, state, sitting = false,
  onClick, onDoubleClick,
}: AgentCharacterProps) {
  const groupRef = useRef<THREE.Group>(null)
  const leftArmRef = useRef<THREE.Group>(null)
  const rightArmRef = useRef<THREE.Group>(null)
  const leftLegRef = useRef<THREE.Group>(null)
  const rightLegRef = useRef<THREE.Group>(null)

  const traits = useMemo(() => CHARACTER_TRAITS[name] ?? DEFAULT_TRAITS, [name])
  const build = BUILD_SIZES[traits.bodyBuild]
  const anim = useCharacterAnimation(state)

  // Apply animation each frame
  useFrame(() => {
    const a = anim.current
    if (groupRef.current) {
      groupRef.current.position.y = position[1] + (sitting ? 0.78 : 0.6) + a.bodyY
      groupRef.current.position.x = position[0] + a.shakeX
      groupRef.current.rotation.x = a.bodyRotX
    }
    if (leftArmRef.current) leftArmRef.current.rotation.x = a.leftArmRotX
    if (rightArmRef.current) rightArmRef.current.rotation.x = a.rightArmRotX
    if (!sitting) {
      if (leftLegRef.current) leftLegRef.current.rotation.x = a.leftLegRotX
      if (rightLegRef.current) rightLegRef.current.rotation.x = a.rightLegRotX
    }
  })

  const headY = 0.42
  const bodyY = 0.15

  return (
    <group
      ref={groupRef}
      position={[position[0], position[1] + (sitting ? 0.78 : 0.6), position[2]]}
      onClick={(e) => { e.stopPropagation(); onClick?.() }}
      onDoubleClick={(e) => { e.stopPropagation(); onDoubleClick?.() }}
    >
      {/* Legs */}
      {sitting ? (
        <group position={[0, -0.2, 0]}>
          <SittingLegs />
        </group>
      ) : (
        <group position={[0, -0.2, 0]}>
          <group ref={leftLegRef}><Leg side="left" /></group>
          <group ref={rightLegRef}><Leg side="right" /></group>
        </group>
      )}

      {/* Body / Clothing */}
      <group position={[0, bodyY, 0]}>
        <Body color={color} tronColor={colorHex} bodyWidth={build.bodyW} />
      </group>

      {/* Left arm */}
      <group ref={leftArmRef} position={[0, bodyY + 0.1, 0]}>
        <Arm skinTone={traits.skinTone} side="left" />
      </group>

      {/* Right arm */}
      <group ref={rightArmRef} position={[0, bodyY + 0.1, 0]}>
        <Arm skinTone={traits.skinTone} side="right" />

        {/* Data cube when delivering */}
        {state === "delivering" && (
          <group position={[0.25, -0.2, 0]}>
            <mesh>
              <boxGeometry args={[0.12, 0.12, 0.12]} />
              <meshBasicMaterial color={colorHex} transparent opacity={0.5} />
            </mesh>
            <pointLight color={colorHex} intensity={0.5} distance={1} />
          </group>
        )}
      </group>

      {/* Head */}
      <group position={[0, headY, 0]}>
        <Head skinTone={traits.skinTone} />
        <Eyes glowColor={colorHex} glowIntensity={anim.current.eyeGlow} />
        <VRVisor color={colorHex} opacity={anim.current.visorOpacity} />
        <CyberImplant color={colorHex} />
        <Hair style={traits.hairStyle} color={traits.hairColor} />

        {/* Hologram bubble when working */}
        <HologramBubble color={colorHex} opacity={anim.current.hologramOpacity} />
      </group>

      {/* Error triangle */}
      {state === "error" && (
        <group position={[0.4, headY + 0.3, 0]}>
          <mesh>
            <coneGeometry args={[0.12, 0.2, 3]} />
            <meshBasicMaterial color={0xef4444} wireframe />
          </mesh>
          <Text
            position={[0, -0.02, 0.08]}
            fontSize={0.12}
            color="#ef4444"
            anchorX="center"
            anchorY="middle"
            font={undefined}
          >
            !
          </Text>
        </group>
      )}

      {/* Red glitch flash overlay */}
      {anim.current.glitchFlash > 0 && (
        <mesh position={[0, 0.2, 0]}>
          <boxGeometry args={[0.5, 0.9, 0.3]} />
          <meshBasicMaterial color={0xef4444} transparent opacity={anim.current.glitchFlash} />
        </mesh>
      )}

      {/* Name label */}
      <Billboard position={[0, sitting ? -0.55 : -0.6, 0]} follow lockX={false} lockY={false} lockZ={false}>
        <Text
          fontSize={0.15}
          color={color}
          anchorX="center"
          anchorY="middle"
          font={undefined}
          outlineWidth={0.01}
          outlineColor="#000000"
        >
          {name}
        </Text>
      </Billboard>
    </group>
  )
}
