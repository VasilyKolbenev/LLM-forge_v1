import { useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { CharacterState } from "./types"

interface GroundGlowProps {
  position: [number, number, number]
  color: number
  state: CharacterState
}

export function GroundGlow({ position, color, state }: GroundGlowProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshBasicMaterial>(null)

  const targetOpacity =
    state === "working" ? 0.25
    : state === "celebrating" ? 0.35
    : state === "error" ? 0.15
    : 0.06

  useFrame(({ clock }) => {
    if (matRef.current) {
      const pulse = state === "working"
        ? Math.sin(clock.elapsedTime * 2) * 0.08
        : state === "celebrating"
        ? Math.sin(clock.elapsedTime * 5) * 0.15
        : 0
      matRef.current.opacity = THREE.MathUtils.lerp(
        matRef.current.opacity,
        targetOpacity + pulse,
        0.05,
      )
    }
    if (meshRef.current && state === "celebrating") {
      const s = 1 + Math.sin(clock.elapsedTime * 6) * 0.3
      meshRef.current.scale.set(s, 1, s)
    }
  })

  return (
    <mesh ref={meshRef} position={position} rotation={[-Math.PI / 2, 0, 0]}>
      <circleGeometry args={[0.6, 24]} />
      <meshBasicMaterial
        ref={matRef}
        color={state === "error" ? 0xef4444 : color}
        transparent
        opacity={0.06}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}
