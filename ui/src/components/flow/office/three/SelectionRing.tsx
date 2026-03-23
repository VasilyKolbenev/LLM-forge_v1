import { useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"

interface SelectionRingProps {
  position: [number, number, number]
  color: number
}

export function SelectionRing({ position, color }: SelectionRingProps) {
  const ringRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (ringRef.current) {
      ringRef.current.rotation.y = clock.elapsedTime * 1.5
      const scale = 1 + Math.sin(clock.elapsedTime * 3) * 0.1
      ringRef.current.scale.set(scale, 1, scale)
    }
  })

  return (
    <group position={position}>
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.5, 0.03, 8, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} />
      </mesh>
      {/* Outer glow ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.55, 0.06, 8, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.15} />
      </mesh>
      <pointLight color={color} intensity={0.8} distance={2} position={[0, 0.1, 0]} />
    </group>
  )
}
