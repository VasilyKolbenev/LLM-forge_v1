import { useRef } from "react"
import { useFrame } from "@react-three/fiber"
import { Text, Billboard } from "@react-three/drei"
import * as THREE from "three"
import type { WalkerData3D } from "./types"
import { AgentCharacter } from "./AgentCharacter"

interface WalkerProps {
  walker: WalkerData3D
}

export function Walker({ walker }: WalkerProps) {
  const trailRef = useRef<THREE.InstancedMesh>(null)
  const dummy = new THREE.Object3D()

  // Neon trail
  useFrame(({ clock }) => {
    if (!trailRef.current) return
    const t = clock.elapsedTime
    for (let i = 0; i < 6; i++) {
      const offset = i * 0.15
      dummy.position.set(
        walker.x - Math.sin(t + offset) * 0.1,
        0.05,
        walker.z - offset * 0.3,
      )
      dummy.scale.setScalar(0.03 * (1 - i * 0.15))
      dummy.updateMatrix()
      trailRef.current.setMatrixAt(i, dummy.matrix)
    }
    trailRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <group>
      <AgentCharacter
        position={[walker.x, walker.y, walker.z]}
        name={walker.name}
        color={walker.color}
        colorHex={walker.colorHex}
        state={walker.state}
        sitting={false}
      />

      {/* Neon trail dots */}
      <instancedMesh ref={trailRef} args={[undefined, undefined, 6]}>
        <sphereGeometry args={[1, 6, 6]} />
        <meshBasicMaterial color={walker.colorHex} transparent opacity={0.4} />
      </instancedMesh>

      {/* "INCOMING DATA" label when delivering */}
      {walker.phase === "delivering" && (
        <Billboard position={[walker.x, walker.y + 1.2, walker.z]}>
          <Text
            fontSize={0.12}
            color={walker.color}
            anchorX="center"
            anchorY="middle"
            font={undefined}
            outlineWidth={0.005}
            outlineColor="#000"
          >
            INCOMING DATA
          </Text>
        </Billboard>
      )}
    </group>
  )
}
