import { useRef, useMemo } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { OfficeEnvironment } from "../../EnvironmentPicker"
import { THEMES_3D } from "./environments"
import { ROOM_SIZE } from "./constants"

const PARTICLE_COUNT = 200

interface ParticlesProps {
  environment: OfficeEnvironment
}

export function Particles({ environment }: ParticlesProps) {
  const theme = THEMES_3D[environment]
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  // Generate random positions/speeds
  const particleData = useMemo(() => {
    const data: { x: number; y: number; z: number; speed: number; phase: number; size: number }[] = []
    const half = ROOM_SIZE / 2 - 2
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      data.push({
        x: (Math.random() - 0.5) * half * 2,
        y: Math.random() * 6 + 0.5,
        z: (Math.random() - 0.5) * half * 2,
        speed: 0.3 + Math.random() * 0.7,
        phase: Math.random() * Math.PI * 2,
        size: 0.015 + Math.random() * 0.025,
      })
    }
    return data
  }, [])

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const t = clock.elapsedTime
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const p = particleData[i]
      const opacity = 0.2 + Math.sin(t * p.speed + p.phase) * 0.3
      dummy.position.set(
        p.x + Math.sin(t * 0.3 + p.phase) * 0.5,
        p.y + Math.sin(t * p.speed + p.phase) * 0.3,
        p.z + Math.cos(t * 0.2 + p.phase) * 0.5,
      )
      dummy.scale.setScalar(p.size * (0.8 + opacity))
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    }
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, PARTICLE_COUNT]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color={theme.accentHex} transparent opacity={0.6} />
    </instancedMesh>
  )
}
