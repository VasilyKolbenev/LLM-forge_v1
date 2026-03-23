import { useRef, useMemo } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"

interface DataStreamProps {
  from: [number, number, number]
  to: [number, number, number]
  color: number
  active: boolean
}

const DOT_COUNT = 8

export function DataStream({ from, to, color, active }: DataStreamProps) {
  const dotsRef = useRef<THREE.InstancedMesh>(null)
  const lineRef = useRef<THREE.Line>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const lineGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const pts = new Float32Array([...from, ...to])
    geo.setAttribute("position", new THREE.BufferAttribute(pts, 3))
    return geo
  }, [from, to])

  useFrame(({ clock }) => {
    if (!dotsRef.current) return
    const t = clock.elapsedTime

    for (let i = 0; i < DOT_COUNT; i++) {
      const progress = ((i / DOT_COUNT) + t * (active ? 0.5 : 0.15)) % 1
      const x = from[0] + (to[0] - from[0]) * progress
      const y = from[1] + (to[1] - from[1]) * progress + Math.sin(progress * Math.PI) * 0.15
      const z = from[2] + (to[2] - from[2]) * progress

      dummy.position.set(x, y, z)
      const scale = active ? 0.04 : 0.02
      dummy.scale.setScalar(scale)
      dummy.updateMatrix()
      dotsRef.current.setMatrixAt(i, dummy.matrix)
    }
    dotsRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <group>
      {/* Base line */}
      <line ref={lineRef} geometry={lineGeo}>
        <lineBasicMaterial
          color={color}
          transparent
          opacity={active ? 0.2 : 0.05}
        />
      </line>

      {/* Flowing dots */}
      <instancedMesh ref={dotsRef} args={[undefined, undefined, DOT_COUNT]}>
        <sphereGeometry args={[1, 6, 6]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 0.7 : 0.2} />
      </instancedMesh>
    </group>
  )
}
