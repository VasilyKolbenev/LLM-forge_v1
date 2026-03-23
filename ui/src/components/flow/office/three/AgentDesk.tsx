import { useRef, useMemo } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { OfficeEnvironment } from "../../EnvironmentPicker"
import { THEMES_3D } from "./environments"

interface AgentDeskProps {
  environment: OfficeEnvironment
}

export function AgentDesk({ environment }: AgentDeskProps) {
  const theme = THEMES_3D[environment]
  const crystalRef = useRef<THREE.Mesh>(null)
  const monitorMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const barRefs = useRef<THREE.Mesh[]>([])
  const timeRef = useRef(0)

  const edgesGeo = useMemo(() => {
    const box = new THREE.BoxGeometry(2.5, 0.08, 1.5)
    return new THREE.EdgesGeometry(box)
  }, [])

  useFrame((_, delta) => {
    timeRef.current += delta
    const t = timeRef.current

    // Rotate data crystal
    if (crystalRef.current) {
      crystalRef.current.rotation.y += delta * 0.8
      crystalRef.current.position.y = 1.0 + Math.sin(t * 1.5) * 0.1
    }

    // Pulse monitor
    if (monitorMatRef.current) {
      monitorMatRef.current.opacity = 0.5 + Math.sin(t * 2) * 0.15
    }

    // Animate bar chart
    barRefs.current.forEach((bar, i) => {
      if (bar) {
        const h = 0.15 + Math.abs(Math.sin(t * 1.2 + i * 0.8)) * 0.3
        bar.scale.y = h / 0.15
        bar.position.y = 0.86 + h / 2
      }
    })
  })

  return (
    <group>
      {/* Desk surface — semi-transparent with glow */}
      <mesh position={[0, 0.8, 0]}>
        <boxGeometry args={[2.5, 0.08, 1.5]} />
        <meshStandardMaterial
          color={theme.accentHex}
          emissive={theme.accentHex}
          emissiveIntensity={0.2}
          transparent
          opacity={0.15}
          metalness={0.3}
          roughness={0.4}
        />
      </mesh>
      {/* Neon edges — brighter */}
      <lineSegments geometry={edgesGeo} position={[0, 0.8, 0]}>
        <lineBasicMaterial color={theme.accentHex} transparent opacity={0.9} />
      </lineSegments>

      {/* Desk legs — thicker, more visible */}
      {[[-1, -0.5], [1, -0.5], [-1, 0.5], [1, 0.5]].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.4, z]}>
          <cylinderGeometry args={[0.04, 0.04, 0.8, 6]} />
          <meshBasicMaterial color={theme.accentHex} transparent opacity={0.5} />
        </mesh>
      ))}

      {/* Floating monitor */}
      <group position={[0, 1.6, -0.4]}>
        {/* Monitor frame */}
        <mesh>
          <boxGeometry args={[1.2, 0.8, 0.04]} />
          <meshStandardMaterial
            color="#0a0a14"
            emissive={theme.accentHex}
            emissiveIntensity={0.05}
          />
        </mesh>
        {/* Screen content */}
        <mesh position={[0, 0, 0.025]}>
          <planeGeometry args={[1.1, 0.7]} />
          <meshBasicMaterial
            ref={monitorMatRef}
            color={theme.accentHex}
            transparent
            opacity={0.3}
          />
        </mesh>
        {/* Scanline overlay */}
        <mesh position={[0, 0, 0.03]}>
          <planeGeometry args={[1.1, 0.7]} />
          <meshBasicMaterial color={theme.accentHex} transparent opacity={0.02} />
        </mesh>
        {/* Float connector to desk */}
        <mesh position={[0, -0.55, 0.2]} rotation={[0.2, 0, 0]}>
          <cylinderGeometry args={[0.01, 0.01, 0.3, 4]} />
          <meshBasicMaterial color={theme.accentHex} transparent opacity={0.2} />
        </mesh>
      </group>

      {/* Data crystal — rotating glowing cube */}
      <mesh ref={crystalRef} position={[0.9, 1.0, 0.3]}>
        <boxGeometry args={[0.2, 0.2, 0.2]} />
        <meshBasicMaterial color={theme.accentHex} transparent opacity={0.5} />
      </mesh>
      {/* Crystal glow */}
      <pointLight position={[0.9, 1.1, 0.3]} color={theme.accentHex} intensity={0.3} distance={2} />

      {/* Mini dashboard — bar chart */}
      <group position={[-0.9, 0.85, 0.3]}>
        {/* Background */}
        <mesh>
          <planeGeometry args={[0.6, 0.4]} />
          <meshBasicMaterial color={theme.accentHex} transparent opacity={0.04} side={THREE.DoubleSide} />
        </mesh>
        {/* Bars */}
        {[0, 1, 2, 3].map((i) => (
          <mesh
            key={i}
            ref={(el) => { if (el) barRefs.current[i] = el }}
            position={[-0.18 + i * 0.12, 0, 0.01]}
          >
            <boxGeometry args={[0.08, 0.15, 0.01]} />
            <meshBasicMaterial color={theme.accentHex} transparent opacity={0.6} />
          </mesh>
        ))}
      </group>

      {/* Holographic keyboard — dots */}
      <group position={[0, 0.86, 0.5]}>
        {[0, 1, 2].map((row) =>
          [-0.4, -0.25, -0.1, 0.05, 0.2, 0.35].map((x, col) => (
            <mesh key={`k-${row}-${col}`} position={[x, 0, row * 0.1]}>
              <sphereGeometry args={[0.02, 6, 6]} />
              <meshBasicMaterial color={theme.accentHex} transparent opacity={0.4} />
            </mesh>
          ))
        )}
      </group>
    </group>
  )
}
