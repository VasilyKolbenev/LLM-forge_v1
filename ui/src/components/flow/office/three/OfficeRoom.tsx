import { useRef, useMemo } from "react"
import { useFrame } from "@react-three/fiber"
import { Text } from "@react-three/drei"
import * as THREE from "three"
import type { OfficeEnvironment } from "../../EnvironmentPicker"
import { THEMES_3D } from "./environments"
import { createGridFloorMaterial } from "./shaders/gridFloor"
import { createHologramMaterial } from "./shaders/hologram"
import { ROOM_SIZE, WALL_HEIGHT } from "./constants"

interface OfficeRoomProps {
  environment: OfficeEnvironment
}

function Floor({ environment }: { environment: OfficeEnvironment }) {
  const theme = THEMES_3D[environment]
  const matRef = useRef<THREE.ShaderMaterial>(null)

  const material = useMemo(() => {
    return createGridFloorMaterial(
      new THREE.Color(theme.accentHex),
      new THREE.Color(theme.floorTintHex),
    )
  }, [])

  // Update uniforms on env change and animate time
  useFrame((_, delta) => {
    const mat = matRef.current ?? material
    mat.uniforms.uTime.value += delta
    mat.uniforms.uAccentColor.value.lerp(new THREE.Color(theme.accentHex), 0.05)
    mat.uniforms.uFloorTint.value.lerp(new THREE.Color(theme.floorTintHex), 0.05)
  })

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
      <planeGeometry args={[ROOM_SIZE, ROOM_SIZE]} />
      <primitive object={material} ref={matRef} attach="material" />
    </mesh>
  )
}

function Walls({ environment }: { environment: OfficeEnvironment }) {
  const theme = THEMES_3D[environment]
  const half = ROOM_SIZE / 2

  return (
    <group>
      {/* Back wall */}
      <mesh position={[0, WALL_HEIGHT / 2, -half]}>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial
          color="#0a0c14"
          emissive={theme.accentHex}
          emissiveIntensity={0.02}
          transparent
          opacity={0.6}
        />
      </mesh>
      {/* Left wall */}
      <mesh position={[-half, WALL_HEIGHT / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[ROOM_SIZE, WALL_HEIGHT]} />
        <meshStandardMaterial
          color="#0a0c14"
          emissive={theme.accentHex}
          emissiveIntensity={0.02}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Wall edge neon lines */}
      <NeonEdgeLine
        points={[[-half, WALL_HEIGHT, -half], [half, WALL_HEIGHT, -half]]}
        color={theme.accentHex}
      />
      <NeonEdgeLine
        points={[[-half, WALL_HEIGHT, -half], [-half, WALL_HEIGHT, half]]}
        color={theme.accentHex}
      />
      <NeonEdgeLine
        points={[[-half, 0, -half], [-half, WALL_HEIGHT, -half]]}
        color={theme.accentHex}
      />
    </group>
  )
}

function NeonEdgeLine({ points, color }: { points: [number, number, number][]; color: number }) {
  const lineGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(points.flat())
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    return geo
  }, [points])

  return (
    <line geometry={lineGeo}>
      <lineBasicMaterial color={color} linewidth={2} transparent opacity={0.6} />
    </line>
  )
}

function WallHologramPanel({
  position, rotation, label, environment,
}: {
  position: [number, number, number]
  rotation?: [number, number, number]
  label: string
  environment: OfficeEnvironment
}) {
  const theme = THEMES_3D[environment]
  const matRef = useRef<THREE.ShaderMaterial>(null)

  const material = useMemo(() => {
    return createHologramMaterial(new THREE.Color(theme.accentHex), 0.4)
  }, [])

  useFrame((_, delta) => {
    const mat = matRef.current ?? material
    mat.uniforms.uTime.value += delta
    mat.uniforms.uColor.value.lerp(new THREE.Color(theme.accentHex), 0.05)
  })

  return (
    <group position={position} rotation={rotation}>
      <mesh>
        <planeGeometry args={[4, 2.5]} />
        <primitive object={material} ref={matRef} attach="material" />
      </mesh>
      <Text
        position={[0, 0.8, 0.01]}
        fontSize={0.25}
        color={theme.accent}
        anchorX="center"
        anchorY="middle"
        font={undefined}
      >
        {label}
      </Text>
    </group>
  )
}

export function OfficeRoom({ environment }: OfficeRoomProps) {
  const half = ROOM_SIZE / 2

  return (
    <group>
      <Floor environment={environment} />
      <Walls environment={environment} />

      {/* Wall hologram panels */}
      <WallHologramPanel
        position={[-6, WALL_HEIGHT * 0.5, -half + 0.1]}
        label="NEURAL LOAD"
        environment={environment}
      />
      <WallHologramPanel
        position={[6, WALL_HEIGHT * 0.5, -half + 0.1]}
        label="AGENTS ACTIVE"
        environment={environment}
      />
      <WallHologramPanel
        position={[-half + 0.1, WALL_HEIGHT * 0.5, -6]}
        rotation={[0, Math.PI / 2, 0]}
        label="DATA FLOW"
        environment={environment}
      />
      <WallHologramPanel
        position={[-half + 0.1, WALL_HEIGHT * 0.5, 6]}
        rotation={[0, Math.PI / 2, 0]}
        label="THROUGHPUT"
        environment={environment}
      />
    </group>
  )
}
