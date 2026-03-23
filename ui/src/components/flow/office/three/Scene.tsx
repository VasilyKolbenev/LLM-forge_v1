import { useRef, useMemo } from "react"
import { useThree, useFrame } from "@react-three/fiber"
import { OrbitControls, OrthographicCamera } from "@react-three/drei"
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing"
import * as THREE from "three"
import type { OfficeEnvironment } from "../../EnvironmentPicker"
import type { AgentData3D, WalkerData3D } from "./types"
import { THEMES_3D } from "./environments"
import { OfficeRoom } from "./OfficeRoom"
import { AgentDesk } from "./AgentDesk"
import { AgentChair } from "./AgentChair"
import { AgentCharacter } from "./AgentCharacter"
import { Walker } from "./Walker"
import { DataStream } from "./DataStream"
import { Particles } from "./Particles"
import { SelectionRing } from "./SelectionRing"
import { GroundGlow } from "./GroundGlow"
import { CHAIR_OFFSET, CHAR_OFFSET } from "./constants"

interface SceneProps {
  agents: AgentData3D[]
  walkers: WalkerData3D[]
  environment: OfficeEnvironment
  selectedNodeId: string | null
  onSelectNode: (id: string) => void
  onDoubleClickNode?: (id: string) => void
  walkingAgentIndices: Set<number>
}

function CameraRig({ agentCount }: { agentCount: number }) {
  const { camera } = useThree()
  const targetZoom = useMemo(() => {
    if (agentCount <= 4) return 55
    if (agentCount <= 8) return 42
    if (agentCount <= 16) return 32
    return 24
  }, [agentCount])

  useFrame(() => {
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = THREE.MathUtils.lerp(camera.zoom, targetZoom, 0.05)
      camera.updateProjectionMatrix()
    }
  })
  return null
}

function FogUpdater({ environment }: { environment: OfficeEnvironment }) {
  const { scene } = useThree()
  const theme = THEMES_3D[environment]
  const targetColor = useRef(new THREE.Color(theme.fogHex))

  useFrame(() => {
    targetColor.current.set(theme.fogHex)
    if (scene.fog instanceof THREE.Fog) {
      scene.fog.color.lerp(targetColor.current, 0.05)
    }
    scene.background = scene.fog?.color ?? null
  })
  return null
}

export function Scene({
  agents, walkers, environment, selectedNodeId,
  onSelectNode, onDoubleClickNode, walkingAgentIndices,
}: SceneProps) {
  const theme = THEMES_3D[environment]

  return (
    <>
      <OrthographicCamera
        makeDefault
        position={[15, 15, 15]}
        zoom={50}
        near={0.1}
        far={200}
      />
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minZoom={15}
        maxZoom={80}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI / 3}
        minAzimuthAngle={-Math.PI / 4}
        maxAzimuthAngle={Math.PI / 4}
        target={[0, 0, 0]}
      />

      <fog attach="fog" args={[theme.fogHex, 20, 60]} />
      <FogUpdater environment={environment} />
      <CameraRig agentCount={agents.length} />

      <ambientLight intensity={theme.ambientIntensity} color="#8888cc" />
      <directionalLight position={[10, 20, 10]} intensity={0.3} color="#aaaaff" />
      <directionalLight position={[-5, 10, -5]} intensity={0.1} color={theme.accentHex} />

      <OfficeRoom environment={environment} />
      <Particles environment={environment} />

      {agents.map((agent) => (
        <group key={agent.id} position={[agent.worldX, 0, agent.worldZ]}>
          <AgentDesk environment={environment} />
          <AgentChair
            position={[CHAIR_OFFSET.x, CHAIR_OFFSET.y, CHAIR_OFFSET.z]}
            accentHex={theme.accentHex}
          />
          <GroundGlow
            position={[CHAR_OFFSET.x, 0.02, CHAR_OFFSET.z]}
            color={agent.colorHex}
            state={agent.characterState}
          />
          {agent.selected && (
            <SelectionRing
              position={[CHAR_OFFSET.x, 0.03, CHAR_OFFSET.z]}
              color={agent.colorHex}
            />
          )}
          {!walkingAgentIndices.has(agent.index) && (
            <AgentCharacter
              position={[CHAR_OFFSET.x, 0, CHAR_OFFSET.z]}
              name={agent.name}
              color={agent.color}
              colorHex={agent.colorHex}
              state={agent.characterState}
              sitting={true}
              onClick={() => onSelectNode(agent.id)}
              onDoubleClick={onDoubleClickNode ? () => onDoubleClickNode(agent.id) : undefined}
            />
          )}
        </group>
      ))}

      {/* Data streams between consecutive agents */}
      {agents.length > 1 && agents.slice(0, -1).map((from, i) => {
        const to = agents[i + 1]
        if (!to) return null
        return (
          <DataStream
            key={`ds-${i}`}
            from={[from.worldX, 0.3, from.worldZ]}
            to={[to.worldX, 0.3, to.worldZ]}
            color={from.colorHex}
            active={from.characterState === "working" || from.characterState === "celebrating"}
          />
        )
      })}

      {/* Walkers */}
      {walkers.map((w) => (
        <Walker key={w.id} walker={w} />
      ))}

      <EffectComposer>
        <Bloom
          luminanceThreshold={0.4}
          luminanceSmoothing={0.9}
          intensity={1.8}
          mipmapBlur
        />
        <Vignette eskil={false} offset={0.1} darkness={0.8} />
      </EffectComposer>
    </>
  )
}
