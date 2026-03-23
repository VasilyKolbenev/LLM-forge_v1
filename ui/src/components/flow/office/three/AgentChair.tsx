interface AgentChairProps {
  position: [number, number, number]
  accentHex: number
}

export function AgentChair({ position, accentHex }: AgentChairProps) {
  return (
    <group position={position}>
      {/* Stem */}
      <mesh position={[0, 0.25, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 0.5, 6]} />
        <meshStandardMaterial color="#444466" metalness={0.5} roughness={0.5} />
      </mesh>
      {/* Base star */}
      {[0, 72, 144, 216, 288].map((angle) => {
        const rad = (angle * Math.PI) / 180
        return (
          <mesh key={angle} position={[Math.cos(rad) * 0.2, 0.05, Math.sin(rad) * 0.2]}>
            <boxGeometry args={[0.07, 0.05, 0.22]} />
            <meshStandardMaterial color="#3a3a4e" metalness={0.4} roughness={0.6} />
          </mesh>
        )
      })}
      {/* Wheels */}
      {[0, 72, 144, 216, 288].map((angle) => {
        const rad = (angle * Math.PI) / 180
        return (
          <mesh key={`w-${angle}`} position={[Math.cos(rad) * 0.3, 0.03, Math.sin(rad) * 0.3]}>
            <sphereGeometry args={[0.045, 6, 6]} />
            <meshStandardMaterial color="#333" metalness={0.3} roughness={0.7} />
          </mesh>
        )
      })}
      {/* Seat — more solid */}
      <mesh position={[0, 0.52, 0]}>
        <boxGeometry args={[0.55, 0.08, 0.55]} />
        <meshStandardMaterial
          color={accentHex}
          emissive={accentHex}
          emissiveIntensity={0.15}
          transparent
          opacity={0.7}
          metalness={0.2}
          roughness={0.6}
        />
      </mesh>
      {/* Seat edge glow */}
      <mesh position={[0, 0.52, 0]}>
        <boxGeometry args={[0.57, 0.02, 0.57]} />
        <meshBasicMaterial color={accentHex} transparent opacity={0.4} />
      </mesh>
      {/* Back — more solid */}
      <mesh position={[0, 0.8, -0.25]}>
        <boxGeometry args={[0.52, 0.55, 0.05]} />
        <meshStandardMaterial
          color={accentHex}
          emissive={accentHex}
          emissiveIntensity={0.1}
          transparent
          opacity={0.6}
          metalness={0.2}
          roughness={0.6}
        />
      </mesh>
      {/* Back edge glow */}
      <mesh position={[0, 0.8, -0.24]}>
        <boxGeometry args={[0.54, 0.57, 0.01]} />
        <meshBasicMaterial color={accentHex} transparent opacity={0.3} />
      </mesh>
    </group>
  )
}
