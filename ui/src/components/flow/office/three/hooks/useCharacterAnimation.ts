import { useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { CharacterState } from "../types"

export interface CharacterAnimState {
  bodyY: number
  bodyRotX: number
  leftArmRotX: number
  rightArmRotX: number
  leftLegRotX: number
  rightLegRotX: number
  eyeGlow: number
  visorOpacity: number
  hologramOpacity: number
  shakeX: number
  glitchFlash: number
  celebrateY: number
  particleBurst: boolean
}

const DEFAULT: CharacterAnimState = {
  bodyY: 0, bodyRotX: 0,
  leftArmRotX: 0, rightArmRotX: 0,
  leftLegRotX: 0, rightLegRotX: 0,
  eyeGlow: 0, visorOpacity: 0, hologramOpacity: 0,
  shakeX: 0, glitchFlash: 0,
  celebrateY: 0, particleBurst: false,
}

export function useCharacterAnimation(state: CharacterState) {
  const anim = useRef<CharacterAnimState>({ ...DEFAULT })
  const time = useRef(0)
  const prevState = useRef(state)

  useFrame((_, delta) => {
    time.current += delta
    const t = time.current
    const a = anim.current
    const lerpSpeed = 8 * delta

    // Reset burst flag each frame
    if (state !== "celebrating") a.particleBurst = false

    // Trigger burst on state transition to celebrating
    if (state === "celebrating" && prevState.current !== "celebrating") {
      a.particleBurst = true
    }
    prevState.current = state

    switch (state) {
      case "idle": {
        a.bodyY = THREE.MathUtils.lerp(a.bodyY, Math.sin(t * 1.2) * 0.03, lerpSpeed)
        a.bodyRotX = THREE.MathUtils.lerp(a.bodyRotX, 0, lerpSpeed)
        a.leftArmRotX = THREE.MathUtils.lerp(a.leftArmRotX, 0, lerpSpeed)
        a.rightArmRotX = THREE.MathUtils.lerp(a.rightArmRotX, 0, lerpSpeed)
        a.leftLegRotX = THREE.MathUtils.lerp(a.leftLegRotX, 0, lerpSpeed)
        a.rightLegRotX = THREE.MathUtils.lerp(a.rightLegRotX, 0, lerpSpeed)
        a.eyeGlow = THREE.MathUtils.lerp(a.eyeGlow, 0, lerpSpeed)
        a.visorOpacity = THREE.MathUtils.lerp(a.visorOpacity, 0, lerpSpeed)
        a.hologramOpacity = THREE.MathUtils.lerp(a.hologramOpacity, 0, lerpSpeed)
        a.shakeX = THREE.MathUtils.lerp(a.shakeX, 0, lerpSpeed)
        a.glitchFlash = THREE.MathUtils.lerp(a.glitchFlash, 0, lerpSpeed)
        a.celebrateY = THREE.MathUtils.lerp(a.celebrateY, 0, lerpSpeed)
        break
      }
      case "working": {
        a.bodyY = THREE.MathUtils.lerp(a.bodyY, Math.sin(t * 1.5) * 0.02, lerpSpeed)
        a.leftArmRotX = Math.sin(t * 6) * 0.3
        a.rightArmRotX = Math.sin(t * 5 + 0.5) * 0.25
        a.eyeGlow = 0.5 + Math.sin(t * 3) * 0.3
        a.visorOpacity = 0.4 + Math.sin(t * 2) * 0.15
        a.hologramOpacity = 0.5 + Math.sin(t * 1.5) * 0.2
        a.shakeX = 0
        a.glitchFlash = 0
        a.celebrateY = THREE.MathUtils.lerp(a.celebrateY, 0, lerpSpeed)
        break
      }
      case "walking": {
        a.bodyY = Math.sin(t * 8) * 0.05
        a.bodyRotX = Math.sin(t * 4) * 0.05
        a.leftArmRotX = Math.sin(t * 8) * 0.5
        a.rightArmRotX = -Math.sin(t * 8) * 0.5
        a.leftLegRotX = -Math.sin(t * 8) * 0.6
        a.rightLegRotX = Math.sin(t * 8) * 0.6
        a.eyeGlow = 0.2
        a.visorOpacity = 0
        a.hologramOpacity = 0
        break
      }
      case "delivering": {
        a.bodyY = Math.sin(t * 2) * 0.02
        a.rightArmRotX = -0.8 // holding cube
        a.leftArmRotX = 0
        a.eyeGlow = 0.3
        break
      }
      case "celebrating": {
        const bounce = Math.abs(Math.sin(t * 6)) * 0.5
        a.celebrateY = bounce
        a.bodyY = bounce
        a.leftArmRotX = -2.5 + Math.sin(t * 4) * 0.3
        a.rightArmRotX = -2.5 + Math.sin(t * 4 + 1) * 0.3
        a.eyeGlow = 0.8
        a.visorOpacity = 0
        a.hologramOpacity = 0
        a.shakeX = 0
        break
      }
      case "error": {
        a.shakeX = Math.sin(t * 30) * 0.05
        a.glitchFlash = Math.sin(t * 15) > 0.5 ? 0.3 : 0
        a.bodyY = THREE.MathUtils.lerp(a.bodyY, 0, lerpSpeed)
        a.eyeGlow = 0.1
        a.visorOpacity = 0
        a.hologramOpacity = 0
        a.leftArmRotX = THREE.MathUtils.lerp(a.leftArmRotX, 0, lerpSpeed)
        a.rightArmRotX = THREE.MathUtils.lerp(a.rightArmRotX, 0, lerpSpeed)
        a.celebrateY = THREE.MathUtils.lerp(a.celebrateY, 0, lerpSpeed)
        break
      }
    }
  })

  return anim
}
