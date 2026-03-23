import * as THREE from "three"

// Shared materials — create once, reuse everywhere

export function neonMaterial(colorHex: number) {
  return new THREE.MeshBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.9,
  })
}

export function holographicSurface(colorHex: number, opacity = 0.08) {
  return new THREE.MeshStandardMaterial({
    color: colorHex,
    emissive: colorHex,
    emissiveIntensity: 0.3,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
}

export function skinMaterial(skinTone: string) {
  return new THREE.MeshStandardMaterial({
    color: skinTone,
    roughness: 0.8,
    metalness: 0,
  })
}

export function clothingMaterial(color: string) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.7,
    metalness: 0.1,
  })
}

export const darkMaterial = new THREE.MeshStandardMaterial({
  color: "#1a1a2e",
  roughness: 0.9,
  metalness: 0,
})
