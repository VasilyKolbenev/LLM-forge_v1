import * as THREE from "three"

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`

const fragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec3 uAccentColor;
  uniform vec3 uFloorTint;
  uniform float uGridSize;

  varying vec2 vUv;
  varying vec3 vWorldPos;

  float gridLine(float coord, float width) {
    float d = abs(fract(coord - 0.5) - 0.5);
    return 1.0 - smoothstep(0.0, width, d);
  }

  void main() {
    // Grid lines
    float gx = gridLine(vWorldPos.x / uGridSize, 0.02);
    float gz = gridLine(vWorldPos.z / uGridSize, 0.02);
    float grid = max(gx, gz);

    // Sub-grid (finer lines)
    float sgx = gridLine(vWorldPos.x / (uGridSize * 0.25), 0.04);
    float sgz = gridLine(vWorldPos.z / (uGridSize * 0.25), 0.04);
    float subGrid = max(sgx, sgz) * 0.15;

    // Distance fade from center
    float dist = length(vWorldPos.xz) / 20.0;
    float fade = 1.0 - smoothstep(0.0, 1.0, dist);

    // Radial pulse from center
    float pulse = sin(dist * 8.0 - uTime * 1.5) * 0.5 + 0.5;
    float radialGlow = (1.0 - smoothstep(0.0, 0.5, dist)) * pulse * 0.15;

    // Scanning line effect
    float scanLine = smoothstep(0.0, 0.3, 1.0 - abs(fract(vWorldPos.z * 0.1 - uTime * 0.3) - 0.5) * 2.0) * 0.08;

    // Combine
    vec3 gridColor = uAccentColor * (grid * 0.4 + subGrid) * fade;
    vec3 glowColor = uAccentColor * (radialGlow + scanLine);
    vec3 baseColor = uFloorTint;

    vec3 finalColor = baseColor + gridColor + glowColor;
    float alpha = 1.0;

    gl_FragColor = vec4(finalColor, alpha);
  }
`

export function createGridFloorMaterial(accentColor: THREE.Color, floorTint: THREE.Color) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uAccentColor: { value: accentColor },
      uFloorTint: { value: floorTint },
      uGridSize: { value: 2.5 },
    },
    side: THREE.DoubleSide,
  })
}
