import * as THREE from "three"

const vertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const fragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec3 uColor;
  uniform float uOpacity;

  varying vec2 vUv;

  void main() {
    // Horizontal scanlines
    float scanline = sin(vUv.y * 80.0 + uTime * 3.0) * 0.5 + 0.5;
    scanline = smoothstep(0.3, 0.7, scanline) * 0.3;

    // Vertical data bars
    float bars = step(0.5, sin(vUv.x * 20.0 + uTime * 0.5));
    float barHeight = smoothstep(0.0, vUv.y, sin(vUv.x * 7.0 + uTime) * 0.5 + 0.5);

    // Edge glow
    float edgeX = smoothstep(0.0, 0.1, vUv.x) * smoothstep(1.0, 0.9, vUv.x);
    float edgeY = smoothstep(0.0, 0.1, vUv.y) * smoothstep(1.0, 0.9, vUv.y);
    float edge = edgeX * edgeY;

    // Flicker
    float flicker = sin(uTime * 12.0) * 0.05 + 0.95;

    float alpha = (scanline + bars * barHeight * 0.2) * edge * uOpacity * flicker;
    vec3 color = uColor * (1.0 + scanline * 0.3);

    gl_FragColor = vec4(color, alpha);
  }
`

export function createHologramMaterial(color: THREE.Color, opacity = 0.6) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: color },
      uOpacity: { value: opacity },
    },
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
}
