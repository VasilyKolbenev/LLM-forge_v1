// Layout constants for 3D office grid

export const DESKS_PER_ROW = 4
export const GRID_SPACING_X = 5    // distance between desks on X axis
export const GRID_SPACING_Z = 4.5  // distance between desks on Z axis

// Convert grid index to 3D world position (XZ plane)
export function gridTo3D(index: number, total: number) {
  const col = index % DESKS_PER_ROW
  const row = Math.floor(index / DESKS_PER_ROW)
  const cols = Math.min(total, DESKS_PER_ROW)
  // Center the grid
  const offsetX = ((cols - 1) * GRID_SPACING_X) / 2
  const offsetZ = ((Math.ceil(total / DESKS_PER_ROW) - 1) * GRID_SPACING_Z) / 2
  return {
    x: col * GRID_SPACING_X - offsetX,
    y: 0,
    z: row * GRID_SPACING_Z - offsetZ,
  }
}

// Character offset relative to desk (sitting position)
export const CHAR_OFFSET = { x: 0, y: 0, z: 1.2 }
// Chair offset relative to desk
export const CHAIR_OFFSET = { x: 0, y: 0, z: 0.8 }

// Camera
export const CAMERA_DISTANCE = 30
export const CAMERA_ZOOM = 50

// Room dimensions
export const ROOM_SIZE = 40
export const WALL_HEIGHT = 8
