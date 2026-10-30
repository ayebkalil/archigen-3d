import * as THREE from 'three';

export function wallFacadeFrame(wall) {
  const pts = wall.polygon_meters || [];
  if (pts.length < 2) return null;
  let edge = null, length = 0, area = 0;
  pts.forEach((a, i) => {
    const b = pts[(i + 1) % pts.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len > length) { edge = [a, b]; length = len; }
    area += a[0] * b[1] - b[0] * a[1];
  });
  if (length < 0.01) return null;
  const [a, b] = edge;
  const dx = (b[0] - a[0]) / length, dz = (b[1] - a[1]) / length;
  const outward = area >= 0 ? [dz, -dx] : [-dz, dx];
  return { origin: area >= 0 ? b : a, direction: [outward[1], -outward[0]], outward, length_m: length };
}

export function facadeOpeningPosition(opening, frame) {
  const s = opening.position_along_wall_m;
  return [frame.origin[0] + frame.direction[0] * s, frame.origin[1] + frame.direction[1] * s];
}

export function createFacadeGeometry(frame, bounds, height, centerX = 0, centerZ = 0) {
  const length = frame.length_m;
  // A dedicated cladding face avoids painting the facade onto wall end caps,
  // interior faces, or applying default ExtrudeGeometry's world-unit UVs.
  const geometry = new THREE.BufferGeometry();
  const positions = [], uvs = [];
  for (const [s, y, u, v] of [
    [0, 0, bounds.u_min, bounds.v_min], [length, 0, bounds.u_max, bounds.v_min],
    [length, height, bounds.u_max, bounds.v_max], [0, height, bounds.u_min, bounds.v_max],
  ]) {
    positions.push(frame.origin[0] + frame.direction[0] * s + frame.outward[0] * 0.003 - centerX,
      y, frame.origin[1] + frame.direction[1] * s + frame.outward[1] * 0.003 - centerZ);
    uvs.push(u, v);
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.computeVertexNormals();
  return geometry;
}

export function facadeBounds(length, height) {
  // Legacy URL-only callers use the bridge's 256px / 12px-margin letterbox.
  const scale = Math.min(232 / length, 232 / height);
  const w = Math.round(length * scale), h = Math.round(height * scale);
  const x = Math.floor((256 - w) / 2), y = Math.floor((256 - h) / 2);
  return { u_min: x / 256, u_max: (x + w) / 256, v_min: (256 - y - h) / 256, v_max: (256 - y) / 256 };
}
