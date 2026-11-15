// Appearance changes are confined to facade skins and existing pane/leaf meshes.
export function createAppearance(asset) {
  const cache = new Map();
  const texture = (name, color = false) => {
    if (!cache.has(name)) {
      const T = window.THREE, map = new T.TextureLoader().load(asset(name));
      if (color) map.colorSpace = T.SRGBColorSpace;
      map.wrapS = map.wrapT = T.ClampToEdgeWrapping;
      map.anisotropy = 8; cache.set(name, map);
    }
    return cache.get(name);
  };
  function apply(mode, neuralOpenings) {
    const viewer = window.__ARCHIGEN_VIEWER__, T = window.THREE;
    if (!viewer || !T) return;
    const {scene} = viewer; scene.updateMatrixWorld(true);
    const skins = scene.children.filter(o => o.userData.type === 'facade');
    for (const skin of skins) {
      if (!['reference', 'locked'].includes(mode) || skin.userData.v3Appearance === mode) continue;
      const id = skin.userData.wallId;
      skin.material.normalMap = texture(`${id}/normal.png`);
      skin.material.roughnessMap = texture(`${id}/arm.png`);
      skin.material.normalScale = new T.Vector2(.35, .35);
      skin.material.roughness = .85; skin.material.metalness = .02;
      skin.material.needsUpdate = true; skin.userData.v3Appearance = mode;
    }
    if (mode === 'locked' && neuralOpenings) {
      for (const group of scene.children.filter(o => o.userData.facadeOpening)) {
        const spec = group.userData.facadeOpening;
        const skin = skins.find(s => s.userData.wallId === spec.associated_wall);
        if (!skin) continue;
        const frame = skin.userData.frame, bounds = skin.userData.uvBounds;
        const p = skin.geometry.attributes.position;
        group.traverse(mesh => {
          if (!mesh.isMesh || mesh.userData.v3NeuralOpening) return;
          const glass = group.userData.type === 'window' && mesh.material.isMeshPhysicalMaterial;
          const door = group.userData.type === 'door' && mesh.material.color?.getHex() === 0x9b6b43;
          if (!glass && !door) return;
          const geometry = mesh.geometry.clone(), positions = geometry.attributes.position;
          const uv = new Float32Array(positions.count * 2);
          for (let i = 0; i < positions.count; i++) {
            const v = mesh.localToWorld(new T.Vector3().fromBufferAttribute(positions, i));
            const along = (v.x - p.getX(0) - skin.position.x) * frame.direction[0]
              + (v.z - p.getZ(0) - skin.position.z) * frame.direction[1];
            const height = v.y - skin.position.y;
            uv[i * 2] = bounds.u_min + (bounds.u_max - bounds.u_min) * along / frame.length_m;
            uv[i * 2 + 1] = bounds.v_min + (bounds.v_max - bounds.v_min) * height / bounds.wall_height_m;
          }
          geometry.setAttribute('uv', new T.BufferAttribute(uv, 2)); mesh.geometry = geometry;
          mesh.material = mesh.material.clone(); mesh.material.map = texture(`${spec.associated_wall}/guided.png`, true);
          mesh.material.color.set(0xffffff);
          if (glass) {mesh.material.transmission = .15; mesh.material.opacity = .95; mesh.material.roughness = .08;}
          mesh.material.needsUpdate = true; mesh.userData.v3NeuralOpening = true;
        });
      }
    }
  }
  return {apply, ready: () => [...cache.values()].every(t => t.image?.complete), dispose: () => {
    for (const map of cache.values()) map.dispose(); cache.clear();
  }};
}
