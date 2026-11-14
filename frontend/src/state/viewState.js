export function cameraPose(camera, controls) {
  return { position: camera.position.toArray(), target: controls.target.toArray(), up: camera.up.toArray(), zoom: camera.zoom };
}

export function restoreCameraPose(camera, controls, pose) {
  if (!pose || !['position', 'target', 'up'].every(key => Array.isArray(pose[key]) && pose[key].length === 3 && pose[key].every(Number.isFinite))) return false;
  camera.position.fromArray(pose.position);
  camera.up.fromArray(pose.up);
  camera.zoom = Number.isFinite(pose.zoom) && pose.zoom > 0 ? pose.zoom : 1;
  camera.updateProjectionMatrix();
  controls.target.fromArray(pose.target);
  controls.update();
  return true;
}

export function disposeScene(scene) {
  const disposed = new Set();
  const dispose = value => { if (value && !disposed.has(value)) { disposed.add(value); value.dispose?.(); } };
  scene?.traverse(object => {
    dispose(object.geometry);
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) dispose(value);
      dispose(material);
    }
  });
  dispose(scene?.environment);
}
