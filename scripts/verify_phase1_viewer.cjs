/** Render the actual React viewer with deterministic architectural fixtures.
 * Requires native API :8000 and Vite :3000. Uses the real facade generator;
 * intercepts segmentation only so roof shape checks do not depend on ML recall.
 * Run: node scripts/verify_phase1_viewer.cjs
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const puppeteer = require('../frontend/node_modules/puppeteer-core');
  const viewerUrl = process.env.ARCHIGEN_VIEWER_URL || 'http://127.0.0.1:3000';
  const out = path.resolve(__dirname, '../outputs/phase1');
  fs.mkdirSync(out, { recursive: true });
  const shapes = {
    L: [[0, 0], [8, 0], [8, 3], [3, 3], [3, 8], [0, 8]],
    T: [[0, 0], [8, 0], [8, 3], [5.5, 3], [5.5, 8], [2.5, 8], [2.5, 3], [0, 3]],
    U: [[0, 0], [8, 0], [8, 8], [5.5, 8], [5.5, 3], [2.5, 3], [2.5, 8], [0, 8]],
    detached: [[0, 0], [8, 0], [8, 4], [0, 4]],
  };
  const frontWall = { id: 'wall_0', height_m: 2.8, is_front: true, is_exterior: true,
    polygon_meters: [[0, 0], [8, 0], [8, 0.18], [0, 0.18]],
    openings: [
      { id: 'door_0', type: 'door', position_meters: [5.8, 0], width_m: 1.1, height_m: 2.1, associated_wall: 'wall_0' },
      { id: 'window_0', type: 'window', position_meters: [1.6, 0], width_m: 1.4, height_m: 1.4, associated_wall: 'wall_0' },
      { id: 'window_1', type: 'window', position_meters: [3.7, 0], width_m: 1.2, height_m: 1.3, associated_wall: 'wall_0' },
    ],
  };
  const res = await fetch('http://127.0.0.1:8000/facade/generate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ wall: frontWall, num_floors: 2, style: 'Modern Minimalist' }),
  });
  assert.ok(res.ok, await res.clone().text());
  const facade = await res.json();
  assert.ok(facade.facade_frame && facade.openings.length === 5);
  fs.writeFileSync(path.join(out, 'facade-metadata.json'), JSON.stringify({ ...facade, texture_base64: undefined, sketch_base64: undefined }, null, 2));
  const executablePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(p => fs.existsSync(p));
  assert.ok(executablePath, 'Chrome or Edge is installed');
  const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl'] });
  const errors = [], reports = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1100, deviceScaleFactor: 1 });
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', msg => { if (msg.type() === 'error' && /THREE|WebGL|shader/i.test(msg.text())) errors.push(msg.text()); });
    let shapeName = 'L', diagnostic = false;
    await page.setRequestInterception(true);
    page.on('request', request => {
      if (!request.url().includes('/predict/floorplan')) return request.continue();
      const polygon = shapes[shapeName];
      const wall = { ...frontWall, facade_texture: { ...facade, texture_base64: diagnostic ? facade.sketch_base64 : facade.texture_base64 } };
      const walls = [wall, ...polygon.slice(1).map((a, i) => {
        const b = polygon[(i + 2) % polygon.length];
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const nx = -(b[1] - a[1]) / len * 0.18, nz = (b[0] - a[0]) / len * 0.18;
        return { id: `wall_${i + 1}`, height_m: 2.8, is_exterior: true,
          polygon_meters: [a, b, [b[0] + nx, b[1] + nz], [a[0] + nx, a[1] + nz]], openings: [] };
      })];
      const envelope = [polygon];
      if (shapeName === 'detached') {
        envelope.push(polygon.map(([x, z]) => [x + 11, z]));
        walls.push(...walls.map(wall => ({ ...wall, id: `${wall.id}_detached`, is_front: false, openings: [],
          facade_texture: undefined, polygon_meters: wall.polygon_meters.map(([x, z]) => [x + 11, z]) })));
      }
      return request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify({
        total_area_m2: 40, num_rooms: 1, walls, rooms: [{ id: 'room_0', room_type: 'Living Room', surface_m2: 40, polygon_points: polygon }],
        doors: frontWall.openings.filter(op => op.type === 'door'), windows: frontWall.openings.filter(op => op.type === 'window'), envelope,
      }) });
    });
    const click = async title => {
      await page.waitForSelector(`button[title="${title}"]`);
      await page.click(`button[title="${title}"]`);
    };
    const ready = async () => {
      await page.goto(`${viewerUrl}/?phase1=${shapeName}&diagnostic=${diagnostic}#studio`, { waitUntil: 'networkidle0' });
      await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__?.scene.children.some(o => o.userData.type === 'facade'));
      await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === '3D PERSPECTIVE').click());
      await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__?.renderer.domElement.isConnected && window.__ARCHIGEN_VIEWER__.scene.children.some(o => o.userData.type === 'facade'));
      await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === '2 STOREYS').click());
      await page.waitForFunction(() => {
        const skin = window.__ARCHIGEN_VIEWER__.scene.children.find(o => o.userData.type === 'facade');
        return skin && Math.abs(skin.geometry.attributes.position.getY(2) - 5.6) < 0.01;
      });
      await click('Toggle 3D Building Roof Enclosure (Activate / Deactivate)');
      await click('Classic Sims Pitched Hipped Gable Roof');
      await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('roofGroup').children.some(o => o.userData.style === 'pitched'));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    for (shapeName of Object.keys(shapes)) {
      await ready();
      const report = await page.evaluate(() => {
        const { scene, renderer, camera, controls } = window.__ARCHIGEN_VIEWER__;
        const roofGroup = scene.getObjectByName('roofGroup');
        const roofs = roofGroup.children.map(o => ({ style: o.userData.style, fallback: o.userData.fallback, triangles: o.geometry.attributes.position.count / 3 }));
        const skin = scene.children.find(o => o.userData.type === 'facade');
        const openings = scene.children.filter(o => o.userData.facadeOpening);
        const frame = skin.userData.frame, p = skin.geometry.attributes.position, uv = skin.geometry.attributes.uv;
        const wallHeight = Math.max(...Array.from({ length: p.count }, (_, i) => p.getY(i)));
        let maxError = 0;
        for (const o of openings) {
          const op = o.userData.facadeOpening;
          // Measure the actual Three.js frame geometry, independently of the
          // backend metadata, against the semantic image rectangle.
          const along = (o.position.x - p.getX(0)) * frame.direction[0] + (o.position.z - p.getZ(0)) * frame.direction[1];
          const bottom = o.children[0], top = op.type === 'window' ? o.children[1] : bottom;
          const low = o.position.y + bottom.position.y - bottom.geometry.parameters.height / 2;
          const high = o.position.y + top.position.y + top.geometry.parameters.height / 2;
          const width = op.type === 'window' ? bottom.geometry.parameters.width
            : (o.children[1].position.x - bottom.position.x + bottom.geometry.parameters.width);
          const u0 = uv.getX(0) + (along - width / 2) / frame.length_m * (uv.getX(1) - uv.getX(0));
          const u1 = uv.getX(0) + (along + width / 2) / frame.length_m * (uv.getX(1) - uv.getX(0));
          const v0 = uv.getY(0) + low / wallHeight * (uv.getY(2) - uv.getY(0));
          const v1 = uv.getY(0) + high / wallHeight * (uv.getY(2) - uv.getY(0));
          maxError = Math.max(maxError, ...[u0, v0, u1, v1].map((v, i) => Math.abs(v - op.uv_rect[i])));
        }
        const center = new window.THREE.Box3().setFromObject(roofGroup).getCenter(new window.THREE.Vector3());
        camera.position.set(center.x - 13, 17, center.z - 17);
        camera.up.set(0, 1, 0); controls.target.copy(center); controls.update();
        renderer.render(scene, camera);
        return { roofs, openings: openings.length, alignmentErrorPixels: maxError * 256, drawCalls: renderer.info.render.calls };
      });
      assert.equal(report.roofs.length, shapeName === 'detached' ? 2 : 1);
      assert.ok(report.roofs.every(r => r.fallback === null && r.style === 'pitched'));
      assert.equal(report.openings, 5);
      assert.ok(report.alignmentErrorPixels < 0.05, `UV alignment error ${report.alignmentErrorPixels}px`);
      reports.push({ shape: shapeName, ...report });
      await page.screenshot({ path: path.join(out, `${shapeName}-pitched.png`) });
      await page.evaluate(() => {
        const { camera, controls, scene, renderer } = window.__ARCHIGEN_VIEWER__;
        const center = new window.THREE.Box3().setFromObject(scene.getObjectByName('roofGroup')).getCenter(new window.THREE.Vector3());
        camera.position.set(center.x, 26, center.z); camera.up.set(0, 0, -1);
        controls.target.set(center.x, 0, center.z); controls.update(); renderer.render(scene, camera);
      });
      await page.screenshot({ path: path.join(out, `${shapeName}-top.png`) });
      if (shapeName === 'L') {
        await click('3D Perspective View');
        await click('Modern Architectural Flat Slab Roof with Overhang Eaves & Fascia Band');
        await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('roofGroup').children.every(o => o.userData.style === 'slab'));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
        await page.screenshot({ path: path.join(out, 'L-slab.png') });
        await click('Focus camera directly on Front Facade wall');
        await page.screenshot({ path: path.join(out, 'facade-neural.png') });
      }
    }
    shapeName = 'L'; diagnostic = true;
    await ready();
    await click('Focus camera directly on Front Facade wall');
    await page.screenshot({ path: path.join(out, 'facade-alignment.png') });
    await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'SPLIT (2D + 3D)').click());
    await page.waitForFunction(() => {
      const viewer = window.__ARCHIGEN_VIEWER__;
      if (!viewer?.renderer.domElement.isConnected) return false;
      const rect = viewer.renderer.domElement.getBoundingClientRect();
      const size = viewer.renderer.getSize(new window.THREE.Vector2());
      return rect.width > 0 && Math.abs(size.x - rect.width) < 1 && Math.abs(viewer.camera.aspect - rect.width / rect.height) < 0.01;
    });
    const fits = await page.evaluate(() => {
      const { scene, camera } = window.__ARCHIGEN_VIEWER__;
      const bounds = new window.THREE.Box3();
      scene.children.filter(o => o.userData.isBuildingMesh).forEach(o => bounds.union(new window.THREE.Box3().setFromObject(o)));
      return [bounds.min.x, bounds.max.x].every(x => [bounds.min.y, bounds.max.y].every(y => [bounds.min.z, bounds.max.z].every(z => {
        const p = new window.THREE.Vector3(x, y, z).project(camera);
        return Math.abs(p.x) < 1 && Math.abs(p.y) < 1;
      })));
    });
    assert.ok(fits, 'default split camera fits the whole building');
    await page.screenshot({ path: path.join(out, 'L-split.png') });
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, 'verification.json'), JSON.stringify({ device: facade.device, errors, reports }, null, 2));
    console.log(JSON.stringify({ output: out, device: facade.device, errors, reports }, null, 2));
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
