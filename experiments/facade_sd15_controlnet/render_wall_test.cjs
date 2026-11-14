/** Inspect and screenshot the actual production Viewer3D with the local candidate. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const project = path.resolve(process.argv[2]);
  const output = path.join(project, 'outputs/facade_house_test');
  const puppeteer = require(path.join(project, 'frontend/node_modules/puppeteer-core'));
  const executablePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(p => fs.existsSync(p));
  assert.ok(executablePath, 'Browser executable must be installed');
  const browser = await puppeteer.launch({executablePath, headless: true,
    args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl']});
  const errors = [], results = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({width: 1600, height: 1050, deviceScaleFactor: 1});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && /THREE|WebGL|shader/i.test(message.text())) errors.push(message.text());
    });
    await page.goto('http://127.0.0.1:3001/tmp/facade-wall-test/', {waitUntil: 'domcontentloaded'});
    for (const mode of ['baseline', 'candidate', 'guide']) {
      await page.click(`[data-mode="${mode}"]`);
      await page.waitForFunction(mode => {
        const viewer = window.__ARCHIGEN_VIEWER__;
        return window.__FACADE_WALL_TEST__?.mode === mode && viewer?.renderer.domElement.isConnected &&
          viewer.scene.children.some(object => object.userData.type === 'facade' && object.material.map?.image?.complete);
      }, {}, mode);
      await new Promise(resolve => setTimeout(resolve, 1800));
      const result = await page.evaluate(() => {
        const {scene} = window.__ARCHIGEN_VIEWER__;
        const {probe, mode} = window.__FACADE_WALL_TEST__;
        const skin = scene.children.find(object => object.userData.type === 'facade');
        const p = skin.geometry.attributes.position, uv = skin.geometry.attributes.uv;
        const frame = probe.facade_frame, bounds = skin.userData.uvBounds;
        let uvError = 0;
        for (let i = 0; i < p.count; i++) {
          const along = (p.getX(i) - p.getX(0)) * frame.direction[0] + (p.getZ(i) - p.getZ(0)) * frame.direction[1];
          const expectedU = bounds.u_min + (bounds.u_max - bounds.u_min) * along / frame.length_m;
          const expectedV = bounds.v_min + (bounds.v_max - bounds.v_min) * p.getY(i) / probe.wall_height_m;
          uvError = Math.max(uvError, Math.abs(expectedU - uv.getX(i)), Math.abs(expectedV - uv.getY(i)));
        }
        const openings = scene.children.filter(object => object.userData.facadeOpening).map(object => {
          const opening = object.userData.facadeOpening;
          const along = (object.position.x - p.getX(0)) * frame.direction[0] + (object.position.z - p.getZ(0)) * frame.direction[1];
          return {id: opening.id, type: opening.type, positionAlong: along,
            expectedAlong: opening.position_along_wall_m,
            centerErrorM: Math.abs(along - opening.position_along_wall_m)};
        });
        return {mode, uvError, openingCount: openings.length, openings,
          textureSize: [skin.material.map.image.width, skin.material.map.image.height]};
      });
      assert.equal(result.openingCount, 7, 'All seven canonical openings must remain in the actual viewer');
      assert.ok(result.uvError < 1e-5, 'Facade UV coordinates must remain normalized and aligned');
      assert.ok(result.openings.every(opening => opening.centerErrorM < 1e-4), '3D frames must stay at their physical positions');
      if (results.length) {
        result.openings.forEach((opening, index) => assert.ok(
          Math.abs(opening.positionAlong - results[0].openings[index].positionAlong) < 1e-7,
          'Changing the texture must not move the 3D openings'));
      }
      results.push(result);
      await page.screenshot({path: path.join(output, `viewer-${mode}-front.png`)});
    }
    await page.click('[data-mode="candidate"]');
    await new Promise(resolve => setTimeout(resolve, 1200));
    await page.evaluate(() => {
      const {scene, camera, controls} = window.__ARCHIGEN_VIEWER__;
      const THREE = window.THREE;
      const box = new THREE.Box3();
      scene.children.filter(object => object.userData.isBuildingMesh).forEach(object => box.expandByObject(object));
      const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.z, size.y);
      camera.position.set(center.x + span, center.y + span * .85, center.z + span);
      controls.target.copy(center); controls.update();
    });
    await new Promise(resolve => setTimeout(resolve, 700));
    await page.screenshot({path: path.join(output, 'viewer-candidate-house.png')});
    assert.deepEqual(errors, [], 'Viewer must not report JavaScript/WebGL errors');
    fs.writeFileSync(path.join(output, 'viewer-checks.json'), JSON.stringify({results, errors}, null, 2));
    console.log(JSON.stringify({results, errors}));
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exit(1);});
