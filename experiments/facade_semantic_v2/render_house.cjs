/** Render every real house side and verify fixed geometry across texture modes. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

async function main() {
  const project = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
  const puppeteer = require(path.join(project, 'frontend/node_modules/puppeteer-core'));
  const executablePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(p => fs.existsSync(p));
  assert.ok(executablePath);
  const browser = await puppeteer.launch({executablePath, headless: true,
    args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl']});
  const errors = [], checks = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({width: 1600, height: 1050, deviceScaleFactor: 1});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && /THREE|WebGL|shader/i.test(message.text())) errors.push(message.text());
    });
    await page.goto('http://127.0.0.1:3001/tmp/facade-multiface-v2/', {waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => window.__FACADE_MULTIFACE__ && window.__ARCHIGEN_VIEWER__?.renderer.domElement.isConnected);
    for (const mode of ['candidate', 'baseline', 'plain', 'guide']) {
      await page.click(`[data-mode="${mode}"]`);
      await page.waitForFunction(mode => {
        if (window.__FACADE_MULTIFACE__?.mode !== mode) return false;
        const skins = window.__ARCHIGEN_VIEWER__.scene.children.filter(o => o.userData.type === 'facade');
        return skins.length === 15 && skins.every(o => o.material.map?.image?.complete && o.material.map.image.src.includes(mode === 'plain' ? 'plain.png' : mode === 'guide' ? 'sketch.png' : `${mode}.png`));
      }, {timeout: 90000}, mode);
      const result = await page.evaluate(() => {
        const {scene} = window.__ARCHIGEN_VIEWER__, {panels, mode} = window.__FACADE_MULTIFACE__;
        return {mode, walls: panels.map(panel => {
          const skin = scene.children.find(o => o.userData.type === 'facade' && o.userData.wallId === panel.wall_id);
          const p = skin.geometry.attributes.position, uv = skin.geometry.attributes.uv;
          const frame = panel.facade_frame, bounds = skin.userData.uvBounds;
          let uvError = 0;
          for (let i = 0; i < p.count; i++) {
            const along = (p.getX(i) - p.getX(0)) * frame.direction[0] + (p.getZ(i) - p.getZ(0)) * frame.direction[1];
            uvError = Math.max(uvError,
              Math.abs(uv.getX(i) - (bounds.u_min + (bounds.u_max - bounds.u_min) * along / frame.length_m)),
              Math.abs(uv.getY(i) - (bounds.v_min + (bounds.v_max - bounds.v_min) * p.getY(i) / panel.wall_height_m)));
          }
          const openings = scene.children.filter(o => o.userData.facadeOpening?.associated_wall === panel.wall_id).map(o => {
            const spec = o.userData.facadeOpening;
            const along = (o.position.x - p.getX(0)) * frame.direction[0] + (o.position.z - p.getZ(0)) * frame.direction[1];
            return {id: spec.id, position: [o.position.x, o.position.y, o.position.z], width: spec.width_m,
              height: spec.height_m, centerErrorM: Math.abs(along - spec.position_along_wall_m)};
          }).sort((a, b) => a.id.localeCompare(b.id));
          return {wallId: panel.wall_id, uvError, expectedOpenings: panel.openings.length, openings,
            textureSize: [skin.material.map.image.width, skin.material.map.image.height]};
        }), unresolved: scene.children.filter(o => o.userData.type === 'wall_assembly').flatMap(o => o.userData.unresolvedOpenings)};
      });
      assert.equal(result.walls.length, 15);
      assert.deepEqual(result.unresolved, []);
      for (const wall of result.walls) {
        assert.equal(wall.openings.length, wall.expectedOpenings, wall.wallId + ' opening count');
        assert.ok(wall.uvError < 1e-5, wall.wallId + ' planar UV mapping');
        assert.ok(wall.openings.every(o => o.centerErrorM < 1e-4), wall.wallId + ' frame position');
        if (mode === 'candidate') assert.deepEqual(wall.textureSize, [512, 512]);
      }
      if (checks.length) {
        assert.deepEqual(result.walls.map(w => w.openings), checks[0].walls.map(w => w.openings),
          'Changing facade texture must preserve all opening positions and sizes');
      }
      checks.push(result);
      await page.select('select[aria-label="Facade side"]', 'ground_east');
      await page.click('[data-action="face"]');
      await new Promise(r => setTimeout(r, 900));
      await page.screenshot({path: path.join(output, `${mode}-ground_east.png`)});
    }
    await page.click('[data-mode="candidate"]');
    const ids = await page.evaluate(() => window.__FACADE_MULTIFACE__.panels.map(p => p.wall_id));
    for (const id of ids) {
      await page.select('select[aria-label="Facade side"]', id);
      await page.click('[data-action="face"]');
      await new Promise(r => setTimeout(r, 800));
      await page.screenshot({path: path.join(output, `candidate-${id}.png`)});
    }
    // Return to all floors before capturing both corners of the complete house.
    await page.select('select[aria-label="Visible building levels"]', 'all');
    await new Promise(r => setTimeout(r, 1000));
    for (const [action, filename] of [['overview', 'candidate-house-overview.png'], ['reverse', 'candidate-house-opposite.png']]) {
      await page.click(`[data-action="${action}"]`); await new Promise(r => setTimeout(r, 900));
      await page.screenshot({path: path.join(output, filename)});
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'viewer-checks.json'), JSON.stringify({checks, errors}, null, 2));
    console.log(JSON.stringify({modes: checks.length, walls: 15, openings: checks[0].walls.reduce((n, w) => n + w.openings.length, 0),
      maxUvError: Math.max(...checks.flatMap(r => r.walls.map(w => w.uvError))), errors}));
  } finally {await browser.close();}
}
main().catch(error => {console.error(error); process.exit(1);});
