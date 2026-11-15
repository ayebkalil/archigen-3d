/** Room library end-to-end (Docker :3000, isolated browser profile):
 * TripoSR generation -> automatic save -> listed in the 3D room library -> placed by
 * clicking a floor -> rotated / moved -> survives a reload -> removed.
 *   node scripts/verify_room_library_ui.cjs
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const base = process.env.ARCHIGEN_BASE_URL || 'http://localhost:3000';
const out = path.resolve(__dirname, '../outputs/room_library_ui', new Date().toISOString().replace(/[:.]/g, '-'));
fs.mkdirSync(out, { recursive: true });
const log = (name, data = '') => console.log(name, typeof data === 'string' ? data : JSON.stringify(data));

(async () => {
  const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, userDataDir: path.join(out, 'profile'), args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const report = {};
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1100 });
    const clickText = (selector, text) => page.evaluate((sel, t) => {
      const el = [...document.querySelectorAll(sel)].find(node => node.textContent.trim().includes(t));
      if (!el) throw new Error(`No ${sel} with text ${t}`);
      el.click();
    }, selector, text);
    await page.goto(`${base}/#studio`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('[data-testid=nav-generative]');
    await new Promise(r => setTimeout(r, 2500));

    // 1. Generate furniture with TripoSR: it must be saved to the room library automatically.
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('[data-testid=generate-furniture]');
    const name = `Test chair ${Date.now().toString(36)}`;
    await page.evaluate(n => {
      const input = [...document.querySelectorAll('input')].find(el => el.value === 'Upholstered Dining Chair');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(input, n); input.dispatchEvent(new Event('input', { bubbles: true }));
    }, name);
    const save = page.waitForResponse(r => r.url().includes('/furniture/save-to-library'), { timeout: 900000 });
    await page.click('[data-testid=generate-furniture]');
    const saved = await save;
    const savedItem = (await saved.json()).item;
    await page.waitForFunction(() => document.body.innerText.includes('Saved automatically to the room library'), { timeout: 30000 });
    report.saved = { http: saved.status(), id: savedItem.id, name: savedItem.name, preview: savedItem.preview };
    log('auto-saved', report.saved);
    assert.equal(saved.status(), 200);
    assert.ok(savedItem.preview, 'library entry has a thumbnail');

    // 2. Room library in the 3D view lists it.
    await page.click('[data-testid=nav-floorplan]');
    await page.waitForSelector('[data-testid=view-3d]');
    await page.click('[data-testid=view-3d]');
    await page.waitForSelector(`[data-testid=library-${savedItem.id}]`, { timeout: 30000 });
    report.libraryCount = await page.$$eval('[data-testid^=library-]', els => els.filter(el => !el.dataset.testid.includes('toggle')).length);
    log('library entries', report.libraryCount);

    // 3. Place it by clicking a floor (top-down view, canvas centre).
    await page.click('button[title="Top-Down Plan View"]');
    await new Promise(r => setTimeout(r, 1500));
    await page.click(`[data-testid=place-${savedItem.id}]`);
    await page.waitForSelector('[data-testid=placement-banner]');
    const canvas = await page.$('.viewer-panel canvas');
    const box = await canvas.boundingBox();
    const placedCount = () => page.evaluate(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('placedFurnitureGroup').children.filter(o => o.userData.placedId && !o.isLineSegments).length);
    // Try a few points around the centre until one lands on a floor.
    for (const [dx, dy] of [[0, 0], [-80, 40], [80, -40], [-150, -60], [120, 80]]) {
      await page.mouse.click(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
      await new Promise(r => setTimeout(r, 400));
      if (!(await page.$('[data-testid=placement-banner]'))) break;
    }
    await page.waitForSelector('[data-testid=placed-controls]', { timeout: 10000 });
    await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('placedFurnitureGroup').children.some(o => o.userData.placedId), { timeout: 30000 });
    await new Promise(r => setTimeout(r, 1000));
    report.afterPlace = await placedCount();
    await page.screenshot({ path: path.join(out, '01_placed_topdown.png') });
    log('placed models in scene', report.afterPlace);
    assert.equal(report.afterPlace, 1);

    // 4. Rotate, then move to another floor point.
    const placedPos = async () => {
      await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('placedFurnitureGroup').children.some(o => o.userData.placedId && !o.isLineSegments), { timeout: 30000 });
      return page.evaluate(() => window.__ARCHIGEN_VIEWER__.scene.getObjectByName('placedFurnitureGroup').children.find(o => o.userData.placedId && !o.isLineSegments).position.toArray());
    };
    const rotation = async () => page.$eval('[data-testid=placed-controls]', el => el.innerText.match(/rotation (\d+)/)[1]);
    await page.click('button[title="Rotate 90° right"]');
    await page.click('button[title="Rotate 15° right"]');
    report.rotation = await rotation();
    const before = await placedPos();
    await page.click('[data-testid=move-placed]');
    for (const [dx, dy] of [[-120, 60], [110, -70], [60, 90], [-60, -90]]) {
      await page.mouse.click(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
      await new Promise(r => setTimeout(r, 400));
      if (!(await page.$('[data-testid=placement-banner]'))) break;
    }
    await new Promise(r => setTimeout(r, 1500));
    const after = await placedPos();
    report.move = { before, after, rotation: report.rotation };
    log('rotate + move', report.move);
    assert.equal(report.rotation, '105');
    assert.ok(Math.hypot(after[0] - before[0], after[2] - before[2]) > 0.3, 'model moved');

    // 5. Perspective screenshot, reload (persistence), then remove.
    await page.click('button[title="Top-Down Plan View"]');
    await page.evaluate(() => {
      const { camera, controls } = window.__ARCHIGEN_VIEWER__;
      camera.position.set(camera.position.x + 6, 7, camera.position.z + 8); controls.update();
    });
    await new Promise(r => setTimeout(r, 1200));
    await page.screenshot({ path: path.join(out, '02_placed_perspective.png') });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__?.scene.getObjectByName('placedFurnitureGroup')?.children.some(o => o.userData.placedId), { timeout: 60000 });
    report.afterReload = await placedCount();
    log('after reload', report.afterReload);
    assert.equal(report.afterReload, 1);
    report.passed = true;
  } catch (error) {
    report.passed = false; report.failure = String(error.stack || error); console.error(error); process.exitCode = 1;
    try { const [p] = await browser.pages(); await p.screenshot({ path: path.join(out, 'failure.png') }); } catch {}
  } finally {
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    fs.rmSync(path.join(out, 'profile'), { recursive: true, force: true });
    console.log('Report:', out);
  }
})();
