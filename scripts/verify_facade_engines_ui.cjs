/** End-to-end check of the Pix2Pix vs "Stable Diffusion — hybrid" facade flow in the
 * real UI (Docker stack on :3000). Isolated browser profile; the user's project is untouched.
 *   node scripts/verify_facade_engines_ui.cjs [plan.png]
 * Writes screenshots and report.json to outputs/facade_engine_ui/<stamp>/.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const base = process.env.ARCHIGEN_BASE_URL || 'http://localhost:3000';
const plan = path.resolve(process.argv[2] || path.join(__dirname, '../data/external_test_plans/01_clean_bw_cad.png'));
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const out = path.resolve(__dirname, '../outputs/facade_engine_ui', stamp);
fs.mkdirSync(out, { recursive: true });
const report = { plan, base, steps: [] };
const step = (name, data = {}) => { report.steps.push({ name, at: new Date().toISOString(), ...data }); console.log(name, JSON.stringify(data)); };

async function main() {
  const executablePath = process.env.CHROME_PATH || ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(p => fs.existsSync(p));
  const browser = await puppeteer.launch({ executablePath, headless: true, userDataDir: path.join(out, 'profile'),
    args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const errors = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1100, deviceScaleFactor: 1 });
    page.on('pageerror', e => errors.push(e.message));
    page.on('dialog', d => d.accept());
    const shot = async name => { await page.screenshot({ path: path.join(out, `${name}.png`) }); step(`screenshot ${name}`); };
    const cardStatus = engine => page.$eval(`[data-testid=engine-card-${engine}] [role=status]`, el => el.textContent.trim());
    const waitCard = async (engine, timeout = 600000) => {
      await page.waitForFunction(e => {
        const text = document.querySelector(`[data-testid=engine-card-${e}] [role=status]`)?.textContent || '';
        return /^(Done|Failed|Discarded|Interrupted)/.test(text.trim());
      }, { timeout, polling: 500 }, engine);
      return cardStatus(engine);
    };

    // 1. Import a plan through the real upload input.
    await page.goto(`${base}/#studio`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForSelector('input[type=file]');
    const analysis = page.waitForResponse(r => r.url().includes('/predict/floorplan') && r.request().method() === 'POST', { timeout: 180000 });
    const input = await page.$('input[type=file]');
    await input.uploadFile(plan);
    const response = await analysis;
    const analysed = await response.json();
    step('plan analysed', { status: response.status(), walls: analysed.walls.length, exterior: analysed.walls.filter(w => w.is_exterior).length,
      doors: analysed.doors.length, windows: analysed.windows.length });
    assert.equal(response.status(), 200);

    // 2. Facade studio.
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('[data-testid=studio-facade]');
    await page.click('[data-testid=studio-facade]');
    await page.waitForSelector('[data-testid=compare-facade-engines]');
    await page.waitForFunction(() => /ready/.test(document.querySelector('[data-testid=hybrid-status]')?.textContent || ''), { timeout: 60000 });
    const wallOptions = await page.$$eval('select[aria-label="Exterior wall"] option:not([disabled])', options => options.map(o => ({ value: o.value, text: o.textContent })));
    const selectedWall = await page.$eval('select[aria-label="Exterior wall"]', el => el.value);
    step('facade studio', { wallOptions: wallOptions.length, selectedWall });
    await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.includes('Traditional Brick'))?.click());

    // 3. Compare both engines on the same wall.
    const started = Date.now();
    await page.click('[data-testid=compare-facade-engines]');
    await page.waitForFunction(() => /Generating|Waiting|Loading|Queued|Submitting|Composing|Checking|Validating/.test(
      document.querySelector('[data-testid=engine-card-hybrid] [role=status]')?.textContent || ''), { timeout: 30000 });
    await shot('01_hybrid_in_progress');
    const pixStatus = await waitCard('pix2pix', 120000);
    const hybStatus = await waitCard('hybrid');
    step('compare finished', { pixStatus, hybStatus, seconds: (Date.now() - started) / 1000 });
    assert.equal(pixStatus, 'Done'); assert.equal(hybStatus, 'Done');
    const compare = await page.$('[data-testid=facade-engine-compare]');
    await compare.screenshot({ path: path.join(out, '02_side_by_side.png') });
    for (let i = 0; i < 3; i++) await page.click('button[title="Zoom in"]');
    await compare.screenshot({ path: path.join(out, '03_side_by_side_zoom.png') });
    step('zoom', { zoom: await page.$eval('[data-testid=compare-zoom]', el => el.textContent) });
    await page.click('button[title="Before / after slider"]');
    await compare.screenshot({ path: path.join(out, '04_slider_zoom.png') });
    for (let i = 0; i < 3; i++) await page.click('button[title="Zoom out"]');
    await compare.screenshot({ path: path.join(out, '05_slider.png') });
    await page.click('button[title="Side by side"]');
    const cards = await page.$$eval('[data-testid^=engine-card-]', els => els.map(el => el.innerText));
    step('engine cards', { cards });

    // 4. Downloads are real PNG textures.
    const downloads = await page.evaluate(async () => {
      const result = {};
      for (const engine of ['pix2pix', 'hybrid']) {
        const href = document.querySelector(`[data-testid=download-${engine}]`)?.href;
        const img = new Image();
        await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = href; });
        result[engine] = { png: href.startsWith('data:image/png'), width: img.naturalWidth, height: img.naturalHeight, bytes: Math.round(href.length * 0.75) };
      }
      return result;
    });
    step('downloads', downloads);
    assert.ok(downloads.pix2pix.png && downloads.hybrid.png);

    // 5. Apply each engine and capture the same wall from the same camera.
    const camera = {};
    const capture3d = async (name) => {
      await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__?.scene.children.some(o => o.userData.type === 'facade'), { timeout: 60000 });
      await new Promise(r => setTimeout(r, 1500));
      const info = await page.evaluate(pose => {
        const { camera, controls, scene } = window.__ARCHIGEN_VIEWER__;
        if (pose.position) {
          camera.position.fromArray(pose.position); controls.target.fromArray(pose.target); controls.update();
        }
        const skins = scene.children.filter(o => o.userData.type === 'facade').map(o => ({ wallId: o.userData.wallId,
          texture: [o.material.map?.image?.naturalWidth || o.material.map?.image?.width, o.material.map?.image?.naturalHeight || o.material.map?.image?.height] }));
        return { position: camera.position.toArray(), target: controls.target.toArray(), skins };
      }, camera);
      if (!camera.position) Object.assign(camera, { position: info.position, target: info.target });
      await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      await page.screenshot({ path: path.join(out, `${name}.png`) });
      step(`3d ${name}`, info);
      return info;
    };
    await page.click('[data-testid=apply-hybrid]');
    await page.waitForFunction(() => /Applied/.test(document.querySelector('[data-testid=engine-card-hybrid]')?.innerText || ''));
    await page.evaluate(() => [...document.querySelectorAll('[data-testid=engine-card-hybrid] button')].find(b => /view in 3D/.test(b.textContent)).click());
    const hybrid3d = await capture3d('06_viewer_hybrid');
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('[data-testid=apply-pix2pix]');
    await page.click('[data-testid=apply-pix2pix]');
    await page.evaluate(() => [...document.querySelectorAll('[data-testid=engine-card-pix2pix] button')].find(b => /view in 3D/.test(b.textContent)).click());
    const pix3d = await capture3d('07_viewer_pix2pix_same_camera');
    report.same_camera = JSON.stringify(hybrid3d.position) === JSON.stringify(pix3d.position);

    // 6. Per-wall isolation: hybrid on a second wall, Pix2Pix stays on the first.
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('select[aria-label="Exterior wall"]');
    const other = wallOptions.find(o => o.value !== selectedWall);
    if (other) {
      await page.select('select[aria-label="Exterior wall"]', other.value);
      await page.click('[data-testid=engine-hybrid]');
      await page.click('[data-testid=generate-facade]');
      const status = await waitCard('hybrid');
      step('second wall hybrid', { wall: other.value, status });
      if (status === 'Done') {
        await page.click('[data-testid=apply-hybrid]');
        await page.evaluate(() => [...document.querySelectorAll('[data-testid=engine-card-hybrid] button')].find(b => /view in 3D/.test(b.textContent)).click());
        const both = await capture3d('08_viewer_two_walls');
        report.isolation = both.skins;
        const first = both.skins.find(s => s.wallId === selectedWall), second = both.skins.find(s => s.wallId === other.value);
        assert.ok(first && second, 'both walls carry their own facade skin');
        assert.notDeepEqual(first.texture, second.texture, 'each wall keeps its own texture');
        await page.click('[data-testid=nav-generative]');
        await page.waitForSelector('select[aria-label="Exterior wall"]');
      } else {
        report.isolation = { skipped: status, reason: await page.$eval('[data-testid=engine-card-hybrid]', el => el.innerText) };
      }
      await page.select('select[aria-label="Exterior wall"]', selectedWall);
    }

    // 7. Geometry change during a job: the late result must not be applied or shown as valid.
    await page.click('[data-testid=engine-hybrid]');
    await page.$eval('[data-testid=hybrid-seed]', el => { el.value = ''; });
    await page.type('[data-testid=hybrid-seed]', '777');
    await page.click('[data-testid=generate-facade]');
    await page.waitForFunction(() => /Generating|Waiting|Loading|Queued|Submitting/.test(document.querySelector('[data-testid=engine-card-hybrid] [role=status]')?.textContent || ''), { timeout: 30000 });
    await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === '2' && b.className.includes('w-6'))?.click());
    await new Promise(r => setTimeout(r, 1000));
    const staleShown = await page.evaluate(() => document.body.innerText.includes('different geometry'));
    await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === '1' && b.className.includes('w-6'))?.click());
    const late = await waitCard('hybrid');
    await shot('09_geometry_changed_during_job');
    step('geometry change during job', { staleNoticeWhileChanged: staleShown, finalStatus: late });
    assert.ok(staleShown, 'stale notice shown while geometry differs');
    assert.match(late, /^Discarded/);

    report.errors = errors;
    report.passed = true;
  } catch (error) {
    report.passed = false;
    report.failure = String(error.stack || error);
    throw error;
  } finally {
    report.errors = errors;
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    fs.rmSync(path.join(out, 'profile'), { recursive: true, force: true });
    console.log(`Report: ${path.join(out, 'report.json')}`);
  }
}

main().catch(error => { console.error(error); process.exit(1); });
