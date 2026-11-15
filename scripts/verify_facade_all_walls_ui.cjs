/** Checks the "All walls" facade option in the real UI (Docker :3000), isolated profile.
 * Sample house (two storeys): Upper floor · All walls with Stable Diffusion, then
 * Ground floor · All walls with Pix2Pix. Every wall of the level must carry a facade in 3D.
 *   node scripts/verify_facade_all_walls_ui.cjs
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const base = process.env.ARCHIGEN_BASE_URL || 'http://localhost:3000';
const out = path.resolve(__dirname, '../outputs/facade_engine_ui', `all-walls-${new Date().toISOString().replace(/[:.]/g, '-')}`);
fs.mkdirSync(out, { recursive: true });

(async () => {
  const executablePath = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
  const browser = await puppeteer.launch({ executablePath, headless: true, userDataDir: path.join(out, 'profile'),
    args: ['--no-sandbox', '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const report = {};
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1100 });
    await page.goto(`${base}/#studio`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('[data-testid=nav-generative]');
    await page.waitForFunction(() => !document.body.innerText.includes('Restoring your project'));
    await new Promise(r => setTimeout(r, 2000));
    await page.click('[data-testid=nav-generative]');
    await page.click('[data-testid=studio-facade]');
    await page.waitForSelector('select[aria-label="Exterior wall"]');
    report.options = await page.$$eval('select[aria-label="Exterior wall"] option', o => o.filter(x => x.value.startsWith('__all__')).map(x => x.textContent));
    console.log('all-walls options', report.options);

    const runLevel = async (level, engine, style) => {
      await page.select('select[aria-label="Exterior wall"]', `__all__:${level}`);
      await page.click(`[data-testid=engine-${engine}]`);
      await page.evaluate(s => [...document.querySelectorAll('button')].find(b => b.textContent.includes(s) && b.getAttribute('aria-pressed') === 'false')?.click(), style);
      const label = await page.$eval('[data-testid=generate-facade]', b => b.textContent.trim());
      const started = Date.now();
      await page.click('[data-testid=generate-facade]');
      await page.waitForFunction(() => /walls applied/.test(document.querySelector('[data-testid=bulk-progress]')?.innerText || ''), { timeout: 900000, polling: 1000 });
      const summary = await page.$eval('[data-testid=bulk-progress]', el => el.innerText);
      const gallery = await page.$('[data-testid=facade-level-gallery]');
      await gallery.screenshot({ path: path.join(out, `${level}_${engine}_gallery.png`) });
      const cards = await page.$$eval('[data-testid^=gallery-]', els => els.map(el => el.innerText.split(/\s*\n\s*/).slice(0, 3).join(' ')));
      console.log('gallery cards:', cards.length, cards.filter(c => /Applied now/.test(c)).length, 'applied now');
      assert.ok(cards.length > 0 && cards.every(c => /Applied now/.test(c)), 'every gallery card shows the new texture');
      await page.screenshot({ path: path.join(out, `${level}_${engine}_sidebar.png`) });
      console.log(level, engine, label, '→', summary.replace(/\n/g, ' | '), `${((Date.now() - started) / 1000).toFixed(0)}s`);
      return { label, summary };
    };
    report.upper = await runLevel('upper', 'hybrid', 'Traditional Brick');
    report.ground = await runLevel('ground', 'pix2pix', 'Mediterranean Stone');

    // 3D: every exterior wall of both levels now carries its own facade skin.
    await page.evaluate(() => [...document.querySelectorAll('[data-testid=bulk-progress] button')].find(b => /View in 3D/.test(b.textContent)).click());
    await page.waitForFunction(() => window.__ARCHIGEN_VIEWER__?.scene.children.some(o => o.userData.type === 'facade'), { timeout: 60000 });
    await new Promise(r => setTimeout(r, 2500));
    report.skins = await page.evaluate(() => window.__ARCHIGEN_VIEWER__.scene.children.filter(o => o.userData.type === 'facade')
      .map(o => ({ wall: o.userData.wallId, texture: [o.material.map?.image?.width, o.material.map?.image?.height] })));
    await page.screenshot({ path: path.join(out, 'viewer_all_walls.png') });
    console.log('facade skins in 3D:', report.skins.length, JSON.stringify(report.skins));
    assert.match(report.upper.summary, /(\d+) \/ \1 walls applied/);
    assert.match(report.ground.summary, /(\d+) \/ \1 walls applied/);
    report.passed = true;
  } catch (error) {
    report.passed = false; report.failure = String(error.stack || error); console.error(error); process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
    fs.rmSync(path.join(out, 'profile'), { recursive: true, force: true });
    console.log('Report:', out);
  }
})();
