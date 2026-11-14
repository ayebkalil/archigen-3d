/** Actual Docker studio generation, preview, navigation, reload and GLB download.
 * Uses an isolated browser profile; does not modify the user's browser project.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

async function main() {
  const restoreOnly = process.argv.includes('--restore-only');
  const output = path.resolve(process.argv.slice(2).find(value => !value.startsWith('--')) || 'outputs/docker/browser');
  fs.mkdirSync(output, {recursive: true});
  const executablePath = process.env.CHROME_PATH || [
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
  ].find(fs.existsSync);
  const browser = await puppeteer.launch({executablePath, headless: true,
    userDataDir: path.join(output, 'profile'),
    args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  const errors = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({width: 1600, height: 1100});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && /GLB|WebGL|shader/i.test(message.text())) errors.push(message.text());
    });
    await page.goto(process.env.ARCHIGEN_BASE_URL || 'http://localhost:3000/#studio', {waitUntil: 'networkidle0'});
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('[data-testid=generate-furniture]');
    let result;
    if (restoreOnly) {
      result = JSON.parse(fs.readFileSync(path.join(output, 'summary.json'), 'utf8'));
    } else {
      const pending = page.waitForResponse(response => response.url().includes('/furniture/generate-3d'), {timeout: 840000});
      await page.click('[data-testid=generate-furniture]');
      const response = await pending;
      result = await response.json();
      assert.equal(response.status(), 200, JSON.stringify(result));
    }
    assert.equal(result.quality.checkpoint_loading, 'strict');
    const waitModel = () => page.waitForFunction(() => document.body.textContent.includes('2048px neural atlas'), {timeout: 60000});
    await waitModel();
    await page.click('[data-testid=studio-facade]');
    await page.click('[data-testid=studio-furniture]');
    await waitModel();
    await page.click('[data-testid=nav-floorplan]');
    await page.click('[data-testid=nav-generative]');
    await waitModel();
    await page.reload({waitUntil: 'networkidle0'});
    await waitModel();
    const client = await page.createCDPSession();
    await client.send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: output});
    const downloaded = path.join(output, result.id + '.glb');
    const expected = restoreOnly ? fs.readFileSync(downloaded) : Buffer.from(result.model_glb_base64.split(',').pop(), 'base64');
    if (fs.existsSync(downloaded)) fs.unlinkSync(downloaded);
    await page.evaluate(() => [...document.querySelectorAll('button')].find(button => button.textContent.includes('Download GLB')).click());
    await page.waitForFunction(() => document.querySelectorAll('canvas').length > 0);
    for (let i = 0; i < 100 && !fs.existsSync(downloaded); i++) await new Promise(resolve => setTimeout(resolve, 100));
    assert.ok(fs.existsSync(downloaded), 'GLB download did not complete');
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(downloaded)).digest('hex'),
      crypto.createHash('sha256').update(expected).digest('hex'));
    await new Promise(resolve => setTimeout(resolve, 3000));
    await page.screenshot({path: path.join(output, restoreOnly ? 'studio-after-restart.png' : 'studio.png'), fullPage: true});
    assert.deepEqual(errors, []);
    delete result.model_glb_base64;
    delete result.preview_image_base64;
    const summary = {...result, navigation_restored: true, reload_restored: true,
      download_bytes: expected.length, browser_errors: errors, verified_after_restart: restoreOnly};
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2));
    console.log(JSON.stringify(summary, null, 2));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
