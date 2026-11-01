/** UI verification against native floorplan API, with recorded neural assets for
 * furniture/facade previews. Never writes to the user's furniture library.
 * Requires Vite :3000, API :8000, and local outputs from the model experiments.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

async function main() {
  const root = path.resolve(__dirname, '..');
  const out = path.join(root, 'outputs/workspace_design');
  fs.mkdirSync(out, { recursive: true });
  const furniture = JSON.parse(
    fs.readFileSync(path.join(root, 'outputs/furniture_quality/studio-fp16/result.json'))
  );
  furniture.model_glb_base64 =
    'data:model/gltf-binary;base64,' +
    fs
      .readFileSync(path.join(root, 'outputs/furniture_quality/studio-fp16/model.glb'))
      .toString('base64');
  const facade = {
    status: 'success',
    ...JSON.parse(
      fs.readFileSync(path.join(root, 'outputs/phase3/composite_brick/viewer_fixture.json'))
    ),
  };
  const browser = await puppeteer.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
    args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const errors = [],
    reports = [];
  let facadeRequests = 0,
    furnitureRequests = 0;
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000 });
    page.on('pageerror', (e) => errors.push(e.message));
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      if (request.url().includes('/furniture/generate-3d')) {
        furnitureRequests++;
        const body =
          furnitureRequests === 1
            ? { detail: 'Preview test: generation unavailable. Please retry.' }
            : furniture;
        return request.respond({
          status: furnitureRequests === 1 ? 503 : 200,
          contentType: 'application/json',
          body: JSON.stringify(body),
        });
      }
      if (request.url().includes('/facade/generate')) {
        facadeRequests++;
        assert.equal(typeof JSON.parse(request.postData()).style, 'string');
        return request.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(facade),
        });
      }
      return request.continue();
    });
    const paint = () =>
      page.evaluate(
        () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
      );
    const screenshot = async (name) => {
      await paint();
      await page.screenshot({ path: path.join(out, name + '.png'), fullPage: true });
    };
    const checkWidth = async (name) => {
      const result = await page.evaluate(() => ({
        viewport: innerWidth,
        document: document.documentElement.scrollWidth,
        bodyOverflow: getComputedStyle(document.body).overflow,
      }));
      assert.ok(
        result.document <= result.viewport + 1,
        `${name}: horizontal overflow ${JSON.stringify(result)}`
      );
      reports.push({ name, ...result });
    };
    await page.goto('http://127.0.0.1:3000/#studio', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelectorAll('.room-chips button').length > 0, {
      timeout: 120000,
    });
    await screenshot('floorplan-desktop');
    await checkWidth('floorplan-desktop');
    await page.click('.room-chips button');
    await page.waitForSelector('.room-design-panel');
    await page.click('[title="Close panel"]');
    await page.click('[data-testid=view-2d]');
    assert.equal(await page.$('.viewer-panel'), null);
    await page.click('[data-testid=view-3d]');
    assert.equal(await page.$('.plan-panel'), null);
    await page.click('[title="Toggle 3D Building Roof Enclosure (Activate / Deactivate)"]');
    await page.click('[title="Classic Sims Pitched Hipped Gable Roof"]');
    await screenshot('house-desktop');
    await page.click('[data-testid=nav-generative]');
    await page.waitForSelector('.furniture-stage canvas');
    await screenshot('furniture-empty');
    await page.click('[data-testid=generate-furniture]');
    await page.waitForSelector('[role=alert]');
    await page.click('[aria-label="Dismiss error"]');
    await page.click('[data-testid=generate-furniture]');
    await page.waitForSelector('.model-result-bar');
    await page.waitForFunction(() => !document.body.textContent.includes('Generating model…'));
    await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 1800)));
    await page.evaluate(() =>
      [...document.querySelectorAll('button')]
        .find((b) => b.textContent.includes('Rotate: ON'))
        ?.click()
    );
    await screenshot('furniture-desktop');
    await page.click('[data-testid=studio-facade]');
    assert.equal(facadeRequests, 0, 'Tab navigation must not start generation');
    await page.click('.studio-style-card:nth-child(3)');
    await page.click('[data-testid=generate-facade]');
    await page.waitForSelector('img[alt="Realistic Facade"]');
    await screenshot('facade-desktop');
    await page.click('[data-testid=apply-facade]');
    await page.waitForFunction(() =>
      document.body.textContent.includes('Applied to 3D Floorplan!')
    );
    for (const width of [1366, 768, 390]) {
      await page.setViewport({ width, height: width === 390 ? 844 : 900 });
      if (!(await page.$('img[alt="Realistic Facade"]'))) {
        await page.click('.studio-style-card:nth-child(3)');
        await page.click('[data-testid=generate-facade]');
        await page.waitForSelector('img[alt="Realistic Facade"]');
      }
      await checkWidth(`facade-${width}`);
      await screenshot(`facade-${width}`);
      await page.click('[data-testid=studio-furniture]');
      if (!(await page.$('.model-result-bar'))) {
        await page.click('[data-testid=generate-furniture]');
        await page.waitForSelector('.model-result-bar');
        await page.evaluate(() => new Promise((r) => setTimeout(r, 1000)));
      }
      await checkWidth(`furniture-${width}`);
      await screenshot(`furniture-${width}`);
      await page.click('[data-testid=nav-floorplan]');
      await page.click('[data-testid=view-split]');
      await checkWidth(`floorplan-${width}`);
      await screenshot(`floorplan-${width}`);
      if (width === 390)
        assert.equal(
          await page.evaluate(() => getComputedStyle(document.body).overflow),
          'auto',
          'Mobile page must scroll'
        );
      await page.click('[data-testid=nav-generative]');
      await page.click('[data-testid=studio-facade]');
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      path.join(out, 'verification.json'),
      JSON.stringify(
        {
          errors,
          reports,
          furnitureRequests,
          facadeRequests,
          source: 'Live floorplan detection; recorded furniture/facade assets for preview UI',
        },
        null,
        2
      )
    );
    console.log(JSON.stringify({ errors, reports, furnitureRequests, facadeRequests }));
  } finally {
    await browser.close();
  }
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
