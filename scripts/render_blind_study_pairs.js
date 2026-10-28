const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

async function renderAllPairs() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--use-gl=angle',
      '--use-angle=d3d11',
      '--enable-gpu-rasterization',
      '--enable-webgl',
      '--window-size=1000,800'
    ]
  });

  const outDir = path.resolve(__dirname, '../outputs/blind_study/renders');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const dataFile = path.resolve(__dirname, '../outputs/blind_study/random_baseline_layouts.json');
  const rawData = fs.readFileSync(dataFile, 'utf8');
  const studyData = JSON.parse(rawData);
  const rooms = studyData.rooms;

  console.log(`Starting headless 3D render of ${rooms.length} rooms (64 images total)...`);

  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 800, deviceScaleFactor: 1 });

  page.on('console', msg => {
    const text = msg.text();
    if (!text.includes('THREE.')) console.log('PAGE:', text);
  });
  page.on('pageerror', err => console.error('PAGE ERROR:', err.toString()));
  page.on('response', response => {
    if (response.status() >= 400) {
      console.log('HTTP ERROR:', response.status(), response.url());
    }
  });

  await page.goto('http://localhost:8089/blind_study_renderer.html', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.__RENDERER_INITIALIZED__ === true, { timeout: 15000 });
  console.log('Renderer page loaded and initialized successfully.');

  const t0 = Date.now();

  for (let i = 0; i < rooms.length; i++) {
    const r = rooms[i];
    const roomLabel = `room_${String(i + 1).padStart(2, '0')}`;
    const rtype = r.room_type;
    console.log(`[${i + 1}/${rooms.length}] Rendering ${roomLabel} (${rtype}, ${r.area_m2} m²)...`);

    // 1. Render Condition A: Engine Layout
    const engineImgPath = path.join(outDir, `${roomLabel}_img_engine.png`);
    await page.evaluate(async (room) => {
      await window.renderRoom(room, 'engine');
    }, r);

    await page.waitForFunction(() => window.__RENDER_DONE === true, { timeout: 20000 });
    // Brief settle for shadow buffer
    await new Promise(res => setTimeout(res, 250));
    await page.screenshot({ path: engineImgPath, type: 'png' });

    // 2. Render Condition B: Random Baseline
    const randImgPath = path.join(outDir, `${roomLabel}_img_random.png`);
    await page.evaluate(async (room) => {
      await window.renderRoom(room, 'random');
    }, r);

    await page.waitForFunction(() => window.__RENDER_DONE === true, { timeout: 20000 });
    await new Promise(res => setTimeout(res, 250));
    await page.screenshot({ path: randImgPath, type: 'png' });
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n[SUCCESS] Rendered all 64 images across ${rooms.length} rooms in ${elapsed}s!`);
  console.log(`Images saved to: ${outDir}`);

  await browser.close();
}

renderAllPairs().catch(err => {
  console.error('Fatal render error:', err);
  process.exit(1);
});
