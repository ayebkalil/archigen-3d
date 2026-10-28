const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

async function renderB2() {
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

  const outDir = path.resolve(__dirname, '../outputs/blind_study/renders_b2');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const dataFile = path.resolve(__dirname, '../outputs/blind_study/data/b2_random_layouts.json');
  const rawData = fs.readFileSync(dataFile, 'utf8');
  const studyData = JSON.parse(rawData);
  const rooms = studyData.rooms;

  console.log(`Starting headless 3D render of B2 baseline for ${rooms.length} rooms...`);

  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 800, deviceScaleFactor: 1 });

  page.on('console', msg => {
    const text = msg.text();
    if (!text.includes('THREE.')) console.log('PAGE:', text);
  });
  page.on('pageerror', err => console.error('PAGE ERROR:', err.toString()));

  await page.goto('http://localhost:8089/blind_study_renderer.html', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.__RENDERER_INITIALIZED__ === true, { timeout: 15000 });
  console.log('Renderer initialized successfully.');

  const t0 = Date.now();

  for (let i = 0; i < rooms.length; i++) {
    const r = rooms[i];
    const roomLabel = `room_${String(i + 1).padStart(2, '0')}`;
    const rtype = r.room_type;
    console.log(`[${i + 1}/${rooms.length}] Rendering B2 ${roomLabel} (${rtype})...`);

    const b2ImgPath = path.join(outDir, `${roomLabel}_img_b2.png`);
    await page.evaluate(async (room) => {
      await window.renderRoom(room, 'b2');
    }, r);

    await page.waitForFunction(() => window.__RENDER_DONE === true, { timeout: 20000 });
    await new Promise(res => setTimeout(res, 250));
    await page.screenshot({ path: b2ImgPath, type: 'png' });
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n[SUCCESS] Rendered all 32 B2 images in ${elapsed}s!`);
  console.log(`Images saved to: ${outDir}`);

  await browser.close();
}

renderB2().catch(err => {
  console.error('Fatal render error:', err);
  process.exit(1);
});
