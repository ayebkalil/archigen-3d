const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

const mode = process.argv[2] || 'before'; // 'before' | 'after'

async function capture() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  console.log(`[STAGE 4] Capturing ${mode.toUpperCase()} state with Edge...`);

  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-gpu-rasterization',
      '--enable-zero-copy',
      '--ignore-gpu-blocklist',
      '--use-gl=angle',
      '--use-angle=d3d11',
      '--window-size=1600,1000'
    ]
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    const url = 'http://localhost:3000/#studio';
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    console.log('[STAGE 4] Waiting for .room-polygon to appear...');
    await page.waitForSelector('.room-polygon', { timeout: 60000 });
    console.log('[STAGE 4] .room-polygon detected.');
    await new Promise(r => setTimeout(r, 1500));

    // Click on Master Bedroom polygon
    await page.evaluate(() => {
      const polys = Array.from(document.querySelectorAll('.room-polygon'));
      const target = polys.length > 1 ? polys[1] : polys[0];
      target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });

    await new Promise(r => setTimeout(r, 1000));

    // Click Generate Design button
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });

    // Wait for GLB models to load
    await new Promise(r => setTimeout(r, 5000));

    // Measure FPS over 120 frames in the canvas
    console.log('[STAGE 4] Measuring rendering frame rate...');
    const fpsData = await page.evaluate(async () => {
      return new Promise((resolve) => {
        let frames = 0;
        let lastTime = performance.now();
        const start = performance.now();
        const frameDeltas = [];

        function tick(now) {
          frames++;
          const delta = now - lastTime;
          lastTime = now;
          if (frames > 10) frameDeltas.push(delta); // Skip initial warm-up

          if (now - start >= 2000) { // Measure over 2.0 seconds
            const avgDelta = frameDeltas.reduce((a, b) => a + b, 0) / frameDeltas.length;
            const fps = 1000 / avgDelta;
            const minDelta = Math.min(...frameDeltas);
            const maxDelta = Math.max(...frameDeltas);
            resolve({
              fps: Math.round(fps * 10) / 10,
              avgDeltaMs: Math.round(avgDelta * 100) / 100,
              minFps: Math.round((1000 / maxDelta) * 10) / 10,
              maxFps: Math.round((1000 / minDelta) * 10) / 10,
              totalFrames: frames
            });
          } else {
            requestAnimationFrame(tick);
          }
        }
        requestAnimationFrame(tick);
      });
    });

    console.log(`[STAGE 4] ${mode.toUpperCase()} Measured FPS:`, fpsData);

    const outDir = path.resolve(__dirname, '../outputs/design_stage4');
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    const imgName = `stage4_${mode}_screenshot.png`;
    const outPath = path.join(outDir, imgName);

    await page.screenshot({ path: outPath });
    console.log(`[STAGE 4] Saved ${mode} screenshot to:`, outPath);

    const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";
    fs.copyFileSync(outPath, path.join(artDir, imgName));

    // Save JSON metrics
    const metricsPath = path.join(outDir, `fps_${mode}.json`);
    fs.writeFileSync(metricsPath, JSON.stringify(fpsData, null, 2));

  } finally {
    await browser.close();
  }
}

capture().catch(err => {
  console.error('[STAGE 4 ERROR]:', err);
  process.exit(1);
});
