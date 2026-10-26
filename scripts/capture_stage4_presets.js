const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function capturePresets() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  console.log('[STAGE 4] Capturing Camera Presets...');

  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--enable-gpu-rasterization',
      '--use-gl=angle',
      '--use-angle=d3d11',
      '--window-size=1600,1000'
    ]
  });

  const outDir = path.resolve(__dirname, '../outputs/design_stage4');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForSelector('.room-polygon', { timeout: 60000 });
    await new Promise(r => setTimeout(r, 1500));

    // Select room & generate furnishings
    await page.evaluate(() => {
      const polys = Array.from(document.querySelectorAll('.room-polygon'));
      const target = polys.length > 1 ? polys[1] : polys[0];
      target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 4500));

    // 1. Capture Axonometric Preset
    console.log('[STAGE 4] Clicking Axo Preset...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const axoBtn = btns.find(b => b.textContent.includes('Axo'));
      if (axoBtn) axoBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));
    const axoPath = path.join(outDir, 'stage4_preset_isometric.png');
    await page.screenshot({ path: axoPath });
    fs.copyFileSync(axoPath, path.join(artDir, 'stage4_preset_isometric.png'));
    console.log('[STAGE 4] Saved Axonometric preset screenshot.');

    // 2. Capture Top-Down Plan Preset
    console.log('[STAGE 4] Clicking Plan Preset...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const planBtn = btns.find(b => b.textContent.includes('Plan'));
      if (planBtn) planBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));
    const planPath = path.join(outDir, 'stage4_preset_topdown.png');
    await page.screenshot({ path: planPath });
    fs.copyFileSync(planPath, path.join(artDir, 'stage4_preset_topdown.png'));
    console.log('[STAGE 4] Saved Top-Down Plan preset screenshot.');

    // 3. Capture Room Focus Preset
    console.log('[STAGE 4] Clicking Focus Preset...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const focusBtn = btns.find(b => b.textContent.includes('Focus'));
      if (focusBtn) focusBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));
    const focusPath = path.join(outDir, 'stage4_preset_room_focus.png');
    await page.screenshot({ path: focusPath });
    fs.copyFileSync(focusPath, path.join(artDir, 'stage4_preset_room_focus.png'));
    console.log('[STAGE 4] Saved Room Focus preset screenshot.');

  } finally {
    await browser.close();
  }
}

capturePresets().catch(err => {
  console.error('[STAGE 4 ERROR]:', err);
  process.exit(1);
});
