const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function captureRoom(browser, roomKey, viewMode, outFileName) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 750, deviceScaleFactor: 1 });

  // Use Vite dev server running on localhost:3000
  const url = `http://localhost:3000/render_showcase_rooms.html?room=${roomKey}&view=${viewMode}`;
  console.log(`Rendering ${roomKey} (${viewMode}) at: ${url}`);

  await page.goto(url, { waitUntil: 'networkidle0' });

  // Wait for signal __SCENE_READY__
  await page.waitForFunction(() => window.__SCENE_READY__ === true, { timeout: 15000 });
  await new Promise(r => setTimeout(r, 800));

  const outDir = path.resolve(__dirname, '../outputs/design_stage2');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, outFileName);

  await page.screenshot({ path: outPath });
  console.log(`Saved screenshot to: ${outPath}`);
  await page.close();
  return outPath;
}

async function main() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  console.log('Launching browser with Edge...');
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--use-gl=angle', '--use-angle=swiftshader']
  });

  try {
    // Capture 3 perspective views
    await captureRoom(browser, 'bedroom', 'perspective', 'real_bedroom_showcase.png');
    await captureRoom(browser, 'living_room', 'perspective', 'real_living_room_showcase.png');
    await captureRoom(browser, 'kitchen', 'perspective', 'real_kitchen_showcase.png');

    // Also capture top-down views
    await captureRoom(browser, 'bedroom', 'top', 'real_bedroom_topdown.png');
    await captureRoom(browser, 'living_room', 'top', 'real_living_room_topdown.png');
    await captureRoom(browser, 'kitchen', 'top', 'real_kitchen_topdown.png');
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error('Error during capture:', err);
  process.exit(1);
});
