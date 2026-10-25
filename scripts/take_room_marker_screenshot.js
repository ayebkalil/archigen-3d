const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function capture() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const htmlPath = path.resolve(__dirname, '../frontend/public/test_room_facing_marker.html');
  const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');

  console.log('Launching browser to render:', fileUrl);
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--allow-file-access-from-files', '--use-gl=angle', '--use-angle=swiftshader']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 600, deviceScaleFactor: 1 });
  await page.goto(fileUrl, { waitUntil: 'load' });

  // Wait for signal __SCENE_READY__
  await page.waitForFunction(() => window.__SCENE_READY__ === true, { timeout: 10000 });
  await new Promise(r => setTimeout(r, 600));

  const outDir = path.resolve(__dirname, '../outputs');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, 'marker_room_facing_verification.png');

  await page.screenshot({ path: outPath });
  console.log('Saved screenshot to:', outPath);

  await browser.close();
}

capture().catch(err => {
  console.error('Error during capture:', err);
  process.exit(1);
});
