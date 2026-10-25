/**
 * scripts/take_l_shape_screenshot.js
 * Captures 3D top view of L-shaped floorplan and combines with 2D plan side-by-side.
 */

const puppeteer = require('../frontend/node_modules/puppeteer-core');
const fs = require('fs');
const path = require('path');
const http = require('http');

const PORT = 8097;
const BASE_DIR = path.resolve(__dirname, '../frontend/public');

// 1. Static file server
const server = http.createServer((req, res) => {
  let filePath = path.join(BASE_DIR, req.url === '/' ? 'test_l_shape_top.html' : req.url);
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    res.writeHead(200, { 'Content-Type': filePath.endsWith('.html') ? 'text/html' : 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(PORT, async () => {
  console.log(`Server listening on port ${PORT}`);
  const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const executablePath = fs.existsSync(edgePath) ? edgePath : chromePath;

  const browser = await puppeteer.launch({
    executablePath,
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--window-size=550,550']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 550, height: 550 });

  await page.goto(`http://localhost:${PORT}/test_l_shape_top.html`, { waitUntil: 'networkidle0', timeout: 20000 });
  await page.waitForFunction('window.__L_READY__ === true', { timeout: 20000 });
  await new Promise(r => setTimeout(r, 600));

  const outPath3d = path.resolve(__dirname, '../outputs/l_shape_3d_top.png');
  await page.screenshot({ path: outPath3d, type: 'png' });
  console.log(`Saved 3D top view to ${outPath3d}`);

  await browser.close();
  server.close();
  console.log('Capture completed.');
});
