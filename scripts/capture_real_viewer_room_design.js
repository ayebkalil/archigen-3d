const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function captureRealViewer() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  console.log('Launching browser with Edge...');

  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--window-size=1600,1000'
    ]
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    const url = 'http://localhost:3000/#studio';
    console.log('Navigating to:', url);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

    console.log('Waiting for .room-polygon...');
    await page.waitForSelector('.room-polygon', { timeout: 35000 });
    console.log('.room-polygon appeared!');

    await new Promise(r => setTimeout(r, 1500));

    // Click on Master Bedroom polygon
    console.log('Clicking on Master Bedroom polygon...');
    const clickRoom = await page.evaluate(() => {
      const polys = Array.from(document.querySelectorAll('.room-polygon'));
      const target = polys.length > 1 ? polys[1] : polys[0];
      target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      return target.getAttribute('data-room-id');
    });
    console.log('Clicked room id:', clickRoom);

    // Wait for panel to open
    await new Promise(r => setTimeout(r, 1200));

    // Log all buttons
    const buttons = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('button')).map(b => b.textContent.trim());
    });
    console.log('Buttons visible:', buttons);

    // Click Generate Design
    console.log('Clicking Generate Design button...');
    const clicked = await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) {
        genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        return true;
      }
      return false;
    });
    console.log('Clicked result:', clicked);

    // Wait 4 seconds for API request and GLB render
    console.log('Waiting 4s for network request and 3D load...');
    await new Promise(r => setTimeout(r, 4500));

    const panelState = await page.evaluate(() => {
      const panel = document.querySelector('.w-84');
      return panel ? panel.innerText : 'NO PANEL';
    });
    console.log('Panel State:\n', panelState);

    const outDir = path.resolve(__dirname, '../outputs/design_stage3');
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, 'real_viewer_furnished_room.png');

    await page.screenshot({ path: outPath });
    console.log('Saved real viewer screenshot to:', outPath);

    // Copy to docs and artifact directory
    const docsDir = path.resolve(__dirname, '../docs/design_stage3_renders');
    if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });
    fs.copyFileSync(outPath, path.join(docsDir, 'real_viewer_furnished_room.png'));

    const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";
    fs.copyFileSync(outPath, path.join(artDir, 'real_viewer_furnished_room.png'));
    console.log('Copied screenshot to docs and artifact directory.');

  } finally {
    await browser.close();
  }
}

captureRealViewer().catch(err => {
  console.error('Error during real viewer capture:', err);
  process.exit(1);
});
