const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

async function main() {
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
      '--window-size=1600,1000'
    ]
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    console.log('Navigating to http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await new Promise(r => setTimeout(r, 2000));

    const uploadInput = await page.$('input[type="file"]');
    if (!uploadInput) {
      throw new Error('File upload input not found on page');
    }

    const testImagePath = path.resolve(__dirname, '../outputs/user_blueprint_cropped.png');
    console.log('Uploading blueprint image:', testImagePath);
    await uploadInput.uploadFile(testImagePath);

    console.log('Waiting for room polygon vectorization & 3D extrusion...');
    await page.waitForSelector('.room-polygon', { timeout: 35000 });
    await new Promise(r => setTimeout(r, 4000));

    const outPath = path.resolve(__dirname, '../outputs/user_blueprint_verified_3d.png');
    await page.screenshot({ path: outPath });
    console.log('Screenshot successfully saved to:', outPath);

    // Also copy to brain artifacts for display
    const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";
    fs.copyFileSync(outPath, path.join(artDir, 'user_blueprint_verified_3d.png'));
    console.log('Artifact copied successfully!');
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error('Error during test:', err);
  process.exit(1);
});
