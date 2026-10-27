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

  const outDir = path.resolve(__dirname, '../outputs/design_stage4');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";

  function saveShot(fileName) {
    const src = path.join(outDir, fileName);
    fs.copyFileSync(src, path.join(artDir, fileName));
    console.log(`Saved screenshot: ${fileName}`);
  }

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    console.log('Navigating to Docker app: http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await new Promise(r => setTimeout(r, 2000));

    // Upload floorplan
    const uploadInput = await page.$('input[type="file"]');
    const samplePath = path.resolve(__dirname, '../frontend/public/sample_floorplan.png');
    console.log('Uploading sample floorplan to Docker stack:', samplePath);
    await uploadInput.uploadFile(samplePath);

    console.log('Waiting for Docker vectorization & 3D extrusion...');
    await page.waitForSelector('.room-polygon', { timeout: 35000 });
    await new Promise(r => setTimeout(r, 3000));

    // -------------------------------------------------------------
    // 1. BEDROOM FURNISHING (Index 1: Bedroom 8.86 m2)
    // -------------------------------------------------------------
    console.log('\n--- 1. Testing Bedroom (Index 1) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys[1]) polys[1].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 4500));
    await page.screenshot({ path: path.join(outDir, 'docker_flow_bedroom.png') });
    saveShot('docker_flow_bedroom.png');

    // -------------------------------------------------------------
    // 2. LIVING ROOM FURNISHING (Index 0: Living Room 36.14 m2)
    // -------------------------------------------------------------
    console.log('\n--- 2. Testing Living Room (Index 0) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys[0]) polys[0].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 5500));
    await page.screenshot({ path: path.join(outDir, 'docker_flow_living_room.png') });
    saveShot('docker_flow_living_room.png');

    // -------------------------------------------------------------
    // 3. BATHROOM FURNISHING (Index 5: Bathroom 4.65 m2)
    // -------------------------------------------------------------
    console.log('\n--- 3. Testing Bathroom (Index 5) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys[5]) polys[5].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 4500));
    await page.screenshot({ path: path.join(outDir, 'docker_flow_bathroom.png') });
    saveShot('docker_flow_bathroom.png');

    // -------------------------------------------------------------
    // 4. FRAGMENT ROOM VERIFICATION (Index 12: Storage / Fragment 1.48 m2)
    // -------------------------------------------------------------
    console.log('\n--- 4. Testing Fragment Room (Index 12: 1.48 m2) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys[12]) polys[12].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));
    await page.screenshot({ path: path.join(outDir, 'docker_flow_fragment_room.png') });
    saveShot('docker_flow_fragment_room.png');

    console.log('\nALL EXTENDED DOCKER FLOW CHECKS COMPLETED!');
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error('Error during extended docker flow:', err);
  process.exit(1);
});
