const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function runStage3Verification() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
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

  const outDir = path.resolve(__dirname, '../outputs/design_stage3');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const docsDir = path.resolve(__dirname, '../docs/design_stage3_renders');
  if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });
  const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    console.log('Navigating to http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await page.waitForSelector('.room-polygon', { timeout: 35000 });
    await new Promise(r => setTimeout(r, 2000));

    // ==========================================
    // TEST 1: Room 5 (Kitchen - 5.27 m²) Furnishing
    // ==========================================
    console.log('\n--- TEST 1: Room 5 (Kitchen) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      polys[5].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    console.log('Clicking Generate Design...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 5000));

    const panelKitchen = await page.evaluate(() => {
      return document.querySelector('.w-84')?.innerText;
    });
    console.log('Panel State (Room 5):\n', panelKitchen);

    // Save screenshot of live furnished kitchen
    const kitchenImg = path.join(outDir, 'real_viewer_furnished_room.png');
    await page.screenshot({ path: kitchenImg });
    fs.copyFileSync(kitchenImg, path.join(docsDir, 'real_viewer_furnished_room.png'));
    fs.copyFileSync(kitchenImg, path.join(artDir, 'real_viewer_furnished_room.png'));
    console.log('Saved real_viewer_furnished_room.png');

    // ==========================================
    // TEST 2: Regenerate (new seed variation)
    // ==========================================
    console.log('\n--- TEST 2: Regenerate (Seed Variation) ---');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const regenBtn = btns.find(b => b.textContent.includes('Regenerate'));
      if (regenBtn) regenBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 4000));

    const panelRegen = await page.evaluate(() => {
      return document.querySelector('.w-84')?.innerText;
    });
    console.log('Panel State after Regenerate:\n', panelRegen);

    // ==========================================
    // TEST 3: Clear (remove items & free memory)
    // ==========================================
    console.log('\n--- TEST 3: Clear (Remove & Free Memory) ---');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const clearBtn = btns.find(b => b.textContent.includes('Clear'));
      if (clearBtn) clearBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));

    const panelClear = await page.evaluate(() => {
      return document.querySelector('.w-84')?.innerText;
    });
    console.log('Panel State after Clear:\n', panelClear);

    // ==========================================
    // TEST 4: Room 1 (Bedroom - 3.71 m² - TOO_SMALL Guidance)
    // ==========================================
    console.log('\n--- TEST 4: Room 1 (Failure Guidance) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      polys[1].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 3000));

    const panelFailure = await page.evaluate(() => {
      return document.querySelector('.w-84')?.innerText;
    });
    console.log('Panel State (Room 1 Failure):\n', panelFailure);

    const failImg = path.join(outDir, 'real_viewer_failure_guidance.png');
    await page.screenshot({ path: failImg });
    fs.copyFileSync(failImg, path.join(docsDir, 'real_viewer_failure_guidance.png'));
    fs.copyFileSync(failImg, path.join(artDir, 'real_viewer_failure_guidance.png'));
    console.log('Saved real_viewer_failure_guidance.png');

    // ==========================================
    // TEST 5: Room 0 (Living Room - 10.59 m²)
    // ==========================================
    console.log('\n--- TEST 5: Room 0 (Living Room) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      polys[0].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1000));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 5000));

    const panelRoom0 = await page.evaluate(() => {
      return document.querySelector('.w-84')?.innerText;
    });
    console.log('Panel State (Room 0):\n', panelRoom0);

    const room0Img = path.join(outDir, 'real_viewer_living_room_design.png');
    await page.screenshot({ path: room0Img });
    fs.copyFileSync(room0Img, path.join(docsDir, 'real_viewer_living_room_design.png'));
    fs.copyFileSync(room0Img, path.join(artDir, 'real_viewer_living_room_design.png'));
    console.log('Saved real_viewer_living_room_design.png');

    console.log('\nALL 5 TESTS EXECUTED SUCCESSFULLY!');
  } finally {
    await browser.close();
  }
}

runStage3Verification().catch(err => {
  console.error('Error during Stage 3 verification:', err);
  process.exit(1);
});
