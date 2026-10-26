const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function testFullFlow() {
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

  const outDir = path.resolve(__dirname, '../outputs/design_stage3/flow_steps');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const docsDir = path.resolve(__dirname, '../docs/design_stage3_renders');
  if (!fs.existsSync(docsDir)) fs.mkdirSync(docsDir, { recursive: true });
  const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";

  function saveShot(fileName) {
    const src = path.join(outDir, fileName);
    fs.copyFileSync(src, path.join(docsDir, fileName));
    fs.copyFileSync(src, path.join(artDir, fileName));
    console.log(`Saved screenshot: ${fileName}`);
  }

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    console.log('Navigating to http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await new Promise(r => setTimeout(r, 2000));

    // ==========================================
    // STEP 1: UPLOAD FLOORPLAN
    // ==========================================
    console.log('\n--- STEP 1: Upload Floorplan ---');
    const uploadInput = await page.$('input[type="file"]');
    const samplePath = path.resolve(__dirname, '../frontend/public/sample_floorplan.png');
    if (uploadInput) {
      console.log('Uploading floorplan image:', samplePath);
      await uploadInput.uploadFile(samplePath);
      // Wait for vectorization & extrusion
      await page.waitForSelector('.room-polygon', { timeout: 35000 });
      await new Promise(r => setTimeout(r, 2000));
    }
    const step1Path = path.join(outDir, 'step1_upload_floorplan.png');
    await page.screenshot({ path: step1Path });
    saveShot('step1_upload_floorplan.png');

    // ==========================================
    // STEP 2: CLICK ROOM (Selects room & opens panel)
    // ==========================================
    console.log('\n--- STEP 2: Click Room (Kitchen / Room 5) ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys.length > 5) {
        polys[5].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      } else if (polys.length > 0) {
        polys[0].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      }
    });
    await new Promise(r => setTimeout(r, 1200));

    const step2Path = path.join(outDir, 'step2_click_room_panel.png');
    await page.screenshot({ path: step2Path });
    saveShot('step2_click_room_panel.png');

    // ==========================================
    // STEP 3: GENERATE DESIGN (Furnishes room with 3D GLBs)
    // ==========================================
    console.log('\n--- STEP 3: Click Generate Design ---');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    // Wait for API call & GLB loading
    await new Promise(r => setTimeout(r, 5000));

    const step3Path = path.join(outDir, 'step3_generate_design_furnished.png');
    await page.screenshot({ path: step3Path });
    saveShot('step3_generate_design_furnished.png');

    // ==========================================
    // STEP 4: REGENERATE (Alters seed variation)
    // ==========================================
    console.log('\n--- STEP 4: Click Regenerate ---');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const regenBtn = btns.find(b => b.textContent.includes('Regenerate'));
      if (regenBtn) regenBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 4000));

    const step4Path = path.join(outDir, 'step4_regenerate_design_seed.png');
    await page.screenshot({ path: step4Path });
    saveShot('step4_regenerate_design_seed.png');

    // ==========================================
    // STEP 5: CLEAR (Removes models & frees GLB WebGL memory)
    // ==========================================
    console.log('\n--- STEP 5: Click Clear ---');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const clearBtn = btns.find(b => b.textContent.includes('Clear'));
      if (clearBtn) clearBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 1500));

    const step5Path = path.join(outDir, 'step5_clear_design_freed.png');
    await page.screenshot({ path: step5Path });
    saveShot('step5_clear_design_freed.png');

    // ==========================================
    // STEP 6: SCALE UNCALIBRATED WARNING TEST
    // Click compact Bedroom (Room 1) to verify Scale Warning next to TOO_SMALL
    // ==========================================
    console.log('\n--- STEP 6: Scale Uncalibrated Warning Badge Verification ---');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys.length > 1) {
        polys[1].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      }
    });
    await new Promise(r => setTimeout(r, 800));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 3000));

    const step6Path = path.join(outDir, 'step6_scale_uncalibrated_warning.png');
    await page.screenshot({ path: step6Path });
    saveShot('step6_scale_uncalibrated_warning.png');

    console.log('\nALL 6 STEPS COMPLETED AND SCREENSHOTS SAVED!');
  } finally {
    await browser.close();
  }
}

testFullFlow().catch(err => {
  console.error('Error during full flow test:', err);
  process.exit(1);
});
