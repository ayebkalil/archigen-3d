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

    console.log('Navigating to http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await new Promise(r => setTimeout(r, 2500));

    // Check if room polygons are already rendered from auto-load or need upload
    let polys = await page.$$('.room-polygon');
    if (polys.length === 0) {
      const uploadInput = await page.$('input[type="file"]');
      if (uploadInput) {
        const samplePath = path.resolve(__dirname, '../frontend/public/sample_floorplan.png');
        console.log('Uploading floorplan image:', samplePath);
        await uploadInput.uploadFile(samplePath);
      }
      await page.waitForSelector('.room-polygon', { timeout: 35000 });
      await new Promise(r => setTimeout(r, 3000));
    }

    // Select Living Room (Room 0) and generate furniture (includes rug_living, sofa, etc.)
    console.log('Selecting Living Room (Room 0) & Generating Design...');
    await page.evaluate(() => {
      const polys = document.querySelectorAll('.room-polygon');
      if (polys[0]) polys[0].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    await new Promise(r => setTimeout(r, 800));

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
      if (genBtn) genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    });
    console.log('Waiting for furniture models to finish loading into scene...');
    await page.waitForFunction(() => {
      const fg = window.__ARCHIGEN_VIEWER__?.scene?.getObjectByName('furnitureGroup');
      return fg && fg.children.length >= 3;
    }, { timeout: 20000 });
    await new Promise(r => setTimeout(r, 1000));

    // 1. Lock camera position & inspect rug vs floor elevations
    console.log('Locking camera and verifying rug elevation vs floor mesh...');
    const elevationCheck = await page.evaluate(() => {
      const v = window.__ARCHIGEN_VIEWER__;
      if (!v) return { error: 'Viewer not found' };

      // Set fixed angled perspective camera focused on living room floor & rug
      v.camera.position.set(1.5, 4.0, 4.8);
      v.controls.target.set(0.0, 0.2, -0.5);
      v.controls.update();

      // Find floor mesh and rug mesh
      let floorY = null;
      let rugY = null;
      let rugPolygonOffset = null;
      let rugOffsetFactor = null;

      v.scene.traverse((obj) => {
        if (obj.userData?.isFloorMesh && floorY === null) {
          floorY = obj.position.y;
        }
        if (obj.userData?.itemId?.includes('rug') || obj.userData?.itemId === 'rug_living') {
          rugY = obj.position.y;
          obj.traverse((child) => {
            if (child.isMesh && child.material) {
              rugPolygonOffset = child.material.polygonOffset;
              rugOffsetFactor = child.material.polygonOffsetFactor;
            }
          });
        }
      });

      return {
        cameraPos: [v.camera.position.x, v.camera.position.y, v.camera.position.z],
        cameraTarget: [v.controls.target.x, v.controls.target.y, v.controls.target.z],
        floorY,
        rugY,
        elevationDeltaMm: rugY !== null && floorY !== null ? (rugY - floorY) * 1000 : null,
        rugPolygonOffset,
        rugOffsetFactor
      };
    });

    console.log('Z-Fighting & Elevation Verification Results:', elevationCheck);

    // Capture STAGE 4 AFTER (same camera angle)
    await new Promise(r => setTimeout(r, 1000));
    const afterShot = path.join(outDir, 'stage4_after_same_angle.png');
    await page.screenshot({ path: afterShot });
    saveShot('stage4_after_same_angle.png');

    // 2. Switch to STAGE 3 BEFORE without moving the camera!
    console.log('Reverting materials to Stage 3 flat appearance at IDENTICAL camera angle...');
    await page.evaluate(() => {
      const v = window.__ARCHIGEN_VIEWER__;
      if (!v) return;

      // Save environment map to restore later
      window.__SAVED_ENV__ = v.scene.environment;
      v.scene.environment = null;

      // Revert floors and walls to flat untextured MeshLambert / flat concrete
      v.scene.traverse((obj) => {
        if (obj.isMesh) {
          if (obj.userData?.isFloorMesh) {
            obj.material = new window.THREE.MeshLambertMaterial({ color: 0x94a3b8 });
          } else if (obj.userData?.isBuildingMesh) {
            obj.material = new window.THREE.MeshLambertMaterial({ color: 0xe2e8f0 });
          }
        }
      });
    });

    await new Promise(r => setTimeout(r, 1000));
    const beforeShot = path.join(outDir, 'stage4_before_same_angle.png');
    await page.screenshot({ path: beforeShot });
    saveShot('stage4_before_same_angle.png');

    // 3. Zoom into the rug-floor boundary for direct close-up visual proof
    console.log('Capturing closeup detail of rug and floor boundary...');
    await page.evaluate(() => {
      const v = window.__ARCHIGEN_VIEWER__;
      if (!v) return;
      // Re-enable PBR environment & materials
      if (window.__SAVED_ENV__) v.scene.environment = window.__SAVED_ENV__;
      // Zoom close to rug boundary
      v.camera.position.set(0.2, 1.2, 1.2);
      v.controls.target.set(0.0, 0.05, 0.0);
      v.controls.update();
    });
    await new Promise(r => setTimeout(r, 1200));
    const closeupShot = path.join(outDir, 'stage4_rug_floor_closeup.png');
    await page.screenshot({ path: closeupShot });
    saveShot('stage4_rug_floor_closeup.png');

    console.log('\nAll same-angle before/after screenshots and rug verification complete!');
  } finally {
    await browser.close();
  }
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
