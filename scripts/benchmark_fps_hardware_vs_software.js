const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

async function runBenchmark(mode = 'gpu') {
  const isGpu = mode === 'gpu';
  const args = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--window-size=1600,1000'
  ];

  if (isGpu) {
    args.push('--use-gl=angle', '--use-angle=d3d11', '--enable-gpu-rasterization', '--enable-webgl');
  } else {
    args.push('--use-gl=angle', '--use-angle=swiftshader', '--disable-gpu');
  }

  console.log(`\n======================================================`);
  console.log(`Starting Benchmark: Mode = ${mode.toUpperCase()}`);
  console.log(`Args:`, args.filter(a => a.includes('gl') || a.includes('gpu')));

  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    console.log('Navigating to http://localhost:3000/#studio ...');
    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 35000 });
    await new Promise(r => setTimeout(r, 2000));

    // Upload the larger floorplan (user blueprint: 16 rooms, 24 walls, 159.9 m2)
    const uploadInput = await page.$('input[type="file"]');
    const blueprintPath = path.resolve(__dirname, '../outputs/user_blueprint_cropped.png');
    console.log('Uploading large plan (16 rooms, 24 walls):', blueprintPath);
    await uploadInput.uploadFile(blueprintPath);

    await page.waitForSelector('.room-polygon', { timeout: 35000 });
    await new Promise(r => setTimeout(r, 3000));

    // Furnish Living Room (Room 0)
    console.log('Selecting and furnishing Living Room (Room 0)...');
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
    // Wait for GLBs to load
    await new Promise(r => setTimeout(r, 5000));

    // Continuous rendering measurement over 3 seconds
    console.log('Measuring frame rates across 3000ms continuous rendering...');
    const result = await page.evaluate(async (durationMs) => {
      return new Promise((resolve) => {
        let frameCount = 0;
        let startTime = performance.now();
        let lastTime = startTime;
        const frameTimes = [];

        function onFrame(now) {
          frameCount++;
          frameTimes.push(now - lastTime);
          lastTime = now;
          if (now - startTime < durationMs) {
            requestAnimationFrame(onFrame);
          } else {
            const totalDuration = now - startTime;
            const avgFps = (frameCount / totalDuration) * 1000;
            const avgFrameTimeMs = totalDuration / frameCount;
            frameTimes.sort((a, b) => a - b);
            const p95FrameTimeMs = frameTimes[Math.floor(frameTimes.length * 0.95)] || avgFrameTimeMs;
            resolve({
              frameCount,
              totalDurationMs: Math.round(totalDuration),
              avgFps: Math.round(avgFps * 10) / 10,
              avgFrameTimeMs: Math.round(avgFrameTimeMs * 100) / 100,
              p95FrameTimeMs: Math.round(p95FrameTimeMs * 100) / 100
            });
          }
        }
        requestAnimationFrame(onFrame);
      });
    }, 3000);

    console.log(`Benchmark Result (${mode.toUpperCase()}):`);
    console.log(`  - Average Framerate:  ${result.avgFps} FPS`);
    console.log(`  - Average Frame Time: ${result.avgFrameTimeMs} ms`);
    console.log(`  - P95 Frame Time:     ${result.p95FrameTimeMs} ms`);
    console.log(`  - Total Frames:       ${result.frameCount} in ${result.totalDurationMs} ms`);
    return result;
  } finally {
    await browser.close();
  }
}

async function runAll() {
  const gpuResults = await runBenchmark('gpu');
  const cpuResults = await runBenchmark('cpu_swiftshader');

  console.log('\n======================================================');
  console.log('FINAL PERFORMANCE BENCHMARK COMPARISON:');
  console.log(`Hardware GPU (NVIDIA RTX 3050 Laptop D3D11): ${gpuResults.avgFps} FPS (${gpuResults.avgFrameTimeMs} ms/frame)`);
  console.log(`Software Render (CPU SwiftShader WebGL):       ${cpuResults.avgFps} FPS (${cpuResults.avgFrameTimeMs} ms/frame)`);
  console.log('======================================================\n');
}

runAll().catch(e => {
  console.error(e);
  process.exit(1);
});
