const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

async function captureScreenshots() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,900']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const study1Path = path.resolve(__dirname, '../outputs/blind_study/shared_bundles/study_1/index.html');
  const study2Path = path.resolve(__dirname, '../outputs/blind_study/shared_bundles/study_2/index.html');

  // 1. Study 1 Modal
  await page.goto(`file://${study1Path}`);
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.resolve(__dirname, '../outputs/blind_study/study_1_modal.png') });
  console.log('Saved study_1_modal.png');

  // 2. Study 1 Main comparison view
  await page.evaluate(() => {
    document.getElementById('raterIdInput').value = 'Rater_Architect_1';
    window.startStudy();
  });
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.resolve(__dirname, '../outputs/blind_study/study_1_comparison.png') });
  console.log('Saved study_1_comparison.png');

  // 3. Study 2 Main comparison view
  await page.goto(`file://${study2Path}`);
  await new Promise(r => setTimeout(r, 600));
  await page.evaluate(() => {
    document.getElementById('raterIdInput').value = 'Rater_Architect_2';
    window.startStudy();
  });
  await new Promise(r => setTimeout(r, 600));
  await page.screenshot({ path: path.resolve(__dirname, '../outputs/blind_study/study_2_comparison.png') });
  console.log('Saved study_2_comparison.png');

  await browser.close();
  console.log('All screenshots captured successfully!');
}

captureScreenshots().catch(err => {
  console.error(err);
  process.exit(1);
});
