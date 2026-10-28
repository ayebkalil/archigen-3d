const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));

async function captureStudyPage() {
  const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const browser = await puppeteer.launch({
    executablePath: edgePath,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--allow-file-access-from-files',
      '--window-size=1600,1050'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1050, deviceScaleFactor: 1 });

  const htmlPath = path.resolve(__dirname, '../outputs/blind_study/index.html');
  const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');
  console.log('Loading study page from:', fileUrl);

  await page.goto(fileUrl, { waitUntil: 'load', timeout: 20000 });
  await new Promise(r => setTimeout(r, 1000));

  const outDir = path.resolve(__dirname, '../outputs/blind_study');
  const artDir = "C:\\Users\\ayebk\\.gemini\\antigravity\\brain\\387c7b87-863d-4c00-831b-93abdc64d4d4";

  // 1. Capture Main Study Overview
  const shot1 = path.join(outDir, 'study_page_overview.png');
  await page.screenshot({ path: shot1 });
  fs.copyFileSync(shot1, path.join(artDir, 'study_page_overview.png'));
  console.log('Captured study_page_overview.png');

  // 2. Open Instructions Modal and capture
  await page.evaluate(() => {
    document.getElementById('btn-instructions').click();
  });
  await new Promise(r => setTimeout(r, 500));

  const shot2 = path.join(outDir, 'study_page_instructions_modal.png');
  await page.screenshot({ path: shot2 });
  fs.copyFileSync(shot2, path.join(artDir, 'study_page_instructions_modal.png'));
  console.log('Captured study_page_instructions_modal.png');

  await browser.close();
  console.log('All study page screenshots captured and saved to artifacts.');
}

captureStudyPage().catch(err => {
  console.error('Error capturing screenshots:', err);
  process.exit(1);
});
