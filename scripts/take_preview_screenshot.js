/**
 * scripts/take_preview_screenshot.js
 * Uses puppeteer-core with local Edge/Chrome to capture the Three.js preview page
 * once all models are fully loaded and rendered.
 */

const puppeteer = require('../frontend/node_modules/puppeteer-core');
const fs = require('fs');
const path = require('path');

const OUTPUT_PATH = 'C:/Users/ayebk/.gemini/antigravity/brain/387c7b87-863d-4c00-831b-93abdc64d4d4/furniture_bank_preview.png';

async function capture() {
  const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const executablePath = fs.existsSync(edgePath) ? edgePath : chromePath;

  console.log(`Using browser executable: ${executablePath}`);

  const browser = await puppeteer.launch({
    executablePath,
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--window-size=1920,1080'
    ]
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err));

  console.log('Navigating to http://localhost:8099/preview_furniture.html...');
  await page.goto('http://localhost:8099/preview_furniture.html', { waitUntil: 'networkidle0', timeout: 30000 });

  console.log('Waiting for window.__PREVIEW_READY__ === true...');
  await page.waitForFunction('window.__PREVIEW_READY__ === true', { timeout: 30000 });

  // Brief pause for render frame
  await new Promise(r => setTimeout(r, 1200));

  console.log(`Saving screenshot to ${OUTPUT_PATH}...`);
  await page.screenshot({ path: OUTPUT_PATH, type: 'png' });

  await browser.close();
  console.log(`[SUCCESS] Screenshot captured (${fs.statSync(OUTPUT_PATH).size} bytes)`);
}

capture().catch(err => {
  console.error('[ERROR]', err);
  process.exit(1);
});
