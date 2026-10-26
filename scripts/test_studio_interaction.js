const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../frontend/node_modules/puppeteer-core'));
const fs = require('fs');

async function testStudio() {
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

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

    await page.goto('http://localhost:3000/#studio', { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForSelector('.room-polygon', { timeout: 35000 });

    const roomData = await page.evaluate(() => {
      const polys = Array.from(document.querySelectorAll('.room-polygon'));
      return polys.map((p, idx) => ({
        index: idx,
        id: p.getAttribute('data-room-id')
      }));
    });
    console.log('Found rooms:', roomData.length);

    let furnishableIndex = -1;
    let tooSmallIndex = -1;

    for (let i = 0; i < roomData.length; i++) {
      await page.evaluate((idx) => {
        const polys = document.querySelectorAll('.room-polygon');
        polys[idx].dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      }, i);
      await new Promise(r => setTimeout(r, 400));

      const roomText = await page.evaluate(() => {
        const p = document.querySelector('.w-84');
        return p ? p.innerText : '';
      });

      console.log(`--- ROOM ${i} (${roomData[i].id}) ---`);
      const lines = roomText.split('\n').filter(l => l.trim().length > 0);
      console.log(lines.slice(0, 5).join(' | '));

      // Click generate design to test outcome
      const genSuccess = await page.evaluate(() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const genBtn = btns.find(b => b.textContent.includes('Generate Design'));
        if (genBtn) {
          genBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
          return true;
        }
        return false;
      });

      await new Promise(r => setTimeout(r, 2000));

      const updatedText = await page.evaluate(() => {
        const p = document.querySelector('.w-84');
        return p ? p.innerText : '';
      });

      if (updatedText.includes('Layout Verified') || updatedText.includes('COLLISION-FREE')) {
        console.log(`-> SUCCESSFUL FURNISHING FOUND AT ROOM ${i}!`);
        if (furnishableIndex === -1) furnishableIndex = i;
      } else if (updatedText.includes('Room Space Too Compact') || updatedText.includes('No Usable Backing Wall')) {
        console.log(`-> DIAGNOSTIC REASON AT ROOM ${i}:`, updatedText.includes('Room Space Too Compact') ? 'TOO_SMALL' : 'NO_USABLE_WALL');
        if (tooSmallIndex === -1) tooSmallIndex = i;
      }
    }

    console.log(`Summary: first furnishable room is ${furnishableIndex}, first diagnosed failure room is ${tooSmallIndex}`);
  } finally {
    await browser.close();
  }
}

testStudio().catch(console.error);
