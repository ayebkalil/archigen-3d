/** Actual studio -> native API -> GLB smoke check. Does not save to the user catalog.
 * Requires Vite :3000 and an API with ARCHIGEN_FURNITURE_QUALITY=1.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');
async function main() {
  const output=path.resolve(process.argv[2]||'outputs/furniture_quality/studio');
  fs.mkdirSync(output,{recursive:true});
  const executablePath=process.env.CHROME_PATH||['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe'].find(fs.existsSync);
  const browser=await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const errors=[];
  try {
    const page=await browser.newPage();await page.setViewport({width:1600,height:1100});
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error'&&/GLB|WebGL|shader/i.test(m.text()))errors.push(m.text());});
    page.on('dialog',async d=>{errors.push(d.message());await d.dismiss();});
    await page.goto('http://127.0.0.1:3000/#studio',{waitUntil:'networkidle0'});
    const clickText=async text=>{
      await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes(t)),{},text);
      await page.evaluate(t=>[...document.querySelectorAll('button')].find(b=>b.textContent.includes(t)).click(),text);
    };
    await clickText('AI Generative Studio');
    await clickText('Rotate: ON');
    const pending=page.waitForResponse(r=>r.url().includes('/furniture/generate-3d'),{timeout:600000});
    await page.click('[data-testid=generate-furniture]');
    const response=await pending;
    const result=await response.json();assert.equal(response.status(),200,JSON.stringify(result));
    assert.equal(result.quality.checkpoint_loading,'strict');
    assert.equal(result.quality.watertight,true);
    assert.ok(result.quality.peak_device_sampled_mib<=3500);
    fs.writeFileSync(path.join(output,'model.glb'),Buffer.from(result.model_glb_base64.split(',')[1],'base64'));
    delete result.model_glb_base64;delete result.preview_image_base64;
    await page.waitForFunction(()=>document.body.textContent.includes('neural atlas'),{timeout:30000});
    // Two frames let the asynchronous GLTFLoader upload the embedded texture.
    await page.evaluate(()=>new Promise(r=>setTimeout(r,2000)));
    await page.screenshot({path:path.join(output,'studio.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'result.json'),JSON.stringify({...result,browser_errors:errors},null,2));
    console.log(JSON.stringify(result));
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
