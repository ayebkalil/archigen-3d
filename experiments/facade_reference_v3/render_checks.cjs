/** Check production house geometry across every experimental appearance mode. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
async function main() {
  const project = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
  const puppeteer = require(path.join(project, 'frontend/node_modules/puppeteer-core'));
  const executablePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(fs.existsSync);
  const browser = await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--use-angle=d3d11','--enable-webgl']});
  const errors = [], checks = [];
  try {
    const page = await browser.newPage(); await page.setViewport({width:1600,height:1050});
    page.on('pageerror', error=>errors.push(error.message));
    page.on('console', message=>{if(message.type()==='error' && /THREE|WebGL|shader/i.test(message.text())) errors.push(message.text());});
    await page.goto('http://127.0.0.1:3001/tmp/facade-reference-v3/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.__FACADE_V3__?.ready);
    for(const mode of ['v2','guided','locked','reference','guide']) {
      await page.click(`[data-mode="${mode}"]`);
      const file={v2:'v2',guided:'guided',locked:'locked',reference:'base',guide:'sketch'}[mode];
      await page.waitForFunction((mode,file)=>{
        if(window.__FACADE_V3__?.mode!==mode)return false;
        const skins=window.__ARCHIGEN_VIEWER__?.scene.children.filter(o=>o.userData.type==='facade')||[];
        return skins.length===15 && skins.every(o=>o.material.map?.image?.complete && o.material.map.image.src.endsWith(`/${file}.png`)) && window.__FACADE_V3__.appearanceReady();
      },{timeout:90000},mode,file);
      await new Promise(r=>setTimeout(r,500));
      const result=await page.evaluate(()=>{
        const {scene}=window.__ARCHIGEN_VIEWER__,{panels,mode}=window.__FACADE_V3__;
        return {mode,walls:panels.map(panel=>{
          const skin=scene.children.find(o=>o.userData.type==='facade'&&o.userData.wallId===panel.wall_id);
          const p=skin.geometry.attributes.position,uv=skin.geometry.attributes.uv,b=skin.userData.uvBounds,f=panel.facade_frame;
          let uvError=0;
          for(let i=0;i<p.count;i++) {
            const along=(p.getX(i)-p.getX(0))*f.direction[0]+(p.getZ(i)-p.getZ(0))*f.direction[1];
            uvError=Math.max(uvError,Math.abs(uv.getX(i)-(b.u_min+(b.u_max-b.u_min)*along/f.length_m)),
              Math.abs(uv.getY(i)-(b.v_min+(b.v_max-b.v_min)*p.getY(i)/panel.wall_height_m)));
          }
          const openings=scene.children.filter(o=>o.userData.facadeOpening?.associated_wall===panel.wall_id).map(o=>{
            const spec=o.userData.facadeOpening;
            return {id:spec.id,position:o.position.toArray(),width:spec.width_m,height:spec.height_m};
          }).sort((a,b)=>a.id.localeCompare(b.id));
          let neuralMeshes=0,nonfiniteOpeningUVs=0;
          scene.traverse(o=>{if(o.userData.v3NeuralOpening) {
            if(o.material.map?.image?.complete)neuralMeshes++;
            nonfiniteOpeningUVs+=Array.from(o.geometry.attributes.uv.array).filter(v=>!Number.isFinite(v)).length;
          }});
          return {wallId:panel.wall_id,expectedOpenings:panel.openings.length,openings,uvError,
            textureSize:[skin.material.map.image.width,skin.material.map.image.height],pbr:!!skin.material.normalMap,neuralMeshes,nonfiniteOpeningUVs};
        }),unresolved:scene.children.filter(o=>o.userData.type==='wall_assembly').flatMap(o=>o.userData.unresolvedOpenings)};
      });
      assert.deepEqual(result.unresolved,[]);
      for(const wall of result.walls) {
        assert.equal(wall.openings.length,wall.expectedOpenings,wall.wallId);
        assert.ok(wall.uvError<1e-5,wall.wallId+' planar UV');
        if(['locked','reference'].includes(mode)) {assert.equal(wall.pbr,true);assert.deepEqual(wall.textureSize,[1024,1024]);}
        if(mode==='locked')assert.ok(wall.neuralMeshes>0);
        assert.equal(wall.nonfiniteOpeningUVs,0);
      }
      if(checks.length)assert.deepEqual(result.walls.map(w=>w.openings),checks[0].walls.map(w=>w.openings));
      checks.push(result);
      for(const id of ['ground_east','ground_south_extension','upper_south']) {
        await page.select('select[aria-label="Facade side"]',id);await page.click('[data-action="face"]');
        await new Promise(r=>setTimeout(r,500));await page.screenshot({path:path.join(output,`${mode}-${id}.png`)});
      }
      await page.select('select[aria-label="Visible building levels"]','all');await new Promise(r=>setTimeout(r,700));
      for(const [action,suffix] of [['overview','overview'],['reverse','opposite']]) {
        await page.click(`[data-action="${action}"]`);await new Promise(r=>setTimeout(r,500));
        await page.screenshot({path:path.join(output,`${mode}-house-${suffix}.png`)});
      }
    }
    await page.click('[data-mode="locked"]');await new Promise(r=>setTimeout(r,700));
    await page.click('input[aria-label="Neural opening detail"]');await new Promise(r=>setTimeout(r,700));
    assert.equal(await page.evaluate(()=>window.__FACADE_V3__.neural),false);
    assert.equal(await page.evaluate(()=>{let n=0;window.__ARCHIGEN_VIEWER__.scene.traverse(o=>{if(o.userData.v3NeuralOpening)n++;});return n;}),0);
    await page.click('[data-action="overview"]');await new Promise(r=>setTimeout(r,500));
    await page.screenshot({path:path.join(output,'locked-house-physical-openings.png')});
    assert.deepEqual(errors,[]);
    const report={modes:checks.length,panels:15,openings:checks[0].walls.reduce((n,w)=>n+w.openings.length,0),
      maxUvError:Math.max(...checks.flatMap(r=>r.walls.map(w=>w.uvError))),errors,checks};
    fs.writeFileSync(path.join(output,'viewer-checks.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({...report,checks:undefined}));
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exit(1)});
