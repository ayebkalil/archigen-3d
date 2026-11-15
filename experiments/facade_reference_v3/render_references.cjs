// Capture layout-aligned orthographic PBR renders using the actual house meshes.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
async function main() {
  const project = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
  const puppeteer = require(path.join(project, 'frontend/node_modules/puppeteer-core'));
  const executablePath = ['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe'].find(fs.existsSync);
  const browser = await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--use-angle=d3d11','--enable-webgl']});
  try {
    const page = await browser.newPage();
    await page.setViewport({width:1200,height:900});
    await page.goto('http://127.0.0.1:3001/tmp/facade-reference-v3/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.__FACADE_V3__ && window.__ARCHIGEN_VIEWER__?.scene.children.filter(o=>o.userData.type==='facade').length===15);
    await page.waitForFunction(()=>window.__FACADE_V3__.appearanceReady() && window.__ARCHIGEN_VIEWER__.pbrTexturesRef.current.loaded && window.__ARCHIGEN_VIEWER__.scene.children.filter(o=>o.userData.type==='facade').every(o=>o.userData.v3Appearance==='reference'));
    await new Promise(r=>setTimeout(r,1500));
    const ids=await page.evaluate(()=>window.__FACADE_V3__.panels.map(p=>p.wall_id));
    for(const id of ids) {
      const data = await page.evaluate(id=>{
        const {scene,renderer}=window.__ARCHIGEN_VIEWER__,T=window.THREE;
        const spec=window.__FACADE_V3__.panels.find(p=>p.wall_id===id);
        const skin=scene.children.find(o=>o.userData.type==='facade'&&o.userData.wallId===id);
        scene.updateMatrixWorld(true);
        const p=skin.geometry.attributes.position;
        const origin=new T.Vector3(p.getX(0),0,p.getZ(0));skin.localToWorld(origin);
        const frame=spec.facade_frame,b=spec.uv_bounds;
        const width=spec.wall_length_m/(b.u_max-b.u_min),height=spec.wall_height_m/(b.v_max-b.v_min);
        const center=origin.clone().add(new T.Vector3(frame.direction[0],0,frame.direction[1]).multiplyScalar(spec.wall_length_m/2+(.5-(b.u_min+b.u_max)/2)*width));
        center.y+=spec.wall_height_m/2+(.5-(b.v_min+b.v_max)/2)*height;
        const camera=new T.OrthographicCamera(-width/2,width/2,height/2,-height/2,.01,100);
        camera.position.copy(center).add(new T.Vector3(frame.outward[0],0,frame.outward[1]).multiplyScalar(10));camera.lookAt(center);
        const visibility=scene.children.map(o=>[o,o.visible]);
        for(const object of scene.children) {
          if(object.isLight)continue;
          object.visible=object===skin || object.userData.facadeOpening?.associated_wall===id;
        }
        // A wall texture is reflectance, not a photograph containing directional light.
        // Keep real frame/leaf meshes, but capture wall albedo without baked shadows.
        const previousMaterial=skin.material;
        skin.material=new T.MeshBasicMaterial({map:previousMaterial.map,color:0xffffff,toneMapped:false});
        const previousBackground=scene.background;scene.background=new T.Color(0x7f7f7f);
        const oldSize=renderer.getSize(new T.Vector2()),oldRatio=renderer.getPixelRatio();
        renderer.setPixelRatio(1);renderer.setSize(512,512,false);renderer.render(scene,camera);
        const data=renderer.domElement.toDataURL('image/png');
        skin.material.dispose();skin.material=previousMaterial;
        scene.background=previousBackground;for(const [object,visible] of visibility)object.visible=visible;
        renderer.setPixelRatio(oldRatio);renderer.setSize(oldSize.x,oldSize.y,false);
        return data;
      },id);
      fs.writeFileSync(path.join(output,id,'reference.png'),Buffer.from(data.split(',')[1],'base64'));
      console.log('Captured PBR reference',id);
    }
    assert.equal(ids.length,15);
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exit(1)});
