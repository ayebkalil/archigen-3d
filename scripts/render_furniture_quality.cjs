/** Reproducible Three.js GLB turntable. Local files only, no GPU model loading.
 * node scripts/render_furniture_quality.cjs model.glb output-directory
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

async function main() {
  const model = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  fs.mkdirSync(output, { recursive: true });
  const three = path.resolve(__dirname, '../frontend/node_modules/three');
  const html = `<!doctype html><style>body{margin:0}</style>
  <script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}</script>
  <script type="module">
  import * as THREE from 'three';
  import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
  const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  renderer.setSize(600,600); renderer.setPixelRatio(1);
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1;
  document.body.appendChild(renderer.domElement);
  const scene=new THREE.Scene(); scene.background=new THREE.Color('#e8e8e5');
  scene.add(new THREE.HemisphereLight(0xffffff,0x667080,2));
  const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(3,5,4);scene.add(key);
  const fill=new THREE.DirectionalLight(0xffffff,1);fill.position.set(-3,2,-4);scene.add(fill);
  const gltf=await new GLTFLoader().loadAsync('/model.glb');scene.add(gltf.scene);
  const box=new THREE.Box3().setFromObject(gltf.scene),size=box.getSize(new THREE.Vector3());
  const target=box.getCenter(new THREE.Vector3());
  const half=Math.max(size.x,size.y,size.z)*.65;
  const camera=new THREE.OrthographicCamera(-half,half,half,-half,.01,100);
  const grid=new THREE.GridHelper(4,40,0xaaaaaa,0xcccccc);grid.position.y=-.002;scene.add(grid);
  window.renderView=(azimuth,elevation=12)=>{
    const a=azimuth*Math.PI/180,e=elevation*Math.PI/180;
    camera.position.copy(target).add(new THREE.Vector3(Math.sin(a)*Math.cos(e),Math.sin(e),Math.cos(a)*Math.cos(e)).multiplyScalar(3));
    camera.lookAt(target);renderer.render(scene,camera);
  };
  window.report={bounds:{min:box.min.toArray(),max:box.max.toArray()},triangles:0,materials:[]};
  gltf.scene.traverse(o=>{if(o.isMesh){window.report.triangles+=(o.geometry.index?.count||o.geometry.attributes.position.count)/3;
    window.report.materials.push({textureWidth:o.material.map?.image.width,textureHeight:o.material.map?.image.height,side:o.material.side});}});
  window.renderView(0);window.ready=true;
  </script>`;
  const server=http.createServer((req,res)=>{
    const url=new URL(req.url,'http://localhost').pathname;
    if(url==='/'){res.setHeader('Content-Type','text/html');res.end(html);return;}
    let file;
    if(url==='/model.glb') file=model;
    else if(url.startsWith('/three/')) {
      file=path.resolve(three,url.slice(7));
      if(!file.startsWith(three+path.sep)){res.writeHead(403).end();return;}
    }
    if(!file||!fs.existsSync(file)){res.writeHead(404).end();return;}
    res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':'model/gltf-binary');
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  const errors=[];
  try {
    const executablePath=process.env.CHROME_PATH||['C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe'].find(fs.existsSync);
    browser=await puppeteer.launch({executablePath,headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
    const page=await browser.newPage();await page.setViewport({width:600,height:600});
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`,{waitUntil:'networkidle0'});
    await page.waitForFunction(()=>window.ready,{timeout:60000});
    for(const [label,azimuth] of [['front',0],['three-quarter',45],['side',90],['back',180]]){
      await page.evaluate(a=>window.renderView(a),azimuth);
      await page.screenshot({path:path.join(output,label+'.png')});
    }
    const report=await page.evaluate(()=>window.report);
    assert.ok(report.triangles>0);assert.ok(Math.abs(report.bounds.min[1])<.001);
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'render.json'),JSON.stringify({...report,errors},null,2));
    console.log(JSON.stringify(report));
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
