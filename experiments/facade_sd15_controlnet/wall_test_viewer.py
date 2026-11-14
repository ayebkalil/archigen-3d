"""Stage an isolated page using the existing production Viewer3D component."""
import argparse
import json
from pathlib import Path
import shutil


def stage(project):
    project = Path(project).resolve()
    source = project / 'outputs/facade_house_test'
    target = project / 'frontend/tmp/facade-wall-test'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('house.json', 'baseline.json', 'probe.json', 'sketch.png', 'baseline.png', 'candidate.png', 'candidate.json', 'profile.json'):
        path = source / name
        if path.is_file():
            shutil.copyfile(path, target / name)
    (target / 'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>ArchiGen · Facade wall test</title></head>
<body><div id="root"></div><script type="module" src="./app.jsx"></script></body></html>''', encoding='utf-8')
    (target / 'app.jsx').write_text('''import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import Viewer3D from '/src/components/Viewer3D.jsx';
import '/src/index.css';
import '/src/workspace.css';
import house from './house.json';
import baseline from './baseline.json';
import probe from './probe.json';
import candidate from './candidate.json';
import profile from './profile.json';
const image = name => new URL(name, import.meta.url).href;
function App(){
 const [mode,setMode]=useState('candidate');
 const [focus,setFocus]=useState(1);
 const layout=React.useMemo(()=>{
   const value=structuredClone(house.layout);
   const spec=mode==='baseline'?baseline:candidate;
   const wall=value.walls.find(w=>w.id===house.wall.id);
   // Hold physical frames fixed across resolutions; only the image/UV letterbox changes.
   wall.facade_texture={...spec,openings:probe.openings,facade_frame:probe.facade_frame,
     wall_height_m:probe.wall_height_m,
     texture_base64:image(mode==='guide'?'sketch.png':mode==='baseline'?'baseline.png':'candidate.png')};
   return value;
 },[mode]);
 window.__FACADE_WALL_TEST__={mode,wallId:house.wall.id,probe,candidate,profile};
 return <div className="design-app" style={{display:'flex',flexDirection:'column',height:'100vh'}}>
  <header style={{padding:'14px 22px',background:'#fffefa',borderBottom:'1px solid #d6d1c7',display:'flex',gap:20,alignItems:'center',flexWrap:'wrap'}}>
   <div><strong style={{fontSize:20}}>FACADE · WALL TEST</strong><div style={{fontSize:12,color:'#697163'}}>Experimental checkpoint 750 · {probe.wall_length_m.toFixed(2)} m × {probe.wall_height_m.toFixed(2)} m · {probe.openings.length} fixed openings</div></div>
   <nav style={{display:'flex',gap:8}}>{[['baseline','Current Pix2Pix'],['candidate','Trained SD 512'],['guide','Opening guide']].map(([key,label])=><button key={key} data-mode={key} onClick={()=>{setMode(key);setFocus(v=>v+1)}} style={{padding:'10px 15px',border:'1px solid #b6aea0',borderRadius:6,background:mode===key?'#20201f':'#fffefa',color:mode===key?'#fff':'#20201f'}}>{label}</button>)}</nav>
   <button onClick={()=>setFocus(v=>v+1)} style={{padding:10,border:'1px solid #b6aea0',borderRadius:6}}>Face selected wall</button>
   <a href="./candidate.png" target="_blank" style={{marginLeft:'auto',fontSize:13}}>Open raw generated image ↗</a>
  </header>
  <main style={{flex:1,minHeight:0,position:'relative'}}><Viewer3D layout={layout} frontWallId={house.wall.id} numFloors={1} facadeFocusTrigger={focus}/></main>
  <footer style={{padding:'10px 22px',fontSize:12,background:'#fffefa'}}>Same house geometry and UV mapping in all three views. Generated imagery can contain extra features. Local generation: {profile.seconds.toFixed(1)} s · sampled GPU peak: {profile.sampled_peak_device_mib} MiB. Production model is unchanged.</footer>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
''', encoding='utf-8')
    print('Staged isolated viewer at ' + str(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    stage(parser.parse_args().project)
