import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import Viewer3D from '/src/components/Viewer3D.jsx';
import '/src/index.css';
import '/src/workspace.css';
import house from './house.json';
import panels from './panels.json';
import baselines from './baselines.json';
import profile from './profile.json';

const asset = name => new URL(name, import.meta.url).href;
const title = id => id.replace('ground_', 'Ground · ').replace('upper_', 'Upper · ').replaceAll('_', ' ');
function overview(direction = [1, 1]) {
  const viewer = window.__ARCHIGEN_VIEWER__, T = window.THREE;
  if (!viewer || !T) return;
  const {scene, camera, controls} = viewer;
  const box = new T.Box3();
  scene.children.filter(o => o.userData.isBuildingMesh).forEach(o => box.expandByObject(o));
  const center = box.getCenter(new T.Vector3()), size = box.getSize(new T.Vector3());
  const span = Math.max(size.x, size.z, size.y);
  camera.position.set(center.x + direction[0] * span * 1.12, center.y + span * .78, center.z + direction[1] * span * 1.12);
  camera.up.set(0, 1, 0); controls.target.copy(center); controls.update();
}
function App() {
  const [mode, setMode] = useState('candidate');
  const [side, setSide] = useState('ground_east');
  const [focus, setFocus] = useState(0);
  const current = panels.find(p => p.wall_id === side);
  const layout = useMemo(() => {
    const value = structuredClone(house);
    for (const panel of panels) {
      const wall = value.walls.find(w => w.id === panel.wall_id);
      const texture = mode === 'plain' ? asset('plain.png') : asset(`${panel.wall_id}/${mode === 'guide' ? 'sketch' : mode === 'baseline' ? 'baseline' : 'candidate'}.png`);
      // Identical canonical opening positions and sizes in every comparison mode.
      wall.facade_texture = {...panel, uv_bounds: mode === 'baseline' ? baselines[panel.wall_id].uv_bounds : panel.uv_bounds,
        texture_base64: texture};
    }
    return value;
  }, [mode]);
  const face = id => {setSide(id); setFocus(value => value + 1);};
  useEffect(() => {const timer = setTimeout(() => overview(), 1700); return () => clearTimeout(timer);}, []);
  window.__FACADE_MULTIFACE__ = {mode, side, panels, profile, layout, face, overview};
  const button = active => ({padding: '9px 12px', border: '1px solid #c6beb0', borderRadius: 7, cursor:'pointer',
    background: active ? '#252824' : '#fffef9', color: active ? '#fff' : '#252824', fontSize: 12});
  return <div className="design-app" style={{height: '100vh', display: 'flex', flexDirection: 'column'}}>
    <header style={{padding: '12px 20px', background: '#fffef9', borderBottom: '1px solid #d6d1c7'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap'}}>
        <div style={{marginRight:'auto'}}><strong style={{fontSize: 20}}>FACADE V2 · HOUSE TEST</strong>
          <div style={{fontSize: 11, color:'#646b60', marginTop: 3}}>15 exterior panels · two floors · checkpoint {profile.checkpoint_step} · experimental neutral finish</div></div>
        <nav style={{display: 'flex', gap: 6}}>{[['candidate', 'V2 diffusion'], ['baseline', 'Current facade'], ['plain', 'Plain finish'], ['guide', 'Opening guide']].map(([key, label]) =>
          <button key={key} data-mode={key} onClick={() => setMode(key)} style={button(mode === key)}>{label}</button>)}</nav>
      </div>
      <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap',marginTop:10}}>
        <label style={{fontSize:12}}>Inspect side <select aria-label="Facade side" value={side} onChange={e=>face(e.target.value)} style={{padding:8,marginLeft:7,border:'1px solid #c6beb0',borderRadius:6}}>
          {['ground','upper'].map(level=><optgroup key={level} label={level === 'ground' ? 'Ground floor' : 'Upper floor'}>
            {panels.filter(p=>p.level_id===level).map(p=><option key={p.wall_id} value={p.wall_id}>{title(p.wall_id)} · {p.openings.length} openings</option>)}
          </optgroup>)}
        </select></label>
        <button data-action="face" onClick={()=>face(side)} style={button(false)}>Face this wall</button>
        <button data-action="overview" onClick={()=>overview()} style={button(false)}>House overview</button>
        <button data-action="reverse" onClick={()=>overview([-1,-1])} style={button(false)}>Opposite corner</button>
        <span style={{fontSize:11,color:'#646b60',marginLeft:'auto'}}>{current.wall_length_m.toFixed(2)} × {current.wall_height_m.toFixed(2)} m · {current.openings.length} fixed openings</span>
        <a style={{fontSize:12}} href={asset(`${side}/candidate.png`)} target="_blank" rel="noreferrer">Raw V2 texture ↗</a>
      </div>
    </header>
    <main style={{flex:1,minHeight:0,position:'relative'}}><Viewer3D layout={layout} frontWallId={side} numFloors={1} facadeFocusTrigger={focus}/></main>
    <footer style={{background:'#fffef9',padding:'9px 20px',fontSize:11,color:'#646b60',borderTop:'1px solid #d6d1c7'}}>
      All sides are applied together. Drag to orbit; use the level selector to inspect either floor. Images may contain invented details; physical holes remain fixed.
      {' '}Generation: {profile.device} · sampled GPU peak {profile.sampled_peak_device_mib} MiB. Current app configuration is unchanged.
    </footer>
  </div>;
}
createRoot(document.getElementById('root')).render(<App/>);
