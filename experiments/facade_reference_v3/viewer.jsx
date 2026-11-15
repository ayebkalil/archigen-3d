import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import Viewer3D from '/src/components/Viewer3D.jsx';
import '/src/index.css';
import '/src/workspace.css';
import {createAppearance} from './surfaceAppearance.js';
import house from './house.json';
import panels from './panels.json';
import state from './state.json';
const asset = name => new URL(name, import.meta.url).href;
const name = id => id.replace('ground_', 'Ground · ').replace('upper_', 'Upper · ').replaceAll('_', ' ');
function overview(direction = [1, 1]) {
  const V = window.__ARCHIGEN_VIEWER__, T = window.THREE;
  if (!V || !T) return;
  const box = new T.Box3();
  V.scene.children.filter(o => o.userData.isBuildingMesh).forEach(o => box.expandByObject(o));
  const center = box.getCenter(new T.Vector3()), size = box.getSize(new T.Vector3());
  const span = Math.max(size.x, size.z, size.y);
  V.camera.position.set(center.x + direction[0] * span * 1.0, center.y + span * .65, center.z + direction[1] * span * 1.0);
  V.controls.target.copy(center); V.controls.update();
}
function App() {
  const [mode, setMode] = useState(state.ready ? 'locked' : 'reference');
  const [side, setSide] = useState('ground_east');
  const [neural, setNeural] = useState(true);
  const [focus, setFocus] = useState(0);
  const appearance = useMemo(() => createAppearance(asset), []);
  useEffect(() => {
    const timer = setInterval(() => appearance.apply(mode, neural), 200);
    return () => clearInterval(timer);
  }, [mode, neural, appearance]);
  useEffect(() => () => appearance.dispose(), [appearance]);
  const layout = useMemo(() => {
    const value = structuredClone(house);
    const file = {v2: 'v2', guided: 'guided', locked: 'locked', reference: 'base', guide: 'sketch'}[mode];
    for (const panel of panels) value.walls.find(w => w.id === panel.wall_id).facade_texture = {
      ...panel, texture_base64: asset(`${panel.wall_id}/${file}.png`)};
    return value;
  }, [mode, neural]);
  const face = id => {setSide(id); setFocus(v => v + 1);};
  useEffect(() => {const timer = setTimeout(() => overview(), 1700); return () => clearTimeout(timer);}, []);
  window.__FACADE_V3__ = {mode, side, neural, panels, layout, appearanceReady: appearance.ready, face, overview, ready: state.ready};
  const button = active => ({padding: '9px 12px', border: '1px solid #c9c1b3', borderRadius: 7,
    background: active ? '#252a26' : '#fffdf7', color: active ? '#fff' : '#252a26', fontSize: 12});
  return <div className="design-app" style={{display:'flex',flexDirection:'column',height:'100vh'}}>
    <header style={{padding:'12px 20px',background:'#fffdf7',borderBottom:'1px solid #d7d0c4'}}>
      <div style={{display:'flex',alignItems:'center',gap:16,flexWrap:'wrap'}}>
        <div style={{marginRight:'auto'}}><strong style={{fontSize:20}}>FACADE · V3 CONSTRAINT TEST</strong>
          <div style={{fontSize:11,color:'#626c5c',marginTop:4}}>Same V2 weights · shared material · reference-guided sampling · bounded neural detail</div></div>
        <nav style={{display:'flex',gap:6,flexWrap:'wrap'}}>{[['locked','Material lock + neural trim'],['guided','Raw reference-guided'],['v2','V2 raw'],['reference','PBR reference input'],['guide','Opening guide']].map(([key,label]) =>
          <button data-mode={key} key={key} disabled={!state.ready && ['locked','guided'].includes(key)} onClick={()=>setMode(key)} style={button(mode===key)}>{label}</button>)}</nav>
      </div>
      <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap',marginTop:10}}>
        <select aria-label="Facade side" value={side} onChange={e=>face(e.target.value)} style={{padding:8,border:'1px solid #c9c1b3',borderRadius:6}}>
          {panels.map(p=><option value={p.wall_id} key={p.wall_id}>{name(p.wall_id)} · {p.openings.length} openings</option>)}
        </select>
        <button data-action="face" onClick={()=>face(side)} style={button(false)}>Face this wall</button>
        <button data-action="overview" onClick={()=>overview()} style={button(false)}>House overview</button>
        <button data-action="reverse" onClick={()=>overview([-1,-1])} style={button(false)}>Opposite corner</button>
        {mode==='locked' && <label style={{fontSize:12,marginLeft:'auto'}}><input type="checkbox" aria-label="Neural opening detail" checked={neural} onChange={e=>setNeural(e.target.checked)}/> Neural glass/door detail</label>}
      </div>
    </header>
    <main style={{flex:1,minHeight:0,position:'relative'}}><Viewer3D layout={layout} frontWallId={side} numFloors={1} facadeFocusTrigger={focus}/></main>
    <footer style={{background:'#fffdf7',padding:'10px 20px',fontSize:11,color:'#626c5c',borderTop:'1px solid #d7d0c4'}}>
      15 panels / 30 fixed physical openings. The locked mode preserves the shared PBR material outside measured openings and 4.5 cm trim bands.
      {' '}This is an inference/compositing experiment; no new trained-model accuracy claim. Production is unchanged.
    </footer>
  </div>;
}
createRoot(document.getElementById('root')).render(<App/>);
