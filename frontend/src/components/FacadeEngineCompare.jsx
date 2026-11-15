import React, { useRef, useState } from 'react';
import { Check, Download, Eye, RefreshCw, ZoomIn, ZoomOut, Columns2, SplitSquareHorizontal, AlertTriangle } from 'lucide-react';

export const ENGINES = {
  pix2pix: {
    label: 'Pix2Pix',
    short: 'Pix2Pix cGAN (reference)',
    controls: 'Style preset only: a post-hoc colour/finish applied to the Pix2Pix output. No text prompt, no seed.',
  },
  hybrid: {
    label: 'Stable Diffusion — hybrid',
    short: 'SD 1.5 material + geometry composition',
    controls: 'Style preset = Stable Diffusion text prompt for the wall material, plus a seed. Openings and wall size come only from the plan.',
  },
};

const PHASES = {
  queued: 'Queued',
  submitting: 'Submitting',
  loading_model: 'Loading model',
  waiting_gpu: 'Waiting for GPU (shared with TripoSR)',
  generating: 'Generating material',
  checking_material: 'Checking material',
  composing: 'Composing facade',
  validating_geometry: 'Validating geometry',
  cache_hit: 'Material cache hit',
  succeeded: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function phaseLabel(entry) {
  if (!entry) return 'Not generated';
  if (entry.status === 'outdated') return 'Discarded: the wall changed during generation';
  if (entry.status === 'interrupted') return 'Interrupted by a page reload';
  if (entry.status === 'failed') return 'Failed';
  if (entry.status === 'succeeded') return 'Done';
  const base = PHASES[entry.phase] || entry.phase || 'Running';
  if (entry.phase === 'generating' && entry.total_steps) {
    return `${base} · step ${entry.step || 0}/${entry.total_steps}${entry.attempt > 1 ? ` · attempt ${entry.attempt}` : ''}`;
  }
  return base;
}

// Region of the texture that is the wall itself, in image fractions (top-origin).
function wallRegion(data) {
  const b = data.uv_bounds || { u_min: 0, u_max: 1, v_min: 0, v_max: 1 };
  return { x0: b.u_min, x1: b.u_max, y0: 1 - b.v_max, y1: 1 - b.v_min };
}

export function nativeDescription(engine, data) {
  if (!data) return '';
  const info = data.model_info || {};
  const { x0, x1, y0, y1 } = wallRegion(data);
  if (engine === 'pix2pix') {
    const [w, h] = info.native_texture_px || [info.resolution || 256, info.resolution || 256];
    const wallW = Math.round((x1 - x0) * w), wallH = Math.round((y1 - y0) * h);
    return `Native ${w}×${h} px texture · wall region ${wallW}×${wallH} px · ${(wallW / data.wall_length_m).toFixed(1)} px/m`;
  }
  const [w, h] = info.texture_size_px || [0, 0];
  return `Material ${info.native_material_px}² px swatch (${(info.native_material_px / info.material_tile_m).toFixed(0)} px/m) · composed ${w}×${h} px · ${info.density_px_per_m?.toFixed(1)} px/m`;
}

function durationText(engine, entry) {
  if (!entry?.data) return null;
  const d = entry.data;
  if (engine === 'pix2pix') return `${(d.generation_ms ?? 0).toFixed(0)} ms server · ${(d.client_ms ?? 0).toFixed(0)} ms round trip`;
  const t = d.timings || {};
  if (d.model_info?.cache_hit) return `Material from cache · ${(t.total_s ?? 0).toFixed(2)} s total`;
  return `${(t.generation_s ?? 0).toFixed(1)} s SD · ${(t.composition_s ?? 0).toFixed(2)} s compose · ${((d.client_ms ?? 0) / 1000).toFixed(1)} s total`;
}

// Displays only the wall region at a shared frame size, so both engines are seen
// with the same framing, size, interpolation and zoom.
function WallView({ data, zoom, pan }) {
  const { x0, x1, y0, y1 } = wallRegion(data);
  return (
    <div className="absolute inset-0 overflow-hidden">
      <div className="absolute inset-0" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: 'center' }}>
        <img src={data.texture_base64} alt="" draggable={false}
          style={{ position: 'absolute', maxWidth: 'none', width: `${100 / (x1 - x0)}%`, height: `${100 / (y1 - y0)}%`,
            left: `${-x0 / (x1 - x0) * 100}%`, top: `${-y0 / (y1 - y0) * 100}%`, imageRendering: 'auto' }} />
      </div>
    </div>
  );
}

function Frame({ aspect, children, onPan }) {
  const drag = useRef(null);
  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-[#1d1d1f]/15 bg-[#d8d4cc] cursor-grab active:cursor-grabbing select-none"
      style={{ aspectRatio: aspect }}
      onPointerDown={event => { drag.current = { x: event.clientX, y: event.clientY }; event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerMove={event => {
        if (!drag.current) return;
        onPan(event.clientX - drag.current.x, event.clientY - drag.current.y);
        drag.current = { x: event.clientX, y: event.clientY };
      }}
      onPointerUp={() => { drag.current = null; }}>
      {children}
    </div>
  );
}

function EngineCard({ engine, entry, applied, onApply, onViewIn3D, applyError }) {
  const data = entry?.data;
  const running = entry?.status === 'running';
  return (
    <div className="rounded-xl border border-[#1d1d1f]/15 bg-white p-3 text-xs space-y-1.5" data-testid={`engine-card-${engine}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-stencil uppercase tracking-wider">{ENGINES[engine].label}</span>
        <span role="status" className={`rounded px-2 py-0.5 font-mono text-[10px] ${entry?.status === 'failed' ? 'bg-red-100 text-red-800' : entry?.status === 'succeeded' ? 'bg-emerald-100 text-emerald-800' : 'bg-[#efebe2] text-[#1d1d1f]/70'}`}>
          {running && <RefreshCw className="mr-1 inline h-3 w-3 animate-spin" />}{phaseLabel(entry)}
        </span>
      </div>
      {data && <p className="font-mono text-[10.5px] text-[#1d1d1f]/70">{nativeDescription(engine, data)}</p>}
      {data && <p className="font-mono text-[10.5px] text-[#1d1d1f]/70">{durationText(engine, entry)} · {data.device}</p>}
      {engine === 'hybrid' && data && (
        <p className="font-mono text-[10.5px] text-[#1d1d1f]/70">
          Seed {data.model_info.seed} · tile {data.model_info.material_tile_m.toFixed(2)} m ({data.model_info.scale?.method}) ·
          repeats {data.model_info.material_repeats.map(v => v.toFixed(1)).join(' × ')} · geometry {data.validation?.passed ? 'checked ✓' : 'not validated'}
          {data.model_info.rejected_attempts ? ` · ${data.model_info.rejected_attempts} rejected attempt(s)` : ''}
          {data.tracking?.status === 'failed' ? ' · MLflow logging failed' : ''}
        </p>
      )}
      {entry?.status === 'failed' && (
        <p className="flex gap-1 text-red-800"><AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          <span>{entry.error_type ? `[${entry.error_type}] ` : ''}{entry.error}. No other engine was substituted.</span></p>
      )}
      {applyError && <p className="text-red-800">{applyError}</p>}
      {data && (
        <div className="flex flex-wrap gap-2 pt-1">
          {!applied ? (
            <button type="button" onClick={onApply} data-testid={`apply-${engine}`}
              className="inline-flex h-8 items-center gap-1 rounded-full bg-[#1d1d1f] px-3 font-bold text-white">
              <Check size={14} /> Apply to wall
            </button>
          ) : (
            <button type="button" onClick={onViewIn3D} className="inline-flex h-8 items-center gap-1 rounded-full bg-emerald-700 px-3 font-bold text-white">
              <Eye size={14} /> Applied · view in 3D
            </button>
          )}
          <a href={data.texture_base64} download={`facade_${engine}_${data.wall_id}_${(data.style || 'default').toLowerCase().replace(/\s+/g, '_')}.png`}
            className="inline-flex h-8 items-center gap-1 rounded-full border border-[#1d1d1f]/30 px-3" data-testid={`download-${engine}`}>
            <Download size={14} /> Texture
          </a>
          {engine === 'hybrid' && data.material_base64 && (
            <a href={data.material_base64} download={`material_${data.model_info.material_key}.png`}
              className="inline-flex h-8 items-center gap-1 rounded-full border border-[#1d1d1f]/30 px-3">
              <Download size={14} /> Material swatch
            </a>
          )}
          {engine === 'hybrid' && data.sketch_base64 && (
            <a href={data.sketch_base64} download={`opening_mask_${data.wall_id}.png`}
              className="inline-flex h-8 items-center gap-1 rounded-full border border-[#1d1d1f]/30 px-3">
              <Download size={14} /> Opening mask
            </a>
          )}
        </div>
      )}
    </div>
  );
}

const ENGINE_OF = { pix2pix: 'pix2pix', stable_diffusion_hybrid: 'hybrid' };

// "All walls" mode: one live card per wall of the level, showing what is applied in 3D.
export function FacadeLevelGallery({ title, walls, progress, onOpenWall, onViewIn3D }) {
  const failed = new Map((progress?.failed || []).map(item => [item.id, item.error]));
  const applied = new Set(progress?.applied || []);
  const status = wall => {
    if (progress?.current === wall.id) return { label: 'Generating…', tone: 'bg-orange-100 text-orange-900', running: true };
    if (failed.has(wall.id)) return { label: 'Failed', tone: 'bg-red-100 text-red-800' };
    if (applied.has(wall.id)) return { label: 'Applied now', tone: 'bg-emerald-100 text-emerald-800' };
    if (progress && !progress.finished && !applied.has(wall.id)) return { label: 'Waiting', tone: 'bg-[#efebe2] text-[#1d1d1f]/70' };
    return wall.facade_texture?.texture_base64 ? { label: 'Applied earlier', tone: 'bg-[#efebe2] text-[#1d1d1f]/70' } : { label: 'Unfinished', tone: 'bg-[#efebe2] text-[#1d1d1f]/60' };
  };
  const finished = walls.filter(wall => wall.facade_texture?.texture_base64).length;
  return (
    <div className="space-y-3" data-testid="facade-level-gallery">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-stencil text-xs uppercase tracking-wider">{title} · {walls.length} walls</h3>
        <span className="font-mono text-[11px] text-[#1d1d1f]/65">{finished} / {walls.length} with a facade in 3D
          {progress && !progress.finished ? ` · generating ${progress.done + 1}/${progress.total}` : ''}</span>
      </div>
      <p className="font-mono text-[10.5px] text-[#1d1d1f]/60">
        Each card shows the texture currently applied to that wall (same framing for every wall). Click a card to open that wall alone and compare engines.
      </p>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
        {walls.map(wall => {
          const data = wall.facade_texture?.texture_base64 ? wall.facade_texture : null;
          const length = data?.facade_frame?.length_m || data?.uv_bounds?.wall_length_m;
          const height = data?.wall_height_m || data?.uv_bounds?.wall_height_m;
          const aspect = length && height ? Math.max(.3, Math.min(6, length / height)) : 2.5;
          const engine = ENGINE_OF[data?.engine] || (data ? 'pix2pix' : null);
          const state = status(wall);
          return (
            <div key={wall.id} className="rounded-xl border border-[#1d1d1f]/15 bg-white p-2 text-xs" data-testid={`gallery-${wall.id}`}>
              <button type="button" onClick={() => onOpenWall(wall.id)} className="block w-full text-left" title="Open this wall alone">
                {/* Same card height for every wall; each wall keeps its own proportions. */}
                <div className="flex h-36 w-full items-center justify-center rounded-lg bg-[#efebe2]">
                  <div className="relative max-h-full max-w-full overflow-hidden rounded bg-[#d8d4cc]"
                    style={aspect >= 2.2 ? { width: '100%', aspectRatio: aspect } : { height: '100%', aspectRatio: aspect }}>
                    {data ? <WallView data={data} zoom={1} pan={{ x: 0, y: 0 }} /> : (
                      <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-[#1d1d1f]/50">No facade yet</div>
                    )}
                    {state.running && (
                      <div className="absolute inset-0 flex items-center justify-center bg-white/60"><RefreshCw className="h-6 w-6 animate-spin text-[#f0500f]" /></div>
                    )}
                  </div>
                </div>
              </button>
              <div className="mt-1.5 flex items-center justify-between gap-2">
                <span className="truncate font-bold">{wall.id}</span>
                <span className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] ${state.tone}`}>{state.label}</span>
              </div>
              <p className="font-mono text-[10px] text-[#1d1d1f]/65">
                {length ? `${length.toFixed(2)} × ${height.toFixed(2)} m · ` : ''}{data?.openings?.length ?? wall.openings?.length ?? 0} opening(s)
                {data ? ` · ${engine ? ENGINES[engine].label : ''} · ${data.style || 'default'}` : ''}
              </p>
              {data && <p className="font-mono text-[10px] text-[#1d1d1f]/55">{nativeDescription(engine, data).split(' · ').slice(0, 2).join(' · ')}</p>}
              {failed.has(wall.id) && <p className="text-[10.5px] text-red-800">{failed.get(wall.id)}</p>}
              {data && (
                <a href={data.texture_base64} download={`facade_${engine}_${wall.id}.png`} className="mt-1 inline-flex items-center gap-1 text-[10.5px] underline">
                  <Download size={12} /> Texture
                </a>
              )}
            </div>
          );
        })}
      </div>
      {finished > 0 && (
        <button type="button" onClick={onViewIn3D} className="inline-flex h-9 items-center gap-1 rounded-full bg-[#1d1d1f] px-4 text-xs font-bold text-white">
          <Eye size={14} /> View the floor in 3D
        </button>
      )}
    </div>
  );
}

export default function FacadeEngineCompare({ record, appliedTexture, onApply, onViewIn3D }) {
  const [mode, setMode] = useState('side');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [split, setSplit] = useState(50);
  const [applyErrors, setApplyErrors] = useState({});
  const entries = record?.entries || {};
  const any = entries.pix2pix?.data || entries.hybrid?.data;
  const reference = entries.pix2pix?.data || entries.hybrid?.data;
  const aspect = reference ? Math.max(0.2, Math.min(8, reference.wall_length_m / reference.wall_height_m)) : 2;
  const both = entries.pix2pix?.data && entries.hybrid?.data;
  const changeZoom = factor => setZoom(z => {
    const next = Math.min(8, Math.max(1, z * factor));
    if (next === 1) setPan({ x: 0, y: 0 });
    return next;
  });
  const movePan = (dx, dy) => zoom > 1 && setPan(p => ({ x: p.x + dx, y: p.y + dy }));
  const apply = engine => {
    try {
      onApply(entries[engine].data);
      setApplyErrors(e => ({ ...e, [engine]: null }));
    } catch (error) {
      setApplyErrors(e => ({ ...e, [engine]: error.message }));
    }
  };
  const label = (engine) => (
    <div className="absolute left-2 top-2 z-10 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-white">
      {ENGINES[engine].label} · {nativeDescription(engine, entries[engine].data).split(' · ')[0]}
    </div>
  );

  return (
    <div className="space-y-3" data-testid="facade-engine-compare">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-stencil text-xs uppercase tracking-wider">Engine comparison · same wall, same openings</h3>
          {reference && <p className="font-mono text-[10.5px] text-[#1d1d1f]/65">
            {reference.wall_id} · {reference.wall_length_m.toFixed(2)} m × {reference.wall_height_m.toFixed(2)} m ·
            {' '}{reference.openings?.length ?? 0} opening(s) · flat 2D preview at identical display size and interpolation (no extra filtering)
          </p>}
        </div>
        <div className="flex items-center gap-1">
          <button type="button" aria-pressed={mode === 'side'} onClick={() => setMode('side')} title="Side by side"
            className="rounded border px-2 py-1 aria-pressed:border-[#f0500f] aria-pressed:bg-[#f0500f]/10"><Columns2 size={15} /></button>
          <button type="button" aria-pressed={mode === 'slider'} onClick={() => setMode('slider')} disabled={!both} title="Before / after slider"
            className="rounded border px-2 py-1 aria-pressed:border-[#f0500f] aria-pressed:bg-[#f0500f]/10 disabled:opacity-40"><SplitSquareHorizontal size={15} /></button>
          <span className="mx-1 h-5 w-px bg-[#1d1d1f]/20" />
          <button type="button" onClick={() => changeZoom(1 / 1.5)} disabled={zoom <= 1} title="Zoom out" className="rounded border px-2 py-1 disabled:opacity-40"><ZoomOut size={15} /></button>
          <span className="w-10 text-center font-mono text-[11px]" data-testid="compare-zoom">{zoom.toFixed(1)}×</span>
          <button type="button" onClick={() => changeZoom(1.5)} disabled={zoom >= 8} title="Zoom in" className="rounded border px-2 py-1 disabled:opacity-40"><ZoomIn size={15} /></button>
        </div>
      </div>

      {!any && (
        <div className="rounded-xl border border-dashed border-[#1d1d1f]/25 p-8 text-center font-mono text-xs text-[#1d1d1f]/60">
          Generate with one engine, or compare both, to see the results here.
        </div>
      )}

      {any && mode === 'side' && (
        <div className={`grid gap-3 ${aspect > 2.2 ? 'grid-cols-1' : 'grid-cols-1 xl:grid-cols-2'}`}>
          {['pix2pix', 'hybrid'].map(engine => (
            <div key={engine}>
              <Frame aspect={aspect} onPan={movePan}>
                {entries[engine]?.data ? <>{label(engine)}<WallView data={entries[engine].data} zoom={zoom} pan={pan} /></> : (
                  <div className="absolute inset-0 flex items-center justify-center font-mono text-xs text-[#1d1d1f]/55">
                    {ENGINES[engine].label}: {phaseLabel(entries[engine])}
                  </div>
                )}
              </Frame>
            </div>
          ))}
        </div>
      )}

      {any && mode === 'slider' && both && (
        <div>
          <Frame aspect={aspect} onPan={movePan}>
            <WallView data={entries.pix2pix.data} zoom={zoom} pan={pan} />
            <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
              <WallView data={entries.hybrid.data} zoom={zoom} pan={pan} />
            </div>
            <div className="absolute inset-y-0 z-10 w-0.5 bg-white shadow" style={{ left: `${split}%` }} />
            <div className="absolute left-2 top-2 z-10 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-white">◀ Pix2Pix</div>
            <div className="absolute right-2 top-2 z-10 rounded bg-black/70 px-2 py-1 font-mono text-[10px] text-white">Stable Diffusion — hybrid ▶</div>
          </Frame>
          <input type="range" min="0" max="100" value={split} onChange={event => setSplit(Number(event.target.value))}
            aria-label="Comparison split" className="mt-2 w-full" />
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {['pix2pix', 'hybrid'].map(engine => (
          <EngineCard key={engine} engine={engine} entry={entries[engine]} applyError={applyErrors[engine]}
            applied={Boolean(entries[engine]?.data && appliedTexture && appliedTexture === entries[engine].data.texture_base64)}
            onApply={() => apply(engine)} onViewIn3D={onViewIn3D} />
        ))}
      </div>
      {entries.hybrid?.data && (
        <p className="font-mono text-[10.5px] text-[#1d1d1f]/60">
          Hybrid: windows and doors are schematic fills limited to the plan's opening masks; in 3D the viewer cuts those
          regions and shows its own door/window meshes. Colour texture only — no normal or roughness maps.
        </p>
      )}
    </div>
  );
}
