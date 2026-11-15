import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Library, MousePointerClick, RotateCcw, RotateCw, Move, Trash2, X, ChevronDown, ChevronUp } from 'lucide-react';
import { useProjectState } from '../state/ProjectContext.jsx';

/** Saved image-to-3D models (TripoSR) that can be placed anywhere on a floor of the house. */
export default function RoomLibraryPanel({ placement, placementError, placed, selected, onStartPlacement, onCancelPlacement,
  onRotate, onMove, onRemove, onDeselect }) {
  const [open, setOpen] = useProjectState('viewer.libraryOpen', true);
  const [revision] = useProjectState('library.revision', 0);
  const [items, setItems] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('/furniture/custom-library')
      .then(res => { setItems(Object.values(res.data || {}).reverse()); setError(null); })
      .catch(() => setError('The room library could not be loaded.'));
  }, [revision, open]);

  useEffect(() => {
    if (!placement) return undefined;
    const onKey = event => { if (event.key === 'Escape') onCancelPlacement(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [placement, onCancelPlacement]);

  return (
    <div className="pointer-events-none absolute bottom-3 left-3 top-16 z-20 flex w-72 max-w-[calc(100%-1.5rem)] flex-col justify-end gap-2" data-testid="room-library">
      {placement && (
        <div className="pointer-events-auto rounded-xl border-2 border-[#f0500f] bg-white p-3 text-xs shadow-lg" role="status" data-testid="placement-banner">
          <p className="flex items-center gap-1.5 font-bold"><MousePointerClick size={15} className="text-[#f0500f]" />
            {placement.moveId ? 'Move' : 'Place'} « {placement.item.name} »</p>
          <p className="mt-1 text-[#1d1d1f]/70">Click on a floor of the house. Esc to cancel.</p>
          {placementError && <p className="mt-1 text-red-800">{placementError}</p>}
          <button type="button" onClick={onCancelPlacement} className="mt-2 underline">Cancel</button>
        </div>
      )}

      {selected && !placement && (
        <div className="pointer-events-auto rounded-xl border border-[#1d1d1f]/20 bg-white p-3 text-xs shadow-lg" data-testid="placed-controls">
          <div className="flex items-center justify-between">
            <strong className="truncate">{selected.name}</strong>
            <button type="button" onClick={onDeselect} title="Close" className="rounded p-0.5 hover:bg-black/5"><X size={14} /></button>
          </div>
          <p className="text-[10px] text-[#1d1d1f]/60">{selected.dimensions_m.map(v => v.toFixed(2)).join(' × ')} m · rotation {Math.round(selected.rotation_deg)}°</p>
          <div className="mt-2 grid grid-cols-4 gap-1">
            <button type="button" onClick={() => onRotate(selected.id, -90)} title="Rotate 90° left" className="rounded border py-1"><RotateCcw size={14} className="mx-auto" /></button>
            <button type="button" onClick={() => onRotate(selected.id, -15)} title="Rotate 15° left" className="rounded border py-1 text-[10px]">-15°</button>
            <button type="button" onClick={() => onRotate(selected.id, 15)} title="Rotate 15° right" className="rounded border py-1 text-[10px]">+15°</button>
            <button type="button" onClick={() => onRotate(selected.id, 90)} title="Rotate 90° right" className="rounded border py-1"><RotateCw size={14} className="mx-auto" /></button>
          </div>
          <div className="mt-1 grid grid-cols-2 gap-1">
            <button type="button" onClick={() => onMove(selected.id)} data-testid="move-placed" className="flex items-center justify-center gap-1 rounded border py-1"><Move size={13} /> Move</button>
            <button type="button" onClick={() => onRemove(selected.id)} data-testid="remove-placed" className="flex items-center justify-center gap-1 rounded border border-red-300 py-1 text-red-800"><Trash2 size={13} /> Remove</button>
          </div>
        </div>
      )}

      <div className="pointer-events-auto flex max-h-full min-h-0 flex-col rounded-xl border border-[#1d1d1f]/20 bg-white/95 shadow-lg backdrop-blur">
        <button type="button" onClick={() => setOpen(!open)} className="flex items-center justify-between px-3 py-2 text-xs" data-testid="room-library-toggle">
          <span className="flex items-center gap-1.5 font-stencil uppercase tracking-wider"><Library size={14} className="text-[#f0500f]" /> Room library ({items.length})</span>
          <span className="flex items-center gap-1 text-[10px] text-[#1d1d1f]/60">{placed.length} placed {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}</span>
        </button>
        {open && (
          <div className="min-h-0 space-y-1.5 overflow-y-auto border-t border-[#1d1d1f]/10 p-2">
            {error && <p className="text-xs text-red-800">{error}</p>}
            {!error && !items.length && (
              <p className="p-2 text-xs text-[#1d1d1f]/60">No saved model yet. Generate furniture from a photo in the Generative Studio: it is saved here automatically.</p>
            )}
            {items.map(item => (
              <div key={item.id} className={`flex items-center gap-2 rounded-lg border p-1.5 ${placement?.item.id === item.id && !placement.moveId ? 'border-[#f0500f] bg-[#f0500f]/5' : 'border-[#1d1d1f]/10'}`} data-testid={`library-${item.id}`}>
                <div className="h-12 w-12 shrink-0 overflow-hidden rounded bg-[#efebe2]">
                  {item.preview ? <img src={item.preview} alt="" className="h-full w-full object-contain" /> : <Library size={18} className="m-3.5 text-[#1d1d1f]/30" />}
                </div>
                <div className="min-w-0 flex-1 text-[11px]">
                  <p className="truncate font-bold">{item.name}</p>
                  <p className="truncate text-[10px] text-[#1d1d1f]/60">{item.category} · {item.dimensions_m?.map(v => v.toFixed(2)).join(' × ')} m</p>
                  <p className="text-[10px] text-[#1d1d1f]/50">{placed.filter(p => p.library_id === item.id).length} in the house</p>
                </div>
                <button type="button" onClick={() => onStartPlacement(item)} data-testid={`place-${item.id}`}
                  className="shrink-0 rounded-full bg-[#1d1d1f] px-3 py-1 text-[11px] font-bold text-white">Place</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
