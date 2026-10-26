import React, { useState } from 'react';
import { Sparkles, RefreshCw, Trash2, X, AlertCircle, AlertTriangle, CheckCircle2, ChevronRight, Info, Layers, Compass } from 'lucide-react';

const PRESET_OPTIONS = [
  { id: 'contemporary_minimal', label: 'Contemporary Minimal', desc: 'Clean lines, neutral tones, light oak & steel' },
  { id: 'warm_wood', label: 'Warm Wood', desc: 'Natural oak textures, warm fabric upholstery' },
  { id: 'tailored_classic', label: 'Tailored Classic', desc: 'Structured profiles, richer tones, grounded proportions' }
];

const ROOM_TYPE_OPTIONS = [
  'Bedroom',
  'Living Room',
  'Kitchen',
  'Bathroom',
  'Office',
  'Dining'
];

export default function RoomDesignPanel({
  selectedRoom,
  onClose,
  onGenerateDesign,
  onRegenerateDesign,
  onClearDesign,
  furnishingData,
  isDesigning,
  onRelabelRoom
}) {
  const [selectedPreset, setSelectedPreset] = useState('contemporary_minimal');
  const [showRelabel, setShowRelabel] = useState(false);

  if (!selectedRoom) return null;

  const currentFurnishings = furnishingData?.[selectedRoom.id] || null;
  const items = currentFurnishings?.items || [];
  const validation = currentFurnishings?.validation || null;
  const failureReason = currentFurnishings?.failure_reason || null;
  const hasFailed = currentFurnishings && !currentFurnishings.is_valid;
  const isFurnished = currentFurnishings && currentFurnishings.is_valid && items.length > 0;
  const meta = currentFurnishings?.metadata || {};
  const actualScale = meta.scale_m_per_px || 0.03;
  const scaleSource = meta.scale_source || (meta.scale_uncalibrated ? 'Model Estimated' : '2-Pt Calibrated');

  // Approximate room width & depth from bounding box
  const poly = selectedRoom.polygon_meters || [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  poly.forEach(([x, y]) => {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  const widthM = (maxX - minX > 0 && maxX !== Infinity) ? (maxX - minX).toFixed(1) : '-';
  const depthM = (maxY - minY > 0 && maxY !== Infinity) ? (maxY - minY).toFixed(1) : '-';

  const getFailureMessage = (reason) => {
    switch (reason) {
      case 'NO_USABLE_WALL':
        return {
          title: 'No Usable Wall Detected',
          desc: 'No continuous solid wall segment (>= required item width) was found unobstructed by door/window openings.',
          action: 'Adjust door positions using the 2D CAD toolbar or enlarge wall span.'
        };
      case 'TOO_SMALL':
        return {
          title: 'Room Space Too Compact',
          desc: `Area (${selectedRoom.surface_m2} m²) is too compact for standard ${selectedRoom.room_type} furnishings.`,
          action: 'Consider relabeling as a single bedroom or adjusting room boundary.'
        };
      case 'DOOR_BLOCKED':
        return {
          title: 'Door Entry Swing Compromised',
          desc: 'Furniture would obstruct the required 0.85m door swing clearance envelope.',
          action: 'Relocate the door opening in the 2D editor or select a smaller preset.'
        };
      case 'UNREACHABLE':
        return {
          title: 'Walking Corridor Blocked',
          desc: 'Internal 0.50m BFS walking path from entrance is pinched by placed assets.',
          action: 'Adjust room boundaries in 2D to provide wider corridor clearance.'
        };
      default:
        return {
          title: 'Placement Verification Failed',
          desc: 'Strict spatial constraints (0 collisions, door clearances) could not be satisfied safely.',
          action: 'Try another preset or use the 2D editor to refine room shape.'
        };
    }
  };

  return (
    <div className="absolute top-4 right-4 z-30 w-84 bg-[#efebe2]/95 backdrop-blur-md border border-[#1d1d1f]/30 shadow-2xl rounded-sm font-mono text-xs overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-[#1d1d1f] text-[#efebe2] px-3 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-[#b8452a]" />
          <span className="font-semibold tracking-wider uppercase text-[11px]">Room Design Engine</span>
        </div>
        <button
          onClick={onClose}
          className="text-[#efebe2]/60 hover:text-[#efebe2] transition-colors p-0.5"
          title="Close panel"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-3.5 space-y-3 max-h-[80vh] overflow-y-auto">
        {/* Room Info Card */}
        <div className="bg-[#1d1d1f]/5 border border-[#1d1d1f]/15 p-2.5 rounded-sm space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[#1d1d1f] text-sm">
              {selectedRoom.room_type || 'Selected Space'}
            </span>
            <button
              onClick={() => setShowRelabel(!showRelabel)}
              className="text-[10px] text-[#b8452a] underline uppercase hover:text-[#96351e]"
            >
              {showRelabel ? 'Cancel' : 'Relabel'}
            </button>
          </div>

          {showRelabel && (
            <div className="grid grid-cols-2 gap-1 pt-1.5 border-t border-[#1d1d1f]/10">
              {ROOM_TYPE_OPTIONS.map((type) => (
                <button
                  key={type}
                  onClick={() => {
                    if (onRelabelRoom) onRelabelRoom(selectedRoom.id, type);
                    setShowRelabel(false);
                  }}
                  className={`px-1.5 py-1 text-[10px] text-left border rounded-sm transition-colors ${
                    selectedRoom.room_type === type
                      ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                      : 'bg-white/80 border-[#1d1d1f]/20 hover:border-[#b8452a]'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] text-[#1d1d1f]/70">
            <div>Area: <span className="font-semibold text-[#1d1d1f]">{selectedRoom.surface_m2} m²</span></div>
            <div>Span: <span className="font-semibold text-[#1d1d1f]">{widthM}m × {depthM}m</span></div>
          </div>
        </div>

        {/* Preset Selector */}
        <div className="space-y-1.5">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-[#1d1d1f]/70 flex items-center gap-1">
            <Compass className="w-3 h-3 text-[#b8452a]" /> Style Preset
          </label>
          <div className="space-y-1">
            {PRESET_OPTIONS.map((opt) => (
              <label
                key={opt.id}
                className={`flex items-start gap-2 p-2 border rounded-sm cursor-pointer transition-all ${
                  selectedPreset === opt.id
                    ? 'bg-[#1d1d1f]/10 border-[#1d1d1f] shadow-sm'
                    : 'bg-white/60 border-[#1d1d1f]/15 hover:bg-white/90'
                }`}
              >
                <input
                  type="radio"
                  name="preset"
                  value={opt.id}
                  checked={selectedPreset === opt.id}
                  onChange={() => setSelectedPreset(opt.id)}
                  className="mt-0.5 accent-[#b8452a]"
                />
                <div className="flex-1 leading-tight">
                  <div className="font-medium text-[11px] text-[#1d1d1f]">{opt.label}</div>
                  <div className="text-[9px] text-[#1d1d1f]/60 mt-0.5">{opt.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Failure Notice / Guidance */}
        {hasFailed && (
          <div className="space-y-2">
            <div className="bg-[#b8452a]/10 border border-[#b8452a]/30 p-2.5 rounded-sm text-[#b8452a] space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-[11px]">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                <span>{getFailureMessage(failureReason).title}</span>
              </div>
              <p className="text-[10px] leading-relaxed text-[#1d1d1f]/80">
                {getFailureMessage(failureReason).desc}
              </p>
              <div className="pt-1 border-t border-[#b8452a]/20 text-[10px] font-medium text-[#b8452a]">
                Tip: {getFailureMessage(failureReason).action}
              </div>
            </div>

            {/* Scale & Calibration Status Badge (shows actual scale & source) */}
            {failureReason === 'TOO_SMALL' && (
              <div className="bg-amber-500/10 border border-amber-600/30 p-2 rounded-sm text-amber-900 space-y-1">
                <div className="flex items-center gap-1.5 font-semibold text-[10.5px] text-amber-800">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
                  <span>Scale Calibration Notice</span>
                </div>
                <div className="text-[10px] space-y-0.5 text-amber-900/90 font-mono">
                  <div>Actual Scale: <strong className="font-semibold">{actualScale} m/px</strong></div>
                  <div>Source: <strong className="font-semibold">{scaleSource}</strong></div>
                </div>
                <p className="text-[9.5px] leading-relaxed text-amber-900/80 pt-0.5 border-t border-amber-600/15 font-sans">
                  Use the <strong>2-Pt Calibrate</strong> tool in the top bar to set precise physical scale from a known doorway or wall.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Placed Items Summary */}
        {isFurnished && (
          <div className="bg-[#2a7a6e]/10 border border-[#2a7a6e]/30 p-2.5 rounded-sm space-y-1.5">
            <div className="flex items-center justify-between text-[#2a7a6e] font-semibold text-[11px]">
              <div className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Collision-free (0 Collisions)</span>
              </div>
              <span className="text-[10px]">{items.length} items</span>
            </div>
            <ul className="text-[10px] text-[#1d1d1f]/80 space-y-0.5 max-h-24 overflow-y-auto">
              {items.map((it, idx) => (
                <li key={idx} className="flex justify-between items-center py-0.5 border-b border-[#1d1d1f]/5">
                  <span className="font-medium">{it.name}</span>
                  <span className="text-[#1d1d1f]/50 text-[9px]">{it.rotation_deg}°</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Action Controls */}
        <div className="space-y-1.5 pt-1">
          <button
            onClick={() => onGenerateDesign(selectedRoom, selectedPreset)}
            disabled={isDesigning}
            className="w-full bg-[#1d1d1f] hover:bg-[#333] text-[#efebe2] py-2 px-3 rounded-sm font-semibold text-[11px] uppercase tracking-wider flex items-center justify-center gap-1.5 transition-all shadow-md disabled:opacity-50"
          >
            {isDesigning ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Generating Placement...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5 text-[#b8452a]" />
                <span>Generate Design</span>
              </>
            )}
          </button>

          <div className="grid grid-cols-2 gap-1.5">
            <button
              onClick={() => onRegenerateDesign(selectedRoom, selectedPreset)}
              disabled={isDesigning || !isFurnished}
              className="bg-white/80 hover:bg-white text-[#1d1d1f] border border-[#1d1d1f]/25 py-1.5 px-2 rounded-sm font-medium text-[10px] uppercase tracking-wider flex items-center justify-center gap-1 transition-all disabled:opacity-40"
              title="Regenerate with a new random seed"
            >
              <RefreshCw className="w-3 h-3 text-[#1d1d1f]/70" />
              <span>Regenerate</span>
            </button>

            <button
              onClick={() => onClearDesign(selectedRoom.id)}
              disabled={isDesigning || !items.length}
              className="bg-white/80 hover:bg-[#b8452a]/10 text-[#b8452a] border border-[#b8452a]/30 py-1.5 px-2 rounded-sm font-medium text-[10px] uppercase tracking-wider flex items-center justify-center gap-1 transition-all disabled:opacity-40"
              title="Remove items and free WebGL memory"
            >
              <Trash2 className="w-3 h-3" />
              <span>Clear</span>
            </button>
          </div>
        </div>

        {/* Offline Rule-Based Engine Note */}
        <div className="text-[9px] text-[#1d1d1f]/50 text-center tracking-tight pt-1">
          Deterministic Rule-Based Placement Engine &bull; Offline &bull; Zero External APIs
        </div>
      </div>
    </div>
  );
}
