import React from 'react';
import { Home, ShieldCheck, Maximize2, LayoutGrid, CheckCircle, AlertTriangle } from 'lucide-react';

function checkRoomPlausibility(room) {
  const t = (room.room_type || '').toLowerCase();
  const s = room.surface_m2 || 0;
  if (t.includes('bed')) {
    if (s < 5.0) return { flag: true, warning: 'Bedroom < 5 m² (under standard living norm)' };
    if (s > 45.0) return { flag: true, warning: 'Bedroom > 45 m² (unusually large)' };
  } else if (t.includes('bath')) {
    if (s < 1.8) return { flag: true, warning: 'Bathroom < 1.8 m² (under sanitary code minimum)' };
    if (s > 20.0) return { flag: true, warning: 'Bathroom > 20 m² (unusually large)' };
  } else if (t.includes('kitchen')) {
    if (s < 3.0) return { flag: true, warning: 'Kitchen < 3 m² (cramped)' };
    if (s > 40.0) return { flag: true, warning: 'Kitchen > 40 m² (unusually large)' };
  }
  return { flag: false, warning: null };
}

export default function MetricsSummary({ layout, frontWallId }) {
  if (!layout) return null;

  const metadata = layout.metadata || {};
  const rooms = layout.rooms || [];
  const walls = layout.walls || [];
  const totalArea = metadata.total_surface_m2 || 0;
  const isShellValid = metadata.is_building_shell_valid ?? true;

  // Run architectural plausibility checks
  const plausibilityResults = rooms.map((r) => ({
    ...r,
    ...checkRoomPlausibility(r),
  }));
  const warningsCount = plausibilityResults.filter((r) => r.flag).length;

  return (
    <footer className="h-40 border-t border-[#1d1d1f]/20 bg-[#efebe2]/95 backdrop-blur-md px-6 py-2.5 flex items-center space-x-6 overflow-x-auto z-20 select-none text-[#1d1d1f]">
      {/* 1. Key Metrics Titleblock Grid */}
      <div className="flex items-stretch border border-[#1d1d1f] rounded-lg overflow-hidden bg-white/70 shadow-sm shrink-0">
        {/* Total Area */}
        <div className="px-4 py-2 border-r border-[#1d1d1f] flex flex-col justify-between min-w-[130px]">
          <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
            Total Living Area
          </span>
          <p className="font-serif text-2xl text-[#1d1d1f] mt-0.5 leading-none">
            {totalArea} <span className="font-mono text-[10px] text-[#1d1d1f]/60 uppercase">m²</span>
          </p>
        </div>

        {/* Room Count */}
        <div className="px-4 py-2 border-r border-[#1d1d1f] flex flex-col justify-between min-w-[110px]">
          <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
            Rooms Partitioned
          </span>
          <p className="font-serif text-2xl text-[#1d1d1f] mt-0.5 leading-none">
            {rooms.length} <span className="font-mono text-[10px] text-[#1d1d1f]/60 uppercase">spaces</span>
          </p>
        </div>

        {/* Exterior Shell */}
        <div className="px-4 py-2 border-r border-[#1d1d1f] flex flex-col justify-between min-w-[130px]">
          <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
            Building Envelope
          </span>
          <p className="font-serif text-2xl text-[#1d1d1f] mt-0.5 leading-none">
            {metadata.building_perimeter_m || 0} <span className="font-mono text-[10px] text-[#1d1d1f]/60 uppercase">m</span>
          </p>
        </div>

        {/* Architectural Plausibility Status */}
        <div className="px-4 py-2 flex flex-col justify-between min-w-[160px] bg-[#efebe2]/50">
          <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
            Plausibility Gate
          </span>
          <div className="flex items-center space-x-1.5 mt-0.5">
            {warningsCount === 0 ? (
              <>
                <CheckCircle className="w-3.5 h-3.5 text-emerald-700" />
                <span className="font-mono text-[10.5px] font-semibold text-emerald-800 uppercase tracking-wider">
                  BIM Validated
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-[#b8452a]" />
                <span className="font-mono text-[10px] font-semibold text-[#b8452a] uppercase tracking-wider">
                  {warningsCount} Area {warningsCount === 1 ? 'Notice' : 'Notices'}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 2. Room Breakdown Chips (Drafting Badges) */}
      <div className="flex-1 flex flex-col justify-center space-y-1.5 min-w-[280px]">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[9.5px] uppercase tracking-wider text-[#1d1d1f]/70 font-semibold">
            Extracted Spatial Registry ({rooms.length})
          </span>
          {frontWallId && (
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#b8452a]">
              Active Facade: Wall #{frontWallId}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2 overflow-x-auto pb-1 scrollbar-thin">
          {plausibilityResults.map((r, i) => (
            <div
              key={r.id || i}
              className={`shrink-0 px-2.5 py-1 rounded border text-[10px] font-mono uppercase tracking-wider flex items-center space-x-1.5 transition-all ${
                r.flag
                  ? 'bg-[#b8452a]/10 border-[#b8452a]/40 text-[#b8452a]'
                  : 'bg-white/80 border-[#1d1d1f]/20 text-[#1d1d1f]'
              }`}
              title={r.warning || `${r.room_type} (${r.surface_m2} m²)`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[#1d1d1f]/40"></span>
              <span className="font-semibold">{r.room_type}</span>
              <span className="text-[#1d1d1f]/60 font-normal">{r.surface_m2}m²</span>
            </div>
          ))}
        </div>
      </div>
    </footer>
  );
}
