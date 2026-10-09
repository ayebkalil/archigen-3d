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
    <div className="h-44 border-t border-slate-800 bg-slate-900/80 px-6 py-3 flex items-center space-x-6 overflow-x-auto z-20 select-none">
      {/* 1. Key Metrics Cards */}
      <div className="flex items-center space-x-3 shrink-0">
        {/* Total Area */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 min-w-[130px]">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs">
            <Maximize2 className="w-3.5 h-3.5 text-sky-400" />
            <span>Living Space</span>
          </div>
          <p className="text-xl font-bold text-white mt-1">
            {totalArea} <span className="text-xs font-normal text-slate-400">m²</span>
          </p>
        </div>

        {/* Room Count */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 min-w-[120px]">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs">
            <LayoutGrid className="w-3.5 h-3.5 text-indigo-400" />
            <span>Rooms</span>
          </div>
          <p className="text-xl font-bold text-white mt-1">
            {rooms.length} <span className="text-xs font-normal text-slate-400">spaces</span>
          </p>
        </div>

        {/* Exterior Perimeter */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 min-w-[140px]">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs">
            <Home className="w-3.5 h-3.5 text-amber-400" />
            <span>Exterior Shell</span>
          </div>
          <p className="text-xl font-bold text-white mt-1">
            {metadata.building_perimeter_m || 0} <span className="text-xs font-normal text-slate-400">m</span>
          </p>
        </div>

        {/* Architectural Plausibility Status */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 min-w-[170px]">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs">
            {warningsCount === 0 ? (
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            )}
            <span>Plausibility Gate</span>
          </div>
          <div className="flex items-center space-x-1.5 mt-1.5">
            {warningsCount === 0 ? (
              <>
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-semibold text-emerald-400">
                  Dimensions Plausible
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-semibold text-amber-400">
                  {warningsCount} Area {warningsCount === 1 ? 'Anomaly' : 'Anomalies'}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="h-24 w-px bg-slate-800 shrink-0" />

      {/* 2. Room Breakdown Chips */}
      <div className="flex-1 flex flex-col justify-center min-w-[320px]">
        <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
          <span className="font-semibold text-slate-200">Semantic Room Layout Breakdown</span>
          <span className="text-slate-500">
            Front Wall: <span className="font-mono text-amber-400">{frontWallId || 'Auto (South)'}</span>
          </span>
        </div>

        <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto pr-2">
          {plausibilityResults.map((room, idx) => {
            const pct = totalArea > 0 ? Math.round((room.surface_m2 / totalArea) * 100) : 0;
            return (
              <div
                key={idx}
                title={room.warning || `${room.room_type}: ${room.surface_m2} m²`}
                className={`border px-2.5 py-1 rounded-lg text-xs flex items-center space-x-2 shrink-0 transition-colors ${
                  room.flag
                    ? 'bg-amber-950/40 border-amber-500/50 text-amber-200'
                    : 'bg-slate-950/80 border-slate-800 text-slate-300'
                }`}
              >
                {room.flag && <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />}
                <span className="font-medium">{room.room_type}</span>
                <span className={`font-semibold ${room.flag ? 'text-amber-400' : 'text-sky-400'}`}>
                  {room.surface_m2} m²
                </span>
                <span className="text-[10px] text-slate-500">({pct}%)</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
