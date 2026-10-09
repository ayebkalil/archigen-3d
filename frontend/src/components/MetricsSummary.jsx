import React from 'react';
import { Home, ShieldCheck, Maximize2, LayoutGrid, CheckCircle } from 'lucide-react';

export default function MetricsSummary({ layout, frontWallId }) {
  if (!layout) return null;

  const metadata = layout.metadata || {};
  const rooms = layout.rooms || [];
  const walls = layout.walls || [];
  const totalArea = metadata.total_surface_m2 || 0;
  const isShellValid = metadata.is_building_shell_valid ?? true;

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

        {/* Watertight Status */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 min-w-[150px]">
          <div className="flex items-center space-x-1.5 text-slate-400 text-xs">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>3D Validity</span>
          </div>
          <div className="flex items-center space-x-1 mt-1.5">
            <CheckCircle className={`w-4 h-4 ${isShellValid ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span className={`text-xs font-semibold ${isShellValid ? 'text-emerald-400' : 'text-amber-400'}`}>
              {isShellValid ? 'Watertight Shell' : 'Complex Boundary'}
            </span>
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
          {rooms.map((room, idx) => {
            const pct = totalArea > 0 ? Math.round((room.surface_m2 / totalArea) * 100) : 0;
            return (
              <div
                key={idx}
                className="bg-slate-950/80 border border-slate-800 px-2.5 py-1 rounded-lg text-xs flex items-center space-x-2 shrink-0"
              >
                <span className="font-medium text-slate-300">{room.room_type}</span>
                <span className="font-semibold text-sky-400">{room.surface_m2} m²</span>
                <span className="text-[10px] text-slate-500">({pct}%)</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
