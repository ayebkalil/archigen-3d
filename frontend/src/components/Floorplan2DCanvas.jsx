import React, { useState } from 'react';
import { Eye, EyeOff, ZoomIn, ZoomOut, CheckCircle2, Info } from 'lucide-react';

const ROOM_COLORS = {
  'Living Room': { fill: 'rgba(56, 189, 248, 0.25)', stroke: '#38bdf8' },
  'Living Room / Common': { fill: 'rgba(56, 189, 248, 0.25)', stroke: '#38bdf8' },
  'Bedroom': { fill: 'rgba(168, 85, 247, 0.25)', stroke: '#a855f7' },
  'Kitchen': { fill: 'rgba(251, 146, 60, 0.25)', stroke: '#fb923c' },
  'Bathroom': { fill: 'rgba(45, 212, 191, 0.25)', stroke: '#2dd4bf' },
  'Bathroom / Storage': { fill: 'rgba(45, 212, 191, 0.25)', stroke: '#2dd4bf' },
  'Entry': { fill: 'rgba(234, 179, 8, 0.25)', stroke: '#eab308' },
  'Outdoor / Balcony': { fill: 'rgba(74, 222, 128, 0.25)', stroke: '#4ade80' },
  'Storage / Closet': { fill: 'rgba(148, 163, 184, 0.25)', stroke: '#94a3b8' },
  'Default': { fill: 'rgba(99, 102, 241, 0.2)', stroke: '#6366f1' },
};

function getRoomStyle(type) {
  for (const key of Object.keys(ROOM_COLORS)) {
    if (type && type.toLowerCase().includes(key.toLowerCase())) {
      return ROOM_COLORS[key];
    }
  }
  return ROOM_COLORS['Default'];
}

export default function Floorplan2DCanvas({
  imageUrl,
  layout,
  frontWallId,
  onSelectFrontWall,
}) {
  const [showRooms, setShowRooms] = useState(true);
  const [showWalls, setShowWalls] = useState(true);
  const [showOpenings, setShowOpenings] = useState(true);
  const [hoveredWall, setHoveredWall] = useState(null);

  const canvasSize = layout?.metadata?.canvas_size || [512, 512];
  const walls = layout?.walls || [];
  const rooms = layout?.rooms || [];
  const doors = layout?.doors || [];
  const windows = layout?.windows || [];

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 relative overflow-hidden border-r border-slate-800">
      {/* Top Toolbar */}
      <div className="h-12 border-b border-slate-800 bg-slate-900/60 px-4 flex items-center justify-between z-10">
        <div className="flex items-center space-x-2 text-xs text-slate-300">
          <span className="font-semibold text-slate-100">2D Semantic Plan</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">
            {walls.length} walls · {rooms.length} rooms · {doors.length} doors · {windows.length} windows
          </span>
        </div>

        {/* Visibility Toggles */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            onClick={() => setShowRooms(!showRooms)}
            className={`px-2.5 py-1 rounded border transition-colors flex items-center space-x-1.5 ${
              showRooms
                ? 'bg-sky-500/10 border-sky-500/30 text-sky-300'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            <span>Rooms</span>
          </button>
          <button
            onClick={() => setShowWalls(!showWalls)}
            className={`px-2.5 py-1 rounded border transition-colors flex items-center space-x-1.5 ${
              showWalls
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            <span>Walls</span>
          </button>
          <button
            onClick={() => setShowOpenings(!showOpenings)}
            className={`px-2.5 py-1 rounded border transition-colors flex items-center space-x-1.5 ${
              showOpenings
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
          >
            <span>Openings</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex items-center justify-center p-6 overflow-auto relative">
        <div
          className="relative rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-900"
          style={{ width: `${canvasSize[0]}px`, height: `${canvasSize[1]}px` }}
        >
          {/* Base Image */}
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Floorplan"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-60"
            />
          )}

          {/* SVG Vector Overlays */}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox={`0 0 ${canvasSize[0]} ${canvasSize[1]}`}
          >
            {/* 1. Room Polygons */}
            {showRooms &&
              rooms.map((room, idx) => {
                const style = getRoomStyle(room.room_type);
                // Convert meters back to pixels for SVG rendering if polygon_px is available or scale
                const scaleMperPx = layout.metadata.scale_meters_per_pixel || 0.03;
                const pointsStr = room.polygon_meters
                  ? room.polygon_meters
                      .map((p) => `${p[0] / scaleMperPx},${p[1] / scaleMperPx}`)
                      .join(' ')
                  : '';
                const centroid = room.centroid_meters
                  ? [room.centroid_meters[0] / scaleMperPx, room.centroid_meters[1] / scaleMperPx]
                  : [0, 0];

                return (
                  <g key={`room-${idx}`}>
                    <polygon
                      points={pointsStr}
                      fill={style.fill}
                      stroke={style.stroke}
                      strokeWidth="1.5"
                      className="transition-all hover:opacity-80"
                    />
                    {/* Centroid Tag */}
                    {centroid[0] > 0 && (
                      <g
                        transform={`translate(${centroid[0]}, ${centroid[1]})`}
                        className="pointer-events-none select-none"
                      >
                        <rect
                          x="-50"
                          y="-14"
                          width="100"
                          height="28"
                          rx="4"
                          fill="rgba(15, 23, 42, 0.85)"
                          stroke={style.stroke}
                          strokeWidth="1"
                        />
                        <text
                          x="0"
                          y="-2"
                          textAnchor="middle"
                          fill="#f8fafc"
                          fontSize="9"
                          fontWeight="600"
                        >
                          {room.room_type}
                        </text>
                        <text
                          x="0"
                          y="9"
                          textAnchor="middle"
                          fill="#38bdf8"
                          fontSize="8"
                          fontWeight="bold"
                        >
                          {room.surface_m2} m²
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}

            {/* 2. Walls */}
            {showWalls &&
              walls.map((wall) => {
                const isSelected = wall.id === frontWallId;
                const isExterior = wall.is_exterior;
                const isHovered = wall.id === hoveredWall;

                // Points in pixels
                const pts = wall.polygon_px || [];
                const pointsStr = pts.map((p) => `${p[0]},${p[1]}`).join(' ');

                let strokeColor = '#64748b'; // Interior default
                let fillColor = 'rgba(100, 116, 139, 0.3)';

                if (isExterior) {
                  strokeColor = '#38bdf8'; // Exterior cyan
                  fillColor = 'rgba(56, 189, 248, 0.35)';
                }
                if (isSelected) {
                  strokeColor = '#f59e0b'; // Front wall amber
                  fillColor = 'rgba(245, 158, 11, 0.5)';
                }

                return (
                  <polygon
                    key={wall.id}
                    points={pointsStr}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? '3' : isExterior ? '2' : '1.5'}
                    className={`cursor-pointer transition-all ${
                      isExterior ? 'hover:stroke-amber-400' : ''
                    }`}
                    onMouseEnter={() => setHoveredWall(wall.id)}
                    onMouseLeave={() => setHoveredWall(null)}
                    onClick={() => {
                      if (isExterior && onSelectFrontWall) {
                        onSelectFrontWall(wall.id);
                      }
                    }}
                  />
                );
              })}

            {/* 3. Doors & Windows Markers */}
            {showOpenings &&
              doors.map((door, idx) => {
                const p = door.position_meters || [0, 0];
                const scaleMperPx = layout.metadata.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <circle
                    key={`door-${idx}`}
                    cx={cx}
                    cy={cy}
                    r="4"
                    fill="#10b981"
                    stroke="#047857"
                    strokeWidth="1.5"
                  />
                );
              })}

            {showOpenings &&
              windows.map((win, idx) => {
                const p = win.position_meters || [0, 0];
                const scaleMperPx = layout.metadata.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <rect
                    key={`win-${idx}`}
                    x={cx - 4}
                    y={cy - 4}
                    width="8"
                    height="8"
                    fill="#0ea5e9"
                    stroke="#0284c7"
                    strokeWidth="1"
                    rx="1"
                  />
                );
              })}
          </svg>
        </div>

        {/* Wall Selection Helper Box */}
        <div className="absolute bottom-4 left-6 bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-3 text-xs text-slate-300 shadow-xl max-w-sm">
          <div className="flex items-center space-x-1.5 font-semibold text-slate-100 mb-1">
            <Info className="w-3.5 h-3.5 text-sky-400" />
            <span>Interactive Facade Wall Selection</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Cyan lines represent <span className="text-sky-300 font-medium">exterior walls</span>. Click any exterior wall to designate it as the front wall for facade synthesis (highlighted in <span className="text-amber-400 font-medium">amber</span>).
          </p>
          {frontWallId && (
            <div className="mt-2 pt-2 border-t border-slate-800 flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Active Front Wall:</span>
              <span className="font-mono text-amber-400 font-bold">{frontWallId}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
