import React, { useState } from 'react';
import { Eye, EyeOff, ZoomIn, ZoomOut, CheckCircle2, Info, Ruler, X, Check } from 'lucide-react';

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
  onCalibrateScale,
  isCalibrating,
  setIsCalibrating,
}) {
  const [showRooms, setShowRooms] = useState(true);
  const [showWalls, setShowWalls] = useState(true);
  const [showOpenings, setShowOpenings] = useState(true);
  const [hoveredWall, setHoveredWall] = useState(null);

  // 2-Point Calibration State
  const [calibPointA, setCalibPointA] = useState(null);
  const [calibPointB, setCalibPointB] = useState(null);
  const [knownDistanceM, setKnownDistanceM] = useState('0.90');

  const canvasSize = layout?.metadata?.canvas_size || [512, 512];
  const walls = layout?.walls || [];
  const rooms = layout?.rooms || [];
  const doors = layout?.doors || [];
  const windows = layout?.windows || [];

  const handleCanvasClick = (e) => {
    if (!isCalibrating) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    if (!calibPointA) {
      setCalibPointA([x, y]);
    } else if (!calibPointB) {
      setCalibPointB([x, y]);
    } else {
      // If both were set, restart with new point A
      setCalibPointA([x, y]);
      setCalibPointB(null);
    }
  };

  const calibPixelDist =
    calibPointA && calibPointB
      ? Math.hypot(calibPointB[0] - calibPointA[0], calibPointB[1] - calibPointA[1])
      : 0;

  const calculatedScale =
    calibPixelDist > 0 && parseFloat(knownDistanceM) > 0
      ? (parseFloat(knownDistanceM) / calibPixelDist).toFixed(4)
      : null;

  const handleApplyCalibration = () => {
    if (calculatedScale && onCalibrateScale) {
      onCalibrateScale(parseFloat(calculatedScale));
      setIsCalibrating(false);
      setCalibPointA(null);
      setCalibPointB(null);
    }
  };

  const handleCancelCalibration = () => {
    if (setIsCalibrating) setIsCalibrating(false);
    setCalibPointA(null);
    setCalibPointB(null);
  };

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

        {/* Visibility Toggles & Calibration Button */}
        <div className="flex items-center space-x-2 text-xs">
          {/* 2-Point Calibration Toggle */}
          <button
            onClick={() => {
              if (setIsCalibrating) setIsCalibrating(!isCalibrating);
              setCalibPointA(null);
              setCalibPointB(null);
            }}
            className={`px-3 py-1 rounded border font-medium transition-colors flex items-center space-x-1.5 ${
              isCalibrating
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-300 animate-pulse'
                : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-white hover:border-slate-600'
            }`}
          >
            <Ruler className="w-3.5 h-3.5 text-amber-400" />
            <span>{isCalibrating ? 'Exit Measurement' : 'Calibrate Scale'}</span>
          </button>

          <span className="text-slate-700">|</span>

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
          onClick={handleCanvasClick}
          className={`relative rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-900 ${
            isCalibrating ? 'cursor-crosshair ring-2 ring-amber-500/50' : ''
          }`}
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

            {/* 4. Doors & Windows Markers */}
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

            {/* 5. Measurement / Calibration Line */}
            {isCalibrating && (
              <g>
                {calibPointA && calibPointB && (
                  <>
                    <line
                      x1={calibPointA[0]}
                      y1={calibPointA[1]}
                      x2={calibPointB[0]}
                      y2={calibPointB[1]}
                      stroke="#f59e0b"
                      strokeWidth="2.5"
                      strokeDasharray="6,4"
                    />
                    <text
                      x={(calibPointA[0] + calibPointB[0]) / 2}
                      y={(calibPointA[1] + calibPointB[1]) / 2 - 8}
                      fill="#f59e0b"
                      fontSize="11"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      {Math.round(calibPixelDist)} px
                    </text>
                  </>
                )}
                {calibPointA && (
                  <g transform={`translate(${calibPointA[0]}, ${calibPointA[1]})`}>
                    <circle r="7" fill="#f59e0b" stroke="#ffffff" strokeWidth="2" />
                    <text x="10" y="4" fill="#f59e0b" fontSize="10" fontWeight="bold">Point A</text>
                  </g>
                )}
                {calibPointB && (
                  <g transform={`translate(${calibPointB[0]}, ${calibPointB[1]})`}>
                    <circle r="7" fill="#10b981" stroke="#ffffff" strokeWidth="2" />
                    <text x="10" y="4" fill="#10b981" fontSize="10" fontWeight="bold">Point B</text>
                  </g>
                )}
              </g>
            )}
          </svg>
        </div>

        {/* Floating Calibration Controls Card */}
        {isCalibrating && (
          <div className="absolute top-4 right-6 bg-slate-900/95 backdrop-blur-md border border-amber-500/40 rounded-xl p-4 text-xs text-slate-200 shadow-2xl w-80 z-30">
            <div className="flex items-center justify-between font-semibold text-slate-100 mb-2">
              <div className="flex items-center space-x-1.5">
                <Ruler className="w-4 h-4 text-amber-400" />
                <span className="text-amber-300">2-Point Scale Calibration</span>
              </div>
              <button onClick={handleCancelCalibration} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
              {!calibPointA
                ? "Click Point 1 on the canvas (e.g. one side of a doorway or exterior wall)."
                : !calibPointB
                ? "Click Point 2 (the other side of the doorway or wall)."
                : `Distance between points: ${Math.round(calibPixelDist)} pixels.`}
            </p>

            {calibPointA && calibPointB && (
              <div className="space-y-3 pt-2 border-t border-slate-800">
                <div>
                  <label className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">
                    Known Distance in Real World (Meters)
                  </label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="number"
                      step="0.05"
                      min="0.1"
                      max="30"
                      value={knownDistanceM}
                      onChange={(e) => setKnownDistanceM(e.target.value)}
                      className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-slate-100 font-mono text-sm w-28 focus:border-amber-400 focus:outline-none"
                    />
                    <span className="text-slate-400 text-xs">meters</span>
                  </div>
                </div>

                {/* Quick Presets */}
                <div className="flex flex-wrap gap-1.5">
                  <button
                    onClick={() => setKnownDistanceM('0.90')}
                    className="text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                  >
                    Door (0.9m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('1.80')}
                    className="text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                  >
                    Double Door (1.8m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('3.50')}
                    className="text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                  >
                    Wall (3.5m)
                  </button>
                </div>

                {/* Calculated Result */}
                <div className="bg-slate-950 p-2.5 rounded border border-slate-800 flex justify-between items-center text-xs">
                  <span className="text-slate-400">Calibrated Ratio:</span>
                  <span className="font-mono font-bold text-amber-400">{calculatedScale} m/px</span>
                </div>

                <div className="flex space-x-2 pt-1">
                  <button
                    onClick={handleApplyCalibration}
                    className="flex-1 bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold py-1.5 px-3 rounded text-xs flex items-center justify-center space-x-1.5 transition-colors"
                  >
                    <Check className="w-3.5 h-3.5" />
                    <span>Apply Calibration</span>
                  </button>
                  <button
                    onClick={() => { setCalibPointA(null); setCalibPointB(null); }}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded text-xs transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

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
