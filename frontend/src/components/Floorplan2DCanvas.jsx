import React, { useState } from 'react';
import {
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  CheckCircle2,
  Info,
  Ruler,
  X,
  Check,
  MousePointer,
  PlusCircle,
  Trash2,
  Tag,
  Layers,
  GripVertical,
  HelpCircle
} from 'lucide-react';

const ROOM_COLORS = {
  'Living Room': { fill: 'rgba(28, 78, 128, 0.20)', stroke: '#1c4e80' },
  'Living Room / Common': { fill: 'rgba(28, 78, 128, 0.20)', stroke: '#1c4e80' },
  'Bedroom': { fill: 'rgba(184, 69, 42, 0.18)', stroke: '#b8452a' },
  'Kitchen': { fill: 'rgba(196, 141, 63, 0.20)', stroke: '#c48d3f' },
  'Bathroom': { fill: 'rgba(42, 122, 110, 0.20)', stroke: '#2a7a6e' },
  'Bathroom / Storage': { fill: 'rgba(42, 122, 110, 0.20)', stroke: '#2a7a6e' },
  'Entry': { fill: 'rgba(110, 90, 70, 0.18)', stroke: '#6e5a46' },
  'Outdoor / Balcony': { fill: 'rgba(60, 110, 60, 0.18)', stroke: '#3c6e3c' },
  'Storage / Closet': { fill: 'rgba(90, 90, 100, 0.18)', stroke: '#5a5a64' },
  'Office': { fill: 'rgba(140, 75, 120, 0.18)', stroke: '#8c4b78' },
  'Default': { fill: 'rgba(60, 70, 90, 0.18)', stroke: '#3c465a' },
};

const ARCHITECTURAL_ROOM_TYPES = [
  'Living Room',
  'Bedroom',
  'Kitchen',
  'Bathroom',
  'Entry',
  'Office',
  'Outdoor / Balcony',
  'Storage / Closet'
];

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
  onUpdateLayout,
  isSplit = false,
  selectedRoom = null,
  onSelectRoom = null,
}) {
  const [showRooms, setShowRooms] = useState(true);
  const [showWalls, setShowWalls] = useState(true);
  const [showOpenings, setShowOpenings] = useState(true);
  const [hoveredWall, setHoveredWall] = useState(null);

  // Active CAD Tool: 'select' | 'add_door' | 'delete' | 'relabel'
  const [activeTool, setActiveTool] = useState('select');
  const [selectedRoomForRelabel, setSelectedRoomForRelabel] = useState(null);
  const [editFeedback, setEditFeedback] = useState(null);
  const [showNotif, setShowNotif] = useState(true);

  // Movable Floating Layer Dock state (for split view)
  const [dockPos, setDockPos] = useState({ x: 20, y: 18 });
  const [isDraggingDock, setIsDraggingDock] = useState(false);
  const dragStartRef = React.useRef({ mouseX: 0, mouseY: 0, posX: 0, posY: 0 });

  const handleStartDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingDock(true);
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      posX: dockPos.x,
      posY: dockPos.y,
    };
  };

  React.useEffect(() => {
    if (!isDraggingDock) return;
    const onMove = (e) => {
      const dx = e.clientX - dragStartRef.current.mouseX;
      const dy = dragStartRef.current.mouseY - e.clientY;
      setDockPos({
        x: Math.max(8, Math.min(window.innerWidth - 250, dragStartRef.current.posX + dx)),
        y: Math.max(8, Math.min(window.innerHeight - 150, dragStartRef.current.posY + dy)),
      });
    };
    const onUp = () => setIsDraggingDock(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isDraggingDock]);

  // 2-Point Calibration State
  const [calibPointA, setCalibPointA] = useState(null);
  const [calibPointB, setCalibPointB] = useState(null);
  const [knownDistanceM, setKnownDistanceM] = useState('0.90');

  const canvasSize = layout?.metadata?.canvas_size || [512, 512];
  const walls = layout?.walls || [];
  const rooms = layout?.rooms || [];
  const doors = layout?.doors || [];
  const windows = layout?.windows || [];

  const triggerFeedback = (msg) => {
    setEditFeedback(msg);
    setTimeout(() => setEditFeedback(null), 2500);
  };

  const handleCanvasClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    // 1. Calibration Tool
    if (isCalibrating) {
      if (!calibPointA) {
        setCalibPointA([x, y]);
      } else if (!calibPointB) {
        setCalibPointB([x, y]);
      } else {
        setCalibPointA([x, y]);
        setCalibPointB(null);
      }
      return;
    }

    // 2. Add Door Tool
    if (activeTool === 'add_door' && onUpdateLayout && layout) {
      const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
      const posM = [parseFloat((x * scaleMperPx).toFixed(2)), parseFloat((y * scaleMperPx).toFixed(2))];
      const newDoor = {
        id: `door_manual_${Date.now()}`,
        position: posM,
        position_meters: posM,
        width_m: 0.9,
        type: 'single'
      };
      const updatedLayout = {
        ...layout,
        doors: [...(layout.doors || []), newDoor],
        metadata: {
          ...layout.metadata,
          num_doors: (layout.doors?.length || 0) + 1
        }
      };
      onUpdateLayout(updatedLayout);
      triggerFeedback(`[+] Added door at (${posM[0]}m, ${posM[1]}m)`);
      return;
    }
  };

  const handleDeleteDoor = (idx, e) => {
    e.stopPropagation();
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedDoors = (layout.doors || []).filter((_, i) => i !== idx);
      onUpdateLayout({
        ...layout,
        doors: updatedDoors,
        metadata: { ...layout.metadata, num_doors: updatedDoors.length }
      });
      triggerFeedback('[-] Door deleted');
    }
  };

  const handleDeleteWindow = (idx, e) => {
    e.stopPropagation();
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedWindows = (layout.windows || []).filter((_, i) => i !== idx);
      onUpdateLayout({
        ...layout,
        windows: updatedWindows,
        metadata: { ...layout.metadata, num_windows: updatedWindows.length }
      });
      triggerFeedback('[-] Window deleted');
    }
  };

  const handleWallClick = (wall, e) => {
    e.stopPropagation();
    if (activeTool === 'delete' && onUpdateLayout && layout) {
      const updatedWalls = layout.walls.filter((w) => w.id !== wall.id);
      onUpdateLayout({
        ...layout,
        walls: updatedWalls,
        metadata: { ...layout.metadata, num_walls: updatedWalls.length }
      });
      triggerFeedback(`[-] Removed wall (${wall.id})`);
      return;
    }

    if (wall.is_exterior && onSelectFrontWall) {
      onSelectFrontWall(wall.id);
    }
  };

  const handleRoomClick = (room, e) => {
    e.stopPropagation();
    if (activeTool === 'relabel') {
      setSelectedRoomForRelabel(room);
    } else if (onSelectRoom) {
      onSelectRoom(room);
    }
  };

  const handleApplyRelabel = (newType) => {
    if (!selectedRoomForRelabel || !onUpdateLayout || !layout) return;
    const updatedRooms = layout.rooms.map((r) => {
      if (r.id === selectedRoomForRelabel.id) {
        return { ...r, room_type: newType };
      }
      return r;
    });
    onUpdateLayout({ ...layout, rooms: updatedRooms });
    triggerFeedback(`[✓] Relabeled room to "${newType}"`);
    setSelectedRoomForRelabel(null);
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
      triggerFeedback(`[✓] Scale calibrated to ${calculatedScale} m/px`);
    }
  };

  const handleCancelCalibration = () => {
    if (setIsCalibrating) setIsCalibrating(false);
    setCalibPointA(null);
    setCalibPointB(null);
  };

  // Cursor style based on active tool
  let canvasCursor = 'cursor-default';
  if (isCalibrating) canvasCursor = 'cursor-crosshair';
  else if (activeTool === 'add_door') canvasCursor = 'cursor-crosshair';
  else if (activeTool === 'delete') canvasCursor = 'cursor-pointer';
  else if (activeTool === 'relabel') canvasCursor = 'cursor-pointer';

  return (
    <div className="flex-1 flex flex-col h-full bg-[#efebe2] relative overflow-hidden border-r border-[#1d1d1f]/15 drafting-grid">
      {/* Top Drafting Toolbar */}
      <div className="h-12 border-b border-[#1d1d1f]/15 bg-[#f7f4ed]/90 backdrop-blur-md px-3 sm:px-4 flex items-center justify-between z-10 select-none overflow-x-auto">
        {/* Left: Title & Elements count */}
        <div className="flex items-center space-x-2.5 text-[#1d1d1f] shrink-0 mr-3">
          <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] whitespace-nowrap">
            2D BLUEPRINT
          </span>
          <span className="text-[#1d1d1f]/20">|</span>
          <span className="font-mono text-[9.5px] text-[#1d1d1f]/70 uppercase tracking-wider whitespace-nowrap">
            {walls.length}W · {rooms.length}R · {doors.length}D {!isSplit && `· ${windows.length}WIN`}
          </span>
        </div>

        {/* Center: CAD Interactive Edit Tools */}
        <div className="flex items-center bg-white/80 border border-[#1d1d1f]/20 rounded-full p-1 space-x-1 shadow-sm shrink-0">
          <button
            onClick={() => { setActiveTool('select'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Inspect / Select Front Wall"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'select' && !isCalibrating
                ? 'bg-[#1d1d1f] text-[#efebe2] shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <MousePointer className="w-3.5 h-3.5 shrink-0" />
            <span>SELECT</span>
          </button>

          <button
            onClick={() => { setActiveTool('add_door'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click on canvas or wall to insert a door"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'add_door'
                ? 'bg-[#f0500f] text-white shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <PlusCircle className="w-3.5 h-3.5 shrink-0" />
            <span>+ DOOR</span>
          </button>

          <button
            onClick={() => { setActiveTool('delete'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click doors, windows, or internal walls to delete them"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'delete'
                ? 'bg-rose-700 text-white shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <Trash2 className="w-3.5 h-3.5 shrink-0" />
            <span>DELETE</span>
          </button>

          <button
            onClick={() => { setActiveTool('relabel'); if (setIsCalibrating) setIsCalibrating(false); }}
            title="Click any room polygon or badge to change space type"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              activeTool === 'relabel'
                ? 'bg-[#c48d3f] text-white shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <Tag className="w-3.5 h-3.5 shrink-0" />
            <span>RELABEL</span>
          </button>

          <button
            onClick={() => {
              if (setIsCalibrating) setIsCalibrating(!isCalibrating);
              setActiveTool('select');
              setCalibPointA(null);
              setCalibPointB(null);
            }}
            title="2-Point Real-world Scale Calibration"
            className={`h-7 px-2.5 sm:px-3 rounded-full flex items-center space-x-1.5 transition-all font-stencil text-[10px] uppercase tracking-wider whitespace-nowrap shrink-0 ${
              isCalibrating
                ? 'bg-[#f0500f] text-white animate-pulse'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <Ruler className="w-3.5 h-3.5 shrink-0" />
            <span>SCALE</span>
          </button>
        </div>

        {/* Right: In 2D-only full view (!isSplit), keep layer toggles in the top bar! */}
        {!isSplit && (
          <div className="flex items-center space-x-1 font-stencil text-[10px] uppercase tracking-wider shrink-0">
            <button
              onClick={() => setShowRooms(!showRooms)}
              className={`h-7 px-3 rounded-full border transition-all whitespace-nowrap ${
                showRooms
                  ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                  : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
              }`}
            >
              ROOMS
            </button>
            <button
              onClick={() => setShowWalls(!showWalls)}
              className={`h-7 px-3 rounded-full border transition-all whitespace-nowrap ${
                showWalls
                  ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                  : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
              }`}
            >
              WALLS
            </button>
            <button
              onClick={() => setShowOpenings(!showOpenings)}
              className={`h-7 px-3 rounded-full border transition-all whitespace-nowrap ${
                showOpenings
                  ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                  : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
              }`}
            >
              DOORS
            </button>
          </div>
        )}
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex items-center justify-center p-6 overflow-auto relative">
        {/* Movable Floating Layer Visibility Dock (Only in 2D+3D Split Mode) */}
        {isSplit && (
          <div
            style={{ left: `${dockPos.x}px`, bottom: `${dockPos.y}px` }}
            className="absolute z-20 flex items-center bg-[#f7f4ed]/95 backdrop-blur-md border border-[#1d1d1f]/20 rounded-full px-2.5 py-1.5 shadow-xl select-none"
          >
            <div
              onMouseDown={handleStartDrag}
              className="flex items-center space-x-1 text-[#1d1d1f]/40 font-mono text-[9px] uppercase tracking-wider pr-2 border-r border-[#1d1d1f]/15 mr-1.5 cursor-grab active:cursor-grabbing hover:text-[#1d1d1f] shrink-0"
              title="Click and drag to move layer dock anywhere"
            >
              <GripVertical className="w-3.5 h-3.5 text-[#f0500f]" />
              <span className="font-semibold hidden sm:inline">LAYERS</span>
            </div>
            <div className="flex items-center space-x-1 font-stencil text-[9.5px] uppercase tracking-wider">
              <button
                onClick={() => setShowRooms(!showRooms)}
                className={`px-2.5 py-1 rounded-full border transition-all whitespace-nowrap shrink-0 ${
                  showRooms
                    ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                    : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
                }`}
              >
                ROOMS
              </button>
              <button
                onClick={() => setShowWalls(!showWalls)}
                className={`px-2.5 py-1 rounded-full border transition-all whitespace-nowrap shrink-0 ${
                  showWalls
                    ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                    : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
                }`}
              >
                WALLS
              </button>
              <button
                onClick={() => setShowOpenings(!showOpenings)}
                className={`px-2.5 py-1 rounded-full border transition-all whitespace-nowrap shrink-0 ${
                  showOpenings
                    ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                    : 'bg-white/60 text-[#1d1d1f]/50 border-[#1d1d1f]/20 hover:text-[#1d1d1f]'
                }`}
              >
                DOORS
              </button>
            </div>
          </div>
        )}
        {/* Real-time Feedback Banner */}
        {editFeedback && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-[#1d1d1f] text-[#efebe2] font-mono text-[10.5px] uppercase tracking-wider rounded-full shadow-xl z-30 flex items-center space-x-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>{editFeedback}</span>
          </div>
        )}

        <div
          onClick={handleCanvasClick}
          className={`relative rounded-xl overflow-hidden shadow-xl border border-[#1d1d1f]/25 bg-white ${canvasCursor} ${
            isCalibrating ? 'ring-2 ring-[#b8452a]' : activeTool !== 'select' ? 'ring-1 ring-[#1d1d1f]/40' : ''
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
                const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
                const pointsStr = room.polygon_meters
                  ? room.polygon_meters
                      .map((p) => `${p[0] / scaleMperPx},${p[1] / scaleMperPx}`)
                      .join(' ')
                  : '';
                const centroid = room.centroid_meters
                  ? [room.centroid_meters[0] / scaleMperPx, room.centroid_meters[1] / scaleMperPx]
                  : [0, 0];

                const isSelected = selectedRoom?.id === room.id;
                return (
                  <g key={`room-${idx}`} data-room-id={room.id} className="room-group">
                    <polygon
                      data-room-id={room.id}
                      points={pointsStr}
                      fill={isSelected ? 'rgba(2, 132, 199, 0.28)' : style.fill}
                      stroke={isSelected ? '#0284c7' : style.stroke}
                      strokeWidth={isSelected ? '2.5' : '1.5'}
                      className={`room-polygon transition-all cursor-pointer ${
                        activeTool === 'relabel' ? 'hover:stroke-purple-400 hover:fill-purple-500/30' : 'hover:stroke-sky-500'
                      }`}
                      onClick={(e) => handleRoomClick(room, e)}
                    />
                    {/* Centroid Tag */}
                    {centroid[0] > 0 && (
                      <g
                        data-room-id={room.id}
                        transform={`translate(${centroid[0]}, ${centroid[1]})`}
                        className="room-centroid-badge cursor-pointer select-none"
                        onClick={(e) => handleRoomClick(room, e)}
                      >
                        <rect
                          x="-50"
                          y="-14"
                          width="100"
                          height="28"
                          rx="4"
                          fill="rgba(247, 244, 237, 0.95)"
                          stroke={isSelected ? '#0284c7' : activeTool === 'relabel' ? '#b8452a' : style.stroke}
                          strokeWidth={isSelected ? '2' : '1.2'}
                        />
                        <text
                          x="0"
                          y="-2"
                          textAnchor="middle"
                          fill="#1d1d1f"
                          fontSize="9"
                          fontWeight="600"
                          fontFamily="'Geist', sans-serif"
                        >
                          {room.room_type}
                        </text>
                        <text
                          x="0"
                          y="9"
                          textAnchor="middle"
                          fill="#b8452a"
                          fontSize="8.5"
                          fontWeight="bold"
                          fontFamily="'Geist Mono', monospace"
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

                const pts = wall.polygon_px || [];
                const pointsStr = pts.map((p) => `${p[0]},${p[1]}`).join(' ');

                let strokeColor = '#1d1d1f'; // Interior drafting ink
                let fillColor = 'rgba(29, 29, 31, 0.20)';

                if (isExterior) {
                  strokeColor = '#b8452a'; // Exterior terracotta
                  fillColor = 'rgba(184, 69, 42, 0.35)';
                }
                if (isSelected) {
                  strokeColor = '#c48d3f'; // Front wall architectural amber
                  fillColor = 'rgba(196, 141, 63, 0.50)';
                }

                if (activeTool === 'delete') {
                  strokeColor = '#f43f5e';
                  fillColor = 'rgba(244, 63, 94, 0.25)';
                }

                return (
                  <polygon
                    key={wall.id}
                    points={pointsStr}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? '3' : isExterior ? '2' : '1.5'}
                    className={`cursor-pointer transition-all ${
                      activeTool === 'delete' ? 'hover:stroke-rose-400 hover:fill-rose-500/40' : isExterior ? 'hover:stroke-amber-400' : ''
                    }`}
                    onMouseEnter={() => setHoveredWall(wall.id)}
                    onMouseLeave={() => setHoveredWall(null)}
                    onClick={(e) => handleWallClick(wall, e)}
                  />
                );
              })}

            {/* 3. Doors Markers */}
            {showOpenings &&
              doors.map((door, idx) => {
                const p = door.position_meters || [0, 0];
                const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <circle
                    key={`door-${idx}`}
                    cx={cx}
                    cy={cy}
                    r="5"
                    fill={activeTool === 'delete' ? '#ef4444' : '#10b981'}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    className={`cursor-pointer transition-transform hover:scale-125`}
                    onClick={(e) => handleDeleteDoor(idx, e)}
                  />
                );
              })}

            {/* 4. Windows Markers */}
            {showOpenings &&
              windows.map((win, idx) => {
                const p = win.position_meters || [0, 0];
                const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
                const cx = p[0] / scaleMperPx;
                const cy = p[1] / scaleMperPx;
                return (
                  <rect
                    key={`win-${idx}`}
                    x={cx - 4}
                    y={cy - 4}
                    width="8"
                    height="8"
                    fill={activeTool === 'delete' ? '#ef4444' : '#0ea5e9'}
                    stroke="#ffffff"
                    strokeWidth="1"
                    rx="1"
                    className={`cursor-pointer transition-transform hover:scale-125`}
                    onClick={(e) => handleDeleteWindow(idx, e)}
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

          {/* Unresolved Rooms Domain Warning */}
          {layout && rooms.length === 0 && (
            <div className="absolute inset-x-6 top-6 bg-slate-950/95 border border-amber-500/50 rounded-xl p-3 text-center pointer-events-none shadow-2xl backdrop-blur-sm z-20">
              <p className="text-xs font-semibold text-amber-400">No Rooms Detected (Plan style may not be supported)</p>
              <p className="text-[10px] text-slate-400 mt-0.5">The neural network could not close topological loops around rooms on this drafting format.</p>
            </div>
          )}
        </div>

        {/* Room Relabeling Modal Popover */}
        {selectedRoomForRelabel && (
          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-40 p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 max-w-sm w-full shadow-2xl">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <Tag className="w-4 h-4 text-purple-400" />
                  <span className="font-semibold text-sm text-slate-100">Relabel Space</span>
                </div>
                <button
                  onClick={() => setSelectedRoomForRelabel(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <p className="text-xs text-slate-400 mt-2 mb-4">
                Current type: <span className="text-purple-300 font-semibold">{selectedRoomForRelabel.room_type}</span> ({selectedRoomForRelabel.surface_m2} m²). Select new architectural classification:
              </p>

              <div className="grid grid-cols-2 gap-2">
                {ARCHITECTURAL_ROOM_TYPES.map((type) => (
                  <button
                    key={type}
                    onClick={() => handleApplyRelabel(type)}
                    className={`px-3 py-2 rounded-lg text-xs font-medium text-left transition-colors border ${
                      selectedRoomForRelabel.room_type === type
                        ? 'bg-purple-500/20 border-purple-500/60 text-purple-200 font-semibold'
                        : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Floating Calibration Controls Card */}
        {/* 2-Point Calibration Tool Box */}
        {isCalibrating && (
          <div className="absolute top-4 right-6 bg-[#f7f4ed]/95 backdrop-blur-md border border-[#b8452a]/40 rounded-xl p-4 text-xs text-[#1d1d1f] shadow-xl w-80 z-30 font-sans">
            <div className="flex items-center justify-between font-semibold text-[#1d1d1f] mb-2 font-mono uppercase tracking-wider text-[11px]">
              <div className="flex items-center space-x-1.5">
                <Ruler className="w-4 h-4 text-[#b8452a]" />
                <span className="text-[#b8452a]">2-Point Scale Calibration</span>
              </div>
              <button onClick={handleCancelCalibration} className="text-[#1d1d1f]/50 hover:text-[#1d1d1f]">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <p className="text-[11px] text-[#1d1d1f]/70 mb-3 leading-relaxed">
              {!calibPointA
                ? "Click Point 1 on the canvas (e.g. one side of a doorway or exterior wall)."
                : !calibPointB
                ? "Click Point 2 (the other side of the doorway or wall)."
                : `Distance between points: ${Math.round(calibPixelDist)} pixels.`}
            </p>

            {calibPointA && calibPointB && (
              <div className="space-y-3 pt-2 border-t border-[#1d1d1f]/15 font-mono">
                <div>
                  <label className="text-[9.5px] uppercase font-semibold text-[#1d1d1f]/70 block mb-1">
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
                      className="bg-white border border-[#1d1d1f]/30 rounded px-2.5 py-1 text-[#1d1d1f] font-mono text-sm w-28 focus:border-[#b8452a] focus:outline-none"
                    />
                    <span className="text-[#1d1d1f]/60 text-xs">meters</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  <button
                    onClick={() => setKnownDistanceM('0.90')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-white hover:bg-[#1d1d1f] hover:text-[#efebe2] text-[#1d1d1f] border border-[#1d1d1f]/20"
                  >
                    Door (0.9m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('1.80')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-white hover:bg-[#1d1d1f] hover:text-[#efebe2] text-[#1d1d1f] border border-[#1d1d1f]/20"
                  >
                    Double Door (1.8m)
                  </button>
                  <button
                    onClick={() => setKnownDistanceM('3.50')}
                    className="text-[10px] px-2 py-0.5 rounded-full bg-white hover:bg-[#1d1d1f] hover:text-[#efebe2] text-[#1d1d1f] border border-[#1d1d1f]/20"
                  >
                    Wall (3.5m)
                  </button>
                </div>

                <div className="bg-[#efebe2] p-2.5 rounded border border-[#1d1d1f]/15 flex justify-between items-center text-xs">
                  <span className="text-[#1d1d1f]/70 uppercase text-[10px]">Calibrated Ratio:</span>
                  <span className="font-mono font-bold text-[#b8452a]">{calculatedScale} m/px</span>
                </div>

                <div className="flex space-x-2 pt-1">
                  <button
                    onClick={handleApplyCalibration}
                    className="flex-1 bg-[#b8452a] hover:bg-[#96351e] text-white font-mono uppercase tracking-wider text-[10px] font-semibold py-1.5 px-3 rounded-full flex items-center justify-center space-x-1.5 transition-all shadow-sm"
                  >
                    <Check className="w-3.5 h-3.5" />
                    <span>Apply Calibration</span>
                  </button>
                  <button
                    onClick={() => { setCalibPointA(null); setCalibPointB(null); }}
                    className="bg-white hover:bg-[#1d1d1f]/10 text-[#1d1d1f] border border-[#1d1d1f]/20 font-mono uppercase tracking-wider text-[10px] px-3 py-1.5 rounded-full transition-all"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Helper Box / Drafting Tip */}
        {showNotif ? (
          <div
            className={`absolute z-10 bg-[#f7f4ed]/95 backdrop-blur-md border border-[#1d1d1f]/20 rounded-xl p-3 text-xs text-[#1d1d1f] shadow-lg max-w-sm font-sans transition-all ${
              isSplit ? 'bottom-16 left-5' : 'bottom-4 left-6'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-1.5 font-mono uppercase tracking-wider text-[10px] font-semibold text-[#1d1d1f]">
                <Info className="w-3.5 h-3.5 text-[#f0500f]" />
                <span>INTERACTIVE DRAFTING DESK</span>
              </div>
              <button
                onClick={() => setShowNotif(false)}
                className="text-[#1d1d1f]/40 hover:text-[#1d1d1f] text-sm leading-none font-bold px-1 ml-2 cursor-pointer"
                title="Dismiss tip"
              >
                ×
              </button>
            </div>
            <p className="font-mono text-[9.5px] text-[#1d1d1f]/70 leading-relaxed">
              {activeTool === 'select' && "Click any exterior terracotta wall to designate it as the front wall for facade synthesis."}
              {activeTool === 'add_door' && "Click anywhere on the plan to insert a door. The Three.js 3D model re-extrudes instantly."}
              {activeTool === 'delete' && "Click any door, window, or wall to delete it from the architectural model."}
              {activeTool === 'relabel' && "Click any room polygon or badge to change its space classification."}
            </p>
          </div>
        ) : (
          <button
            onClick={() => setShowNotif(true)}
            className="absolute bottom-4 right-4 z-10 p-2 rounded-full bg-[#f7f4ed]/95 border border-[#1d1d1f]/20 shadow-md hover:bg-white text-[#1d1d1f]/70 hover:text-[#f0500f] transition-all cursor-pointer"
            title="Show drafting tips"
          >
            <HelpCircle className="w-4 h-4 text-[#f0500f]" />
          </button>
        )}
      </div>
    </div>
  );
}
