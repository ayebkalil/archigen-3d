import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ControlPanel from './components/ControlPanel';
import Floorplan2DCanvas from './components/Floorplan2DCanvas';
import Viewer3D from './components/Viewer3D';
import MetricsSummary from './components/MetricsSummary';
import RoomDesignPanel from './components/RoomDesignPanel';

export default function App() {
  const [currentView, setCurrentView] = useState(() => {
    return window.location.hash === '#studio' ? 'studio' : 'showcase';
  });
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [transitionTarget, setTransitionTarget] = useState('studio');
  const [activeTab, setActiveTab] = useState('split');
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('/sample_floorplan.png');
  const [scale, setScale] = useState(0.03);
  const [numFloors, setNumFloors] = useState(1);
  const [selectedStyle, setSelectedStyle] = useState('Modern Minimalist');
  const [frontWallId, setFrontWallId] = useState(null);
  const [facadeTextureUrl, setFacadeTextureUrl] = useState(null);

  const [layout, setLayout] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingFacade, setIsGeneratingFacade] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [health, setHealth] = useState(null);
  const [latency, setLatency] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Room Design State (Stage 3)
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [furnishingData, setFurnishingData] = useState({});
  const [isDesigning, setIsDesigning] = useState(false);

  // 1. Health check & Initial Demo load
  useEffect(() => {
    axios
      .get('/health')
      .then((res) => setHealth(res.data))
      .catch((err) => console.log('Backend not reachable yet:', err));

    // Automatically analyze default sample floorplan on initial startup
    loadDefaultSample();

    // Listen for cross-frame messages from the Casa Serra Hero Showcase
    const handleHash = () => {
      const nextView = window.location.hash === '#studio' ? 'studio' : 'showcase';
      if (nextView !== currentView) {
        triggerTransition(nextView);
      }
    };
    const handleMsg = (e) => {
      if (e.data && e.data.type === 'LAUNCH_STUDIO') {
        triggerTransition('studio');
      }
    };
    window.addEventListener('hashchange', handleHash);
    window.addEventListener('message', handleMsg);
    return () => {
      window.removeEventListener('hashchange', handleHash);
      window.removeEventListener('message', handleMsg);
    };
  }, [currentView]);

  const triggerTransition = (targetView) => {
    setIsTransitioning(true);
    setTransitionTarget(targetView);
    // Smoothly flip underlying view halfway through when paper has fully covered viewport
    setTimeout(() => {
      setCurrentView(targetView);
      window.location.hash = targetView === 'studio' ? '#studio' : '#showcase';
    }, 2200);
    // Complete deliberate, majestic unroll across full screen
    setTimeout(() => {
      setIsTransitioning(false);
    }, 3800);
  };

  const loadDefaultSample = async () => {
    try {
      setIsAnalyzing(true);
      const urlParams = new URLSearchParams(window.location.search);
      const sampleParam = urlParams.get('sample');
      const samplePath = sampleParam === 'fresh_12' ? '/fresh_12.jpg' : '/sample_floorplan.png';
      const sampleMime = sampleParam === 'fresh_12' ? 'image/jpeg' : 'image/png';
      const res = await fetch(samplePath);
      const blob = await res.blob();
      const sampleFile = new File([blob], samplePath.replace('/', ''), { type: sampleMime });
      setFile(sampleFile);
      setImageUrl(samplePath);
      await executeAnalysis(sampleFile, scale, numFloors, null);
    } catch (err) {
      console.error('Failed to load default sample:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = (uploadedFile) => {
    setFile(uploadedFile);
    const url = URL.createObjectURL(uploadedFile);
    setImageUrl(url);
    setFrontWallId(null);
    setFacadeTextureUrl(null);
    executeAnalysis(uploadedFile, scale, numFloors, null);
  };

  const executeAnalysis = async (fileObj, curScale, curFloors, wallId) => {
    if (!fileObj) return;
    setIsAnalyzing(true);
    setErrorMsg(null);
    const startTime = performance.now();

    const formData = new FormData();
    formData.append('file', fileObj);
    if (curScale) formData.append('scale_override', curScale.toString());
    formData.append('num_floors', curFloors.toString());
    if (wallId) formData.append('front_wall_id', wallId);

    try {
      const response = await axios.post('/predict/floorplan', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      const duration = Math.round(performance.now() - startTime);

      // Read backend measured latency or fallback to client roundtrip
      const headerLatency = response.headers['x-inference-latency-ms'];
      setLatency(headerLatency ? parseFloat(headerLatency).toFixed(1) : duration);

      // Reconstruct layout structure
      const parsedLayout = {
        metadata: {
          canvas_size: [512, 512],
          scale_meters_per_pixel: curScale,
          total_surface_m2: data.total_area_m2,
          is_building_shell_valid: true,
          building_perimeter_m: Math.round(
            data.walls.reduce((acc, w) => acc + (w.polygon_meters?.length || 0) * 0.8, 0)
          ),
          num_walls: data.walls.length,
          num_rooms: data.rooms.length,
        },
        walls: data.walls,
        rooms: data.rooms.map((r, i) => ({
          id: `room_${i}`,
          room_type: r.room_type,
          surface_m2: r.surface_m2,
          polygon_meters: r.polygon_points,
          centroid_meters: r.polygon_points?.[0] || [0, 0],
        })),
        doors: data.doors || [],
        windows: data.windows || [],
      };

      setLayout(parsedLayout);

      if (data.rooms.length === 0) {
        setErrorMsg('No rooms detected. This plan style may not be supported (e.g. dense CAD dimension text, low-contrast raster, or non-standard drafting symbols).');
      }

      // Find Front Wall & its Facade Texture
      const frontWall = data.walls.find((w) => w.is_front) || data.walls.find((w) => w.is_exterior);
      if (frontWall) {
        setFrontWallId(frontWall.id);
        if (frontWall.facade_texture?.texture_base64) {
          setFacadeTextureUrl(frontWall.facade_texture.texture_base64);
        }
      }
    } catch (err) {
      console.error('Inference error:', err);
      setErrorMsg(err.response?.data?.detail || 'Floorplan analysis failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectFrontWall = (newWallId) => {
    setFrontWallId(newWallId);
    if (file) {
      executeAnalysis(file, scale, numFloors, newWallId);
    }
  };

  const handleCalibrateScale = (newScale) => {
    setScale(newScale);
    if (file) {
      executeAnalysis(file, newScale, numFloors, frontWallId);
    }
  };

  const handleGenerateFacade = () => {
    if (file) {
      setIsGeneratingFacade(true);
      executeAnalysis(file, scale, numFloors, frontWallId).finally(() => {
        setIsGeneratingFacade(false);
      });
    }
  };

  // --- Room Design Engine Handlers (Stage 3) ---
  const handleSelectRoom = (room) => {
    setSelectedRoom(room);
  };

  const handleGenerateDesign = async (room, preset = 'contemporary_minimal', seed = 42) => {
    if (!room) return;
    setIsDesigning(true);
    try {
      const payload = {
        room_id: room.id,
        room_type: room.room_type,
        polygon_meters: room.polygon_meters,
        doors: layout?.doors || [],
        windows: layout?.windows || [],
        wall_segments: layout?.walls || [],
        preset: preset,
        seed: seed
      };
      const response = await axios.post('/design', payload);
      const result = response.data;
      setFurnishingData((prev) => ({
        ...prev,
        [room.id]: result
      }));
    } catch (err) {
      console.error('Room design request failed:', err);
      setFurnishingData((prev) => ({
        ...prev,
        [room.id]: {
          is_valid: false,
          failure_reason: err.response?.data?.detail || 'INFERENCE_ERROR',
          items: []
        }
      }));
    } finally {
      setIsDesigning(false);
    }
  };

  const handleRegenerateDesign = (room, preset) => {
    const newSeed = Math.floor(Math.random() * 1000000) + 1;
    handleGenerateDesign(room, preset, newSeed);
  };

  const handleClearDesign = (roomId) => {
    setFurnishingData((prev) => {
      const next = { ...prev };
      delete next[roomId];
      return next;
    });
  };

  const handleRelabelRoom = (roomId, newType) => {
    if (!layout) return;
    const updatedRooms = layout.rooms.map((r) =>
      r.id === roomId ? { ...r, room_type: newType } : r
    );
    setLayout({ ...layout, rooms: updatedRooms });
    if (selectedRoom?.id === roomId) {
      setSelectedRoom((prev) => ({ ...prev, room_type: newType }));
    }
  };


  const renderTransitionOverlay = () => {
    if (!isTransitioning) return null;
    return (
      <div className="fixed inset-0 z-50 pointer-events-none overflow-hidden select-none architectural-unroll-overlay">
        {/* Full-Screen Architectural Parchment Sheet Unfurling */}
        <div className="architectural-unrolled-sheet">
          {/* Centered Blueprint Stencil Manifesto from User Design */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none px-6">
            <div className="border border-[#1d1d1f]/25 p-8 sm:p-12 md:p-16 bg-[#efebe2]/95 backdrop-blur-md max-w-3xl w-full shadow-2xl relative overflow-hidden">
              {/* Drafting Corner Registration Marks */}
              <div className="absolute top-3 left-4 font-mono text-[9px] text-[#1d1d1f]/40 uppercase tracking-widest">+ PT-00</div>
              <div className="absolute top-3 right-4 font-mono text-[9px] text-[#1d1d1f]/40 uppercase tracking-widest">+ SEC-01</div>
              <div className="absolute bottom-3 left-4 font-mono text-[9px] text-[#1d1d1f]/40 uppercase tracking-widest">+ ARCHIGEN</div>
              <div className="absolute bottom-3 right-4 font-mono text-[9px] text-[#1d1d1f]/40 uppercase tracking-widest">+ 2026</div>

              <p className="font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.24em] text-[#1d1d1f]/60 mb-4 text-center">
                ARCHIGEN 3D • DEEP LEARNING ARCHITECTURAL STUDIO
              </p>
              
              <h2 className="font-stencil text-2xl sm:text-3xl md:text-5xl text-[#1d1d1f] tracking-wide uppercase text-center leading-[1.08]">
                EVERY BUILDING STARTS FLAT.
              </h2>
              <h2 className="font-stencil text-2xl sm:text-3xl md:text-5xl text-[#1d1d1f] tracking-wide uppercase text-center leading-[1.08] mt-2">
                OURS DON'T <span className="text-[#f0500f]">STAY</span> THAT WAY.
              </h2>
              
              <div className="mt-8 flex items-center justify-center gap-4">
                <span className="w-16 h-[1px] bg-[#1d1d1f]/25"></span>
                <span className="font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.2em] text-[#1d1d1f]/80 font-semibold">
                  {transitionTarget === 'studio' ? 'UNROLLING AI GENERATOR STUDIO...' : 'UNROLLING SHOWCASE TOUR...'}
                </span>
                <span className="w-16 h-[1px] bg-[#1d1d1f]/25"></span>
              </div>
            </div>
          </div>
        </div>

        {/* Physical 3D Roll Cylinder Leading Edge */}
        <div className="architectural-roll-cylinder">
          <div className="roll-paper-layers" />
          <div className="absolute inset-0 opacity-40 mix-blend-multiply bg-[repeating-linear-gradient(45deg,#1d1d1f_0,#1d1d1f_1px,transparent_0,transparent_10px)]" />
        </div>
      </div>
    );
  };

  if (currentView === 'showcase') {
    return (
      <div className="relative w-screen h-screen overflow-hidden bg-[#efebe2]">
        {renderTransitionOverlay()}
        <iframe
          src="/showcase/index.html?type=stencil"
          title="ArchiGen 3D Showcase Tour"
          className="w-full h-full border-none"
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#efebe2] text-[#1d1d1f] font-sans selection:bg-[#b8452a]/20 selection:text-[#b8452a]">
      {renderTransitionOverlay()}

      {/* 1. Header */}
      <Navbar
        health={health}
        latency={latency}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onReturnToShowcase={() => triggerTransition('showcase')}
      />

      {/* 2. Main Work Area */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Sidebar Controls */}
        <ControlPanel
          onFileUpload={handleFileUpload}
          onRunAnalysis={() => executeAnalysis(file, scale, numFloors, frontWallId)}
          onGenerateFacade={handleGenerateFacade}
          isAnalyzing={isAnalyzing}
          isGeneratingFacade={isGeneratingFacade}
          scale={scale}
          setScale={setScale}
          numFloors={numFloors}
          setNumFloors={setNumFloors}
          selectedStyle={selectedStyle}
          setSelectedStyle={setSelectedStyle}
          frontWallId={frontWallId}
          hasLayout={!!layout}
          facadeTextureUrl={facadeTextureUrl}
          isCalibrating={isCalibrating}
          setIsCalibrating={setIsCalibrating}
        />

        {/* Center Viewport Area */}
        <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#efebe2] drafting-grid">
          {errorMsg && (
            <div className="bg-[#b8452a]/10 border-b border-[#b8452a]/30 px-6 py-2.5 font-mono text-xs text-[#b8452a] flex justify-between items-center z-20">
              <span className="font-medium tracking-wide">Notice: {errorMsg}</span>
              <button onClick={() => setErrorMsg(null)} className="underline uppercase tracking-wider text-[10px] ml-4 font-semibold hover:text-[#96351e]">
                Dismiss ×
              </button>
            </div>
          )}

          <div className="flex-1 flex overflow-hidden relative">
            {/* Split View: 2D Canvas + 3D Viewer */}
            {activeTab === 'split' && (
              <>
                <Floorplan2DCanvas
                  imageUrl={imageUrl}
                  layout={layout}
                  frontWallId={frontWallId}
                  onSelectFrontWall={handleSelectFrontWall}
                  onCalibrateScale={handleCalibrateScale}
                  isCalibrating={isCalibrating}
                  setIsCalibrating={setIsCalibrating}
                  onUpdateLayout={setLayout}
                  isSplit={true}
                  selectedRoom={selectedRoom}
                  onSelectRoom={handleSelectRoom}
                />
                <Viewer3D
                  layout={layout}
                  frontWallId={frontWallId}
                  facadeTextureUrl={facadeTextureUrl}
                  numFloors={numFloors}
                  isSplit={true}
                  selectedRoom={selectedRoom}
                  onSelectRoom={handleSelectRoom}
                  furnishingData={furnishingData}
                  onClearRoomFurnishings={handleClearDesign}
                />
              </>
            )}

            {/* 2D Only View */}
            {activeTab === '2d' && (
              <Floorplan2DCanvas
                imageUrl={imageUrl}
                layout={layout}
                frontWallId={frontWallId}
                onSelectFrontWall={handleSelectFrontWall}
                onCalibrateScale={handleCalibrateScale}
                isCalibrating={isCalibrating}
                setIsCalibrating={setIsCalibrating}
                onUpdateLayout={setLayout}
                isSplit={false}
                selectedRoom={selectedRoom}
                onSelectRoom={handleSelectRoom}
              />
            )}

            {/* 3D Only View */}
            {activeTab === '3d' && (
              <Viewer3D
                layout={layout}
                frontWallId={frontWallId}
                facadeTextureUrl={facadeTextureUrl}
                numFloors={numFloors}
                isSplit={false}
                selectedRoom={selectedRoom}
                onSelectRoom={handleSelectRoom}
                furnishingData={furnishingData}
                onClearRoomFurnishings={handleClearDesign}
              />
            )}

            {/* Room Design Floating Panel (Stage 3) */}
            {selectedRoom && (
              <RoomDesignPanel
                selectedRoom={selectedRoom}
                onClose={() => setSelectedRoom(null)}
                onGenerateDesign={handleGenerateDesign}
                onRegenerateDesign={handleRegenerateDesign}
                onClearDesign={handleClearDesign}
                furnishingData={furnishingData}
                isDesigning={isDesigning}
                onRelabelRoom={handleRelabelRoom}
              />
            )}
          </div>

          {/* Bottom Summary Bar */}
          <MetricsSummary layout={layout} frontWallId={frontWallId} />
        </main>
      </div>
    </div>
  );
}
