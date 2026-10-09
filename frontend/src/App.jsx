import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ControlPanel from './components/ControlPanel';
import Floorplan2DCanvas from './components/Floorplan2DCanvas';
import Viewer3D from './components/Viewer3D';
import MetricsSummary from './components/MetricsSummary';

export default function App() {
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
  const [health, setHealth] = useState(null);
  const [latency, setLatency] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // 1. Health check & Initial Demo load
  useEffect(() => {
    axios
      .get('/health')
      .then((res) => setHealth(res.data))
      .catch((err) => console.log('Backend not reachable yet:', err));

    // Automatically analyze default sample floorplan on initial startup
    loadDefaultSample();
  }, []);

  const loadDefaultSample = async () => {
    try {
      setIsAnalyzing(true);
      const res = await fetch('/sample_floorplan.png');
      const blob = await res.blob();
      const sampleFile = new File([blob], 'sample_floorplan.png', { type: 'image/png' });
      setFile(sampleFile);
      setImageUrl('/sample_floorplan.png');
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

  const handleGenerateFacade = () => {
    if (file) {
      setIsGeneratingFacade(true);
      executeAnalysis(file, scale, numFloors, frontWallId).finally(() => {
        setIsGeneratingFacade(false);
      });
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* 1. Header */}
      <Navbar
        health={health}
        latency={latency}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
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
        />

        {/* Center Viewport Area */}
        <main className="flex-1 flex flex-col h-full overflow-hidden bg-slate-950">
          {errorMsg && (
            <div className="bg-rose-500/10 border-b border-rose-500/20 px-6 py-2 text-xs text-rose-300 flex justify-between items-center">
              <span>{errorMsg}</span>
              <button onClick={() => setErrorMsg(null)} className="underline">
                Dismiss
              </button>
            </div>
          )}

          <div className="flex-1 flex overflow-hidden">
            {/* Split View: 2D Canvas + 3D Viewer */}
            {activeTab === 'split' && (
              <>
                <Floorplan2DCanvas
                  imageUrl={imageUrl}
                  layout={layout}
                  frontWallId={frontWallId}
                  onSelectFrontWall={handleSelectFrontWall}
                />
                <Viewer3D
                  layout={layout}
                  frontWallId={frontWallId}
                  facadeTextureUrl={facadeTextureUrl}
                  numFloors={numFloors}
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
              />
            )}

            {/* 3D Only View */}
            {activeTab === '3d' && (
              <Viewer3D
                layout={layout}
                frontWallId={frontWallId}
                facadeTextureUrl={facadeTextureUrl}
                numFloors={numFloors}
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
