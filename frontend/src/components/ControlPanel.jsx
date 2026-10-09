import React, { useRef } from 'react';
import { Upload, Sparkles, Building, Sliders, Palette, Download, RefreshCw, Check } from 'lucide-react';

const SAMPLE_PLANS = [
  { name: 'Standard Apartment', path: '/samples/plan_sample1.png', scale: 0.03 },
  { name: 'Villa 3-Bed', path: '/samples/plan_sample2.png', scale: 0.028 },
];

const STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'White stucco, floor-to-ceiling glass' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Warm sandstone, rustic arches' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Exposed clay brick, framed windows' },
  { id: 'Lush Vegetation', label: 'Lush Vegetation', desc: 'Biophilic facade, vertical greening' },
];

export default function ControlPanel({
  onFileUpload,
  onRunAnalysis,
  onGenerateFacade,
  isAnalyzing,
  isGeneratingFacade,
  scale,
  setScale,
  numFloors,
  setNumFloors,
  selectedStyle,
  setSelectedStyle,
  frontWallId,
  hasLayout,
  facadeTextureUrl,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <aside className="w-80 h-full border-r border-slate-800 bg-slate-900/40 p-5 flex flex-col space-y-6 overflow-y-auto select-none">
      {/* 1. Upload Floorplan */}
      <div>
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
          <Upload className="w-3.5 h-3.5 text-sky-400" />
          <span>1. Floorplan Blueprint</span>
        </label>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/png,image/jpeg"
          className="hidden"
        />

        <div
          onClick={() => fileInputRef.current?.click()}
          className="mt-1 border-2 border-dashed border-slate-700 hover:border-sky-500/60 bg-slate-950/60 hover:bg-slate-950 rounded-xl p-4 text-center cursor-pointer transition-all group"
        >
          <div className="w-10 h-10 mx-auto rounded-full bg-slate-900 group-hover:bg-sky-500/10 flex items-center justify-center transition-colors">
            <Upload className="w-5 h-5 text-slate-400 group-hover:text-sky-400 transition-colors" />
          </div>
          <p className="mt-2 text-xs font-medium text-slate-200">
            Click to upload PNG or JPG
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">
            Architectural blueprint or sketch
          </p>
        </div>
      </div>

      {/* 2. Scale & Geometry Parameters */}
      <div className="space-y-4">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
          <Sliders className="w-3.5 h-3.5 text-indigo-400" />
          <span>2. Dimensions & Floors</span>
        </label>

        {/* Scale Slider */}
        <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800/80">
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="text-slate-400">Scale Ratio:</span>
            <span className="font-mono text-sky-400 font-semibold">{scale} m/px</span>
          </div>
          <input
            type="range"
            min="0.015"
            max="0.060"
            step="0.001"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
          />
          <span className="text-[10px] text-slate-500 block mt-1">
            Standard residential door: ~0.90 m
          </span>
        </div>

        {/* Floor Count */}
        <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800/80">
          <div className="flex justify-between items-center text-xs mb-2">
            <span className="text-slate-400">Building Floors:</span>
            <span className="font-semibold text-indigo-400">{numFloors} Storey ({numFloors * 2.8}m)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[1, 2, 3].map((f) => (
              <button
                key={f}
                onClick={() => setNumFloors(f)}
                className={`py-1.5 rounded-lg text-xs font-medium border transition-all ${
                  numFloors === f
                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {f} Floor{f > 1 ? 's' : ''}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Facade Style Selector */}
      <div className="space-y-2.5">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
          <Palette className="w-3.5 h-3.5 text-amber-400" />
          <span>3. Exterior Architectural Style</span>
        </label>

        <div className="space-y-2">
          {STYLES.map((style) => (
            <div
              key={style.id}
              onClick={() => setSelectedStyle(style.id)}
              className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                selectedStyle === style.id
                  ? 'bg-amber-500/10 border-amber-500/40 text-amber-200'
                  : 'bg-slate-950/60 border-slate-800/80 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">{style.label}</span>
                {selectedStyle === style.id && <Check className="w-3.5 h-3.5 text-amber-400" />}
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">{style.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. Action Buttons */}
      <div className="pt-2 space-y-2.5 mt-auto">
        <button
          onClick={onRunAnalysis}
          disabled={isAnalyzing}
          className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white text-xs font-semibold shadow-lg shadow-sky-600/20 flex items-center justify-center space-x-2 transition-all disabled:opacity-50"
        >
          {isAnalyzing ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Analyzing Architecture...</span>
            </>
          ) : (
            <>
              <Building className="w-3.5 h-3.5" />
              <span>Extract 3D Floorplan</span>
            </>
          )}
        </button>

        <button
          onClick={onGenerateFacade}
          disabled={!hasLayout || isGeneratingFacade}
          className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white text-xs font-semibold shadow-lg shadow-amber-600/20 flex items-center justify-center space-x-2 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {isGeneratingFacade ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Synthesizing Facade...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>Generate Facade Texture</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
