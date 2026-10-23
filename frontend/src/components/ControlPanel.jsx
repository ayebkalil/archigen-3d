import React, { useRef } from 'react';
import { Upload, Sparkles, Building2, Sliders, Palette, RefreshCw, Check, Ruler, FileText } from 'lucide-react';

const STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'White stucco, floor-to-ceiling glass' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Warm sandstone, rustic arches' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Exposed clay brick, framed lintels' },
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
  isCalibrating,
  setIsCalibrating,
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <aside className="w-80 h-full border-r border-[#1d1d1f]/15 bg-[#f7f4ed]/90 backdrop-blur-sm p-5 flex flex-col space-y-6 overflow-y-auto select-none font-sans text-[#1d1d1f]">
      {/* 1. Upload Blueprint */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="font-mono text-[10.5px] font-semibold text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
            <Upload className="w-3.5 h-3.5 text-[#b8452a]" />
            <span>1. Architectural Drawing</span>
          </label>
          <span className="font-mono text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-[#1d1d1f]/20 text-[#1d1d1f]/60">
            2D Raster
          </span>
        </div>

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/png,image/jpeg"
          className="hidden"
        />

        <div
          onClick={() => fileInputRef.current?.click()}
          className="mt-1 border-2 border-dashed border-[#1d1d1f]/30 hover:border-[#b8452a] bg-[#efebe2]/60 hover:bg-[#efebe2] rounded-xl p-4 text-center cursor-pointer transition-all group shadow-sm"
        >
          <div className="w-10 h-10 mx-auto rounded-full bg-white/80 border border-[#1d1d1f]/20 group-hover:border-[#b8452a] group-hover:bg-[#b8452a]/10 flex items-center justify-center transition-colors">
            <Upload className="w-4 h-4 text-[#1d1d1f]/70 group-hover:text-[#b8452a] transition-colors" />
          </div>
          <p className="mt-2 font-mono text-[11px] font-medium text-[#1d1d1f] uppercase tracking-wide">
            Upload Plan (PNG or JPG)
          </p>
          <p className="font-mono text-[9.5px] text-[#1d1d1f]/60 mt-0.5">
            Architectural sketch, scan, or CAD blueprint
          </p>
        </div>
      </div>

      {/* 2. Scale & Geometry Parameters */}
      <div className="space-y-4">
        <label className="font-mono text-[10.5px] font-semibold text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
          <Sliders className="w-3.5 h-3.5 text-[#b8452a]" />
          <span>2. Dimensions & Calibration</span>
        </label>

        {/* Scale Slider & 2-Point Calibration */}
        <div className="bg-[#efebe2]/80 rounded-xl p-3.5 border border-[#1d1d1f]/15 shadow-sm">
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-[#1d1d1f]/70">Scale Ratio:</span>
            <span className="font-mono text-[11px] text-[#b8452a] font-semibold">{scale} m/px</span>
          </div>
          <input
            type="range"
            min="0.015"
            max="0.060"
            step="0.001"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-[#1d1d1f]/15 rounded-lg appearance-none cursor-pointer accent-[#b8452a]"
          />
          <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-[#1d1d1f]/10">
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
              Standard door ~0.90m
            </span>
            <button
              onClick={() => setIsCalibrating && setIsCalibrating(!isCalibrating)}
              className={`font-mono text-[9.5px] uppercase tracking-wider px-2.5 py-1 rounded-full border transition-all flex items-center space-x-1 ${
                isCalibrating
                  ? 'bg-[#b8452a] border-[#b8452a] text-white font-semibold'
                  : 'bg-white/80 border-[#1d1d1f]/30 text-[#1d1d1f] hover:bg-[#1d1d1f] hover:text-[#efebe2]'
              }`}
            >
              <Ruler className="w-3 h-3" />
              <span>{isCalibrating ? 'Measuring...' : '2-Pt Calibrate'}</span>
            </button>
          </div>
        </div>

        {/* Floor Count */}
        <div className="bg-[#efebe2]/80 rounded-xl p-3.5 border border-[#1d1d1f]/15 shadow-sm">
          <div className="flex justify-between items-center text-xs mb-2">
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-[#1d1d1f]/70">Building Storeys:</span>
            <span className="font-mono text-[11px] font-semibold text-[#1d1d1f]">{numFloors} Floor{numFloors > 1 ? 's' : ''} ({(numFloors * 2.8).toFixed(1)}m)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[1, 2, 3].map((f) => (
              <button
                key={f}
                onClick={() => setNumFloors(f)}
                className={`py-1.5 rounded-lg font-mono text-[10px] uppercase tracking-wider border transition-all ${
                  numFloors === f
                    ? 'bg-[#1d1d1f] border-[#1d1d1f] text-[#efebe2] font-semibold shadow-sm'
                    : 'bg-white/60 border-[#1d1d1f]/20 text-[#1d1d1f]/70 hover:bg-white hover:text-[#1d1d1f]'
                }`}
              >
                {f} Storey{f > 1 ? 's' : ''}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Facade Style Selector */}
      <div className="space-y-2.5">
        <label className="font-mono text-[10.5px] font-semibold text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
          <Palette className="w-3.5 h-3.5 text-[#b8452a]" />
          <span>3. Facade Material & Style</span>
        </label>

        <div className="space-y-2">
          {STYLES.map((style) => (
            <div
              key={style.id}
              onClick={() => setSelectedStyle(style.id)}
              className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                selectedStyle === style.id
                  ? 'bg-white border-[#b8452a] shadow-sm'
                  : 'bg-[#efebe2]/60 border-[#1d1d1f]/15 hover:border-[#1d1d1f]/40 hover:bg-[#efebe2]'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-serif text-sm text-[#1d1d1f] font-normal">{style.label}</span>
                {selectedStyle === style.id && <Check className="w-3.5 h-3.5 text-[#b8452a]" />}
              </div>
              <p className="font-mono text-[9px] text-[#1d1d1f]/60 mt-0.5">{style.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 4. Action Buttons (Pill CTA) */}
      <div className="pt-2 space-y-2.5 mt-auto">
        <button
          onClick={onRunAnalysis}
          disabled={isAnalyzing}
          className="w-full h-10 px-5 rounded-full bg-[#1d1d1f] hover:bg-black text-[#efebe2] font-mono text-[10.5px] uppercase tracking-widest font-semibold flex items-center justify-center space-x-2 transition-all shadow-md disabled:opacity-50"
        >
          {isAnalyzing ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#b8452a]" />
              <span>Analyzing Blueprint...</span>
            </>
          ) : (
            <>
              <Building2 className="w-3.5 h-3.5 text-[#b8452a]" />
              <span>Extract 3D Floorplan →</span>
            </>
          )}
        </button>

        <button
          onClick={onGenerateFacade}
          disabled={!hasLayout || isGeneratingFacade}
          className="w-full h-10 px-5 rounded-full bg-[#b8452a] hover:bg-[#96351e] text-white font-mono text-[10.5px] uppercase tracking-widest font-semibold flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#b8452a]/20 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {isGeneratingFacade ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Synthesizing cGAN Texture...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>Generate Facade Texture →</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
