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
  facadeModelInfo,
  isCalibrating,
  setIsCalibrating,
  onOpenFacadeModal,
  onOpenGenerativeStudio,
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
          <label className="font-stencil text-xs text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
            <Upload className="w-3.5 h-3.5 text-[#f0500f]" />
            <span>01 // ARCHITECTURAL DRAWING</span>
          </label>
          <span className="font-mono text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-[#1d1d1f]/20 text-[#1d1d1f]/60">
            2D RASTER
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
          className="mt-1 border-2 border-dashed border-[#1d1d1f]/30 hover:border-[#f0500f] bg-[#efebe2]/60 hover:bg-[#efebe2] rounded-xl p-4 text-center cursor-pointer transition-all group shadow-sm"
        >
          <div className="w-10 h-10 mx-auto rounded-full bg-white/80 border border-[#1d1d1f]/20 group-hover:border-[#f0500f] group-hover:bg-[#f0500f]/10 flex items-center justify-center transition-colors">
            <Upload className="w-4 h-4 text-[#1d1d1f]/70 group-hover:text-[#f0500f] transition-colors" />
          </div>
          <p className="mt-2 font-stencil text-xs text-[#1d1d1f] uppercase tracking-wider">
            UPLOAD PLAN (PNG OR JPG)
          </p>
          <p className="font-mono text-[9px] text-[#1d1d1f]/60 mt-0.5">
            Architectural sketch, scan, or CAD blueprint
          </p>
        </div>
      </div>

      {/* 2. Scale & Geometry Parameters */}
      <div className="space-y-4">
        <label className="font-stencil text-xs text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
          <Sliders className="w-3.5 h-3.5 text-[#f0500f]" />
          <span>02 // SPATIAL CALIBRATION</span>
        </label>

        {/* Scale Slider & 2-Point Calibration */}
        <div className="bg-[#efebe2]/80 rounded-xl p-3.5 border border-[#1d1d1f]/15 shadow-sm">
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="font-mono text-[10px] uppercase tracking-wider text-[#1d1d1f]/70">Scale Ratio:</span>
            <span className="font-stencil text-xs text-[#f0500f] tracking-wide">{scale} M/PX</span>
          </div>
          <input
            type="range"
            min="0.015"
            max="0.060"
            step="0.001"
            value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-[#1d1d1f]/15 rounded-lg appearance-none cursor-pointer accent-[#f0500f]"
          />
          <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-[#1d1d1f]/10">
            <span className="font-mono text-[9px] uppercase tracking-wider text-[#1d1d1f]/60">
              Standard door ~0.90m
            </span>
            <button
              onClick={() => setIsCalibrating && setIsCalibrating(!isCalibrating)}
              className={`font-stencil text-[10px] uppercase tracking-wider px-2.5 py-1 rounded-full border transition-all flex items-center space-x-1 ${
                isCalibrating
                  ? 'bg-[#f0500f] border-[#f0500f] text-white'
                  : 'bg-white/80 border-[#1d1d1f]/30 text-[#1d1d1f] hover:bg-[#1d1d1f] hover:text-[#efebe2]'
              }`}
            >
              <Ruler className="w-3 h-3" />
              <span>{isCalibrating ? 'MEASURING...' : '2-PT CALIBRATE'}</span>
            </button>
          </div>
        </div>

        {/* Floor Count */}
        <div className="bg-[#efebe2]/80 rounded-xl p-3.5 border border-[#1d1d1f]/15 shadow-sm">
          <div className="flex justify-between items-center text-xs mb-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-[#1d1d1f]/70">Building Storeys:</span>
            <span className="font-mono text-[11px] font-semibold text-[#1d1d1f]">{numFloors} Floor{numFloors > 1 ? 's' : ''} ({(numFloors * 2.8).toFixed(1)}m)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[1, 2, 3].map((f) => (
              <button
                key={f}
                onClick={() => setNumFloors(f)}
                className={`py-1.5 rounded-lg font-stencil text-[11px] uppercase tracking-wider border transition-all ${
                  numFloors === f
                    ? 'bg-[#1d1d1f] border-[#1d1d1f] text-[#efebe2] shadow-sm'
                    : 'bg-white/60 border-[#1d1d1f]/20 text-[#1d1d1f]/70 hover:bg-white hover:text-[#1d1d1f]'
                }`}
              >
                {f} STOREY{f > 1 ? 'S' : ''}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Facade Style Selector */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <label className="font-stencil text-xs text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
            <Palette className="w-3.5 h-3.5 text-[#f0500f]" />
            <span>03 // FACADE SYNTHESIS</span>
          </label>
          {onOpenGenerativeStudio && (
            <button
              type="button"
              onClick={onOpenGenerativeStudio}
              className="px-2 py-0.5 rounded text-[9.5px] font-mono text-[#f0500f] hover:bg-[#f0500f]/10 flex items-center space-x-1"
              title="Open full Generative AI Studio page with 2D-to-3D Furniture and Facades"
            >
              <Sparkles className="w-2.5 h-2.5" />
              <span>AI STUDIO ↗</span>
            </button>
          )}
        </div>

        <div className="space-y-2">
          {STYLES.map((style) => (
            <div
              key={style.id}
              onClick={() => setSelectedStyle(style.id)}
              className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                selectedStyle === style.id
                  ? 'bg-white border-[#f0500f] shadow-sm'
                  : 'bg-[#efebe2]/60 border-[#1d1d1f]/15 hover:border-[#1d1d1f]/40 hover:bg-[#efebe2]'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-stencil text-xs tracking-wider uppercase text-[#1d1d1f]">{style.label}</span>
                {selectedStyle === style.id && <Check className="w-3.5 h-3.5 text-[#f0500f]" />}
              </div>
              <p className="font-mono text-[9px] text-[#1d1d1f]/60 mt-0.5">{style.desc}</p>
            </div>
          ))}
        </div>

        {facadeTextureUrl && (
          <div className="mt-2.5 p-2 bg-[#efebe2] border border-[#f0500f]/30 rounded-xl flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <img
                src={facadeTextureUrl}
                alt="Active Facade"
                className="w-10 h-10 object-cover rounded border border-[#1d1d1f]/20 shadow-xs"
              />
              <div className="text-[10px] font-mono leading-tight">
                <span className="font-bold text-[#1d1d1f] block">Facade Ready</span>
                <span className="text-[#f0500f]">{facadeModelInfo?.effective_style || selectedStyle}</span>
                {facadeModelInfo?.resolution === 512 && <span className="block">512px experiment</span>}
              </div>
            </div>
            {onOpenFacadeModal && (
              <button
                type="button"
                onClick={onOpenFacadeModal}
                className="px-2 py-1 bg-white hover:bg-[#f6f3eb] border border-[#1d1d1f]/20 rounded-md font-mono text-[10px] text-[#1d1d1f] flex items-center space-x-1 shadow-xs cursor-pointer"
              >
                <span>STUDIO</span>
                <span>↗</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* 4. Action Buttons (Pill CTA) */}
      <div className="pt-2 space-y-2.5 mt-auto">
        <button
          onClick={onRunAnalysis}
          disabled={isAnalyzing}
          className="w-full h-10 px-5 rounded-full bg-[#1d1d1f] hover:bg-black text-[#efebe2] font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md disabled:opacity-50"
        >
          {isAnalyzing ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#f0500f]" />
              <span>ANALYZING BLUEPRINT...</span>
            </>
          ) : (
            <>
              <Building2 className="w-3.5 h-3.5 text-[#f0500f]" />
              <span>EXTRACT 3D FLOORPLAN →</span>
            </>
          )}
        </button>

        <button
          onClick={onGenerateFacade}
          disabled={!hasLayout || isGeneratingFacade}
          className="w-full h-10 px-5 rounded-full bg-[#f0500f] hover:bg-[#d8450a] text-white font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#f0500f]/20 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {isGeneratingFacade ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>SYNTHESIZING cGAN TEXTURE...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>GENERATE FACADE TEXTURE →</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
