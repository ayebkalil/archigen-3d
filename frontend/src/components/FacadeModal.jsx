import React, { useState } from 'react';
import { 
  X, 
  Sparkles, 
  Download, 
  Eye, 
  Layers, 
  RefreshCw, 
  Check, 
  Maximize2, 
  Info,
  Palette,
  ExternalLink
} from 'lucide-react';

const STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'White stucco, floor-to-ceiling glass' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Warm sandstone, rustic masonry' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Exposed clay brick, framed lintels' },
  { id: 'Lush Vegetation', label: 'Lush Vegetation', desc: 'Biophilic facade, vertical greening' },
];

const CMP_LEGEND = [
  { label: 'Wall', color: '#0030ff', desc: 'Primary facade surface' },
  { label: 'Window', color: '#0080ff', desc: 'Glazed openings' },
  { label: 'Door', color: '#00cfff', desc: 'Entrance / egress' },
  { label: 'Window Sill', color: '#ff5000', desc: 'Bottom stone sill' },
  { label: 'Window Lintel', color: '#ffff00', desc: 'Top structural lintel' },
  { label: 'Cornice / Trim', color: '#0000de', desc: 'Roofline & base molding' },
];

export default function FacadeModal({
  isOpen,
  onClose,
  facadeData,
  onSwitchStyle,
  isSwitchingStyle = false,
  onFocusIn3D
}) {
  const [activeTab, setActiveTab] = useState('side_by_side'); // 'side_by_side' | 'realistic' | 'sketch'

  if (!isOpen || !facadeData) return null;

  const {
    wall_id = 'Front Wall',
    style = 'Modern Minimalist',
    wall_length_m = 0,
    wall_height_m = 0,
    num_floors = 1,
    texture_base64,
    sketch_base64,
    device = 'ONNX Runtime',
    model_info
  } = facadeData;

  const handleDownload = () => {
    if (!texture_base64) return;
    const link = document.createElement('a');
    link.href = texture_base64;
    link.download = `facade_${style.toLowerCase().replace(/\s+/g, '_')}_${wall_id}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#efebe2] border-2 border-[#1d1d1f] rounded-2xl shadow-2xl w-full max-w-5xl max-h-[92vh] flex flex-col overflow-hidden text-[#1d1d1f]">
        
        {/* Modal Header */}
        <div className="px-6 py-4 bg-[#e5e0d4] border-b border-[#1d1d1f]/20 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-[#f0500f] text-white flex items-center justify-center shadow-sm">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="font-stencil text-base uppercase tracking-wider text-[#1d1d1f]">
                  FACADE SYNTHESIS STUDIO
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#f0500f]/15 text-[#f0500f] border border-[#f0500f]/30">
                  PIX2PIX cGAN
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#1d1d1f]/10 text-[#1d1d1f]/70">
                  {device}
                </span>
              </div>
              <p className="font-mono text-xs text-[#1d1d1f]/65">
                {model_info?.model === 'facade_residual_film_512_v1'
                  ? `Experimental 512 facade · ${model_info.effective_style}`
                  : 'Semantic layout sketch translated into an architectural facade'}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-black/10 text-[#1d1d1f]/70 hover:text-[#1d1d1f] transition-colors"
              title="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Dimension & Wall Context Bar */}
        <div className="px-6 py-2.5 bg-[#f6f3eb] border-b border-[#1d1d1f]/10 flex flex-wrap items-center justify-between text-xs font-mono">
          <div className="flex items-center space-x-4">
            <span className="text-[#1d1d1f]/60">Wall Target:</span>
            <span className="font-bold text-[#1d1d1f] bg-white px-2 py-0.5 rounded border border-[#1d1d1f]/20">
              {wall_id}
            </span>
            <span className="text-[#1d1d1f]/60">Dimensions:</span>
            <span className="font-bold text-[#1d1d1f]">
              {wall_length_m ? `${wall_length_m.toFixed(2)}m (W) × ${(wall_height_m || 2.8).toFixed(2)}m (H)` : 'Aspect preserved'}
            </span>
            <span className="text-[#1d1d1f]/60">Stories:</span>
            <span className="font-bold text-[#1d1d1f]">
              {num_floors} Floor{num_floors > 1 ? 's' : ''}
            </span>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-[#e5e0d4] p-0.5 rounded-lg border border-[#1d1d1f]/15">
            <button
              onClick={() => setActiveTab('side_by_side')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'side_by_side'
                  ? 'bg-white shadow text-[#1d1d1f] font-semibold'
                  : 'text-[#1d1d1f]/60 hover:text-[#1d1d1f]'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveTab('realistic')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'realistic'
                  ? 'bg-white shadow text-[#1d1d1f] font-semibold'
                  : 'text-[#1d1d1f]/60 hover:text-[#1d1d1f]'
              }`}
            >
              Realistic Only
            </button>
            <button
              onClick={() => setActiveTab('sketch')}
              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                activeTab === 'sketch'
                  ? 'bg-white shadow text-[#1d1d1f] font-semibold'
                  : 'text-[#1d1d1f]/60 hover:text-[#1d1d1f]'
              }`}
            >
              Semantic Scrap
            </button>
          </div>
        </div>

        {/* Modal Body: Side-by-Side Visualizer */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div className={`grid gap-6 ${activeTab === 'side_by_side' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
            
            {/* LEFT: Input Semantic Layout Scrap */}
            {(activeTab === 'side_by_side' || activeTab === 'sketch') && (
              <div className="bg-white rounded-xl border border-[#1d1d1f]/20 p-4 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-[#0030ff]" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f]">
                      01 // INPUT SEMANTIC SKETCH ("THE SCRAP")
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#1d1d1f]/50">CMP 12-Class Colormap</span>
                </div>

                <p className="text-[11px] font-mono text-[#1d1d1f]/70 mb-3">
                  This is the semantic layout "scrap" projected automatically from your floorplan's exterior wall, doors, and windows geometry.
                </p>

                <div className="relative aspect-square w-full bg-[#0030ff]/10 rounded-lg overflow-hidden border border-[#1d1d1f]/10 flex items-center justify-center p-2">
                  {sketch_base64 ? (
                    <img
                      src={sketch_base64}
                      alt="Input Semantic Facade Sketch"
                      className="max-h-full max-w-full object-contain rounded shadow-inner"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  ) : (
                    <div className="text-center p-4 text-[#1d1d1f]/50 font-mono text-xs">
                      No semantic sketch available
                    </div>
                  )}
                </div>

                {/* CMP Color Palette Legend */}
                <div className="mt-3 pt-3 border-t border-[#1d1d1f]/10">
                  <span className="font-mono text-[10px] font-bold text-[#1d1d1f]/60 uppercase tracking-wider block mb-1.5">
                    CMP Semantic Legend:
                  </span>
                  <div className="grid grid-cols-3 gap-1.5 text-[10px] font-mono">
                    {CMP_LEGEND.map((item) => (
                      <div key={item.label} className="flex items-center space-x-1.5 bg-[#efebe2]/60 px-1.5 py-1 rounded">
                        <span
                          className="w-2.5 h-2.5 rounded-sm border border-black/20 flex-shrink-0"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="truncate text-[#1d1d1f]/80" title={item.desc}>
                          {item.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* RIGHT: Synthesized Realistic Facade */}
            {(activeTab === 'side_by_side' || activeTab === 'realistic') && (
              <div className="bg-white rounded-xl border border-[#1d1d1f]/20 p-4 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-[#f0500f] animate-pulse" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f]">
                      02 // SYNTHESIZED REALISTIC FACADE
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#f0500f]/10 text-[#f0500f] font-bold">
                    {model_info?.effective_style || style}
                  </span>
                </div>

                <p className="text-[11px] font-mono text-[#1d1d1f]/70 mb-3">
                  Conditional GAN inference output translated into a realistic architectural facade with material rendering.
                </p>

                <div className="relative aspect-square w-full bg-[#1d1d1f]/5 rounded-lg overflow-hidden border border-[#1d1d1f]/10 flex items-center justify-center p-2">
                  {isSwitchingStyle ? (
                    <div className="flex flex-col items-center justify-center space-y-2 text-[#f0500f]">
                      <RefreshCw className="w-6 h-6 animate-spin" />
                      <span className="font-mono text-xs">Synthesizing {style}...</span>
                    </div>
                  ) : texture_base64 ? (
                    <img
                      src={texture_base64}
                      alt="Synthesized Realistic Facade"
                      className="max-h-full max-w-full object-contain rounded shadow-md"
                    />
                  ) : (
                    <div className="text-center p-4 text-[#1d1d1f]/50 font-mono text-xs">
                      No realistic facade texture available
                    </div>
                  )}
                </div>

                {/* Synthesis Quality Badges */}
                <div className="mt-3 pt-3 border-t border-[#1d1d1f]/10 flex items-center justify-between text-[10px] font-mono text-[#1d1d1f]/60">
                  <span>Aspect-Ratio Preserved (P95 &lt; 35ms)</span>
                  <span className="text-[#f0500f] font-semibold">Active Texture in 3D</span>
                </div>
              </div>
            )}

          </div>

          {/* Style Selector Section */}
          <div className="bg-[#e5e0d4]/60 border border-[#1d1d1f]/15 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <label className="font-stencil text-xs text-[#1d1d1f] uppercase tracking-wider flex items-center space-x-1.5">
                <Palette className="w-3.5 h-3.5 text-[#f0500f]" />
                <span>SWITCH ARCHITECTURAL STYLE & RE-SYNTHESIZE</span>
              </label>
              <span className="font-mono text-[10px] text-[#1d1d1f]/60">
                Click any style to generate instantly
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
              {STYLES.map((st) => (
                <button
                  key={st.id}
                  onClick={() => onSwitchStyle && onSwitchStyle(st.id)}
                  disabled={isSwitchingStyle}
                  className={`p-2.5 rounded-lg border text-left transition-all ${
                    style === st.id
                      ? 'bg-white border-[#f0500f] shadow-sm ring-1 ring-[#f0500f]'
                      : 'bg-white/60 border-[#1d1d1f]/15 hover:border-[#1d1d1f]/40 hover:bg-white'
                  } disabled:opacity-50`}
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="font-stencil text-xs tracking-wider uppercase text-[#1d1d1f]">
                      {st.label}
                    </span>
                    {style === st.id && <Check className="w-3.5 h-3.5 text-[#f0500f]" />}
                  </div>
                  <p className="font-mono text-[9px] text-[#1d1d1f]/60 leading-tight">
                    {st.desc}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer Controls */}
        <div className="px-6 py-4 bg-[#e5e0d4] border-t border-[#1d1d1f]/20 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-mono text-[#1d1d1f]/65">
            <Info className="w-4 h-4 text-[#f0500f]" />
            <span>The generated texture is already applied to the front wall in your 3D view.</span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleDownload}
              className="h-9 px-4 rounded-full bg-white hover:bg-[#efebe2] border border-[#1d1d1f]/30 font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] flex items-center space-x-2 transition-all shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              <span>DOWNLOAD PNG</span>
            </button>

            {onFocusIn3D && (
              <button
                onClick={() => {
                  onFocusIn3D();
                  onClose();
                }}
                className="h-9 px-4 rounded-full bg-[#1d1d1f] hover:bg-black text-[#efebe2] font-stencil text-xs uppercase tracking-wider flex items-center space-x-2 transition-all shadow-md"
              >
                <Eye className="w-3.5 h-3.5 text-[#f0500f]" />
                <span>FOCUS IN 3D SCENE</span>
              </button>
            )}

            <button
              onClick={onClose}
              className="h-9 px-5 rounded-full bg-[#f0500f] hover:bg-[#d8450a] text-white font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-md"
            >
              <span>DONE</span>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
