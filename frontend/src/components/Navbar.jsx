import React from 'react';
import { ArrowLeft, Box, Compass, Sparkles } from 'lucide-react';

export default function Navbar({ health, latency, activeTab, setActiveTab, onReturnToShowcase, pageView = 'floorplan', setPageView }) {
  return (
    <header className="h-16 border-b border-[#1d1d1f]/15 bg-[#efebe2]/95 backdrop-blur-md px-6 flex items-center justify-between z-30 select-none">
      {/* Left: Brand & Page Switcher */}
      <div className="flex items-center space-x-5">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-full border border-[#1d1d1f] bg-[#efebe2] flex items-center justify-center shadow-sm">
            <Compass className="w-5 h-5 text-[#b8452a]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-stencil text-xl tracking-wider uppercase text-[#1d1d1f] font-normal leading-none">
                ARCHIGEN <span className="text-[#f0500f]">3D</span>
              </h1>
              <span className="font-mono text-[9px] uppercase tracking-wider px-2 py-0.5 rounded-full border border-[#1d1d1f]/30 bg-[#efebe2] text-[#1d1d1f]/80">
                STUDIO
              </span>
            </div>
            <p className="font-mono text-[9px] text-[#1d1d1f]/60 tracking-wider uppercase mt-0.5">
              NEURAL BIM & 3D RECONSTRUCTION
            </p>
          </div>
        </div>

        {/* Primary Page Switcher (Floorplan BIM vs AI Generative Studio) */}
        {setPageView && (
          <div className="flex items-center p-1 rounded-full border border-[#1d1d1f]/20 bg-white/60 ml-2">
            <button
              onClick={() => setPageView('floorplan')}
              className={`h-8 px-3.5 rounded-full uppercase tracking-wider font-stencil text-[11px] flex items-center space-x-1.5 transition-all ${
                pageView === 'floorplan'
                  ? 'bg-[#1d1d1f] text-[#efebe2] shadow-sm'
                  : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f]'
              }`}
            >
              <Box className="w-3.5 h-3.5 text-[#f0500f]" />
              <span>3D Floorplan & BIM</span>
            </button>

            <button
              onClick={() => setPageView('generative_studio')}
              className={`h-8 px-3.5 rounded-full uppercase tracking-wider font-stencil text-[11px] flex items-center space-x-1.5 transition-all ${
                pageView === 'generative_studio'
                  ? 'bg-[#1d1d1f] text-[#efebe2] shadow-sm'
                  : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f]'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-[#f0500f] animate-pulse" />
              <span>AI Generative Studio</span>
            </button>
          </div>
        )}

        {/* Return to Showcase Pill Button */}
        {onReturnToShowcase && (
          <button
            onClick={onReturnToShowcase}
            className="hidden lg:inline-flex items-center space-x-1.5 h-8 px-3.5 rounded-full border border-[#1d1d1f]/40 bg-white/40 hover:bg-[#1d1d1f] hover:text-[#efebe2] text-[#1d1d1f] transition-all font-stencil text-[11px] uppercase tracking-wider ml-1"
            title="Return to the Casa Serra 3D Scroll Walkthrough"
          >
            <ArrowLeft className="w-3 h-3" />
            <span>SHOWCASE</span>
          </button>
        )}
      </div>

      {/* Right / Center: View Switcher (Architectural Pills for Floorplan) or Studio Tag */}
      {pageView === 'floorplan' ? (
        <div className="flex items-center p-1 rounded-full border border-[#1d1d1f]/20 bg-white/50 backdrop-blur-sm text-xs">
          <button
            onClick={() => setActiveTab('split')}
            className={`h-7 px-3.5 rounded-full uppercase tracking-wider font-stencil text-[11px] transition-all ${
              activeTab === 'split'
                ? 'bg-[#1d1d1f] text-[#efebe2] font-normal shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            SPLIT (2D + 3D)
          </button>
          <button
            onClick={() => setActiveTab('2d')}
            className={`h-7 px-3.5 rounded-full uppercase tracking-wider font-stencil text-[11px] transition-all ${
              activeTab === '2d'
                ? 'bg-[#1d1d1f] text-[#efebe2] font-normal shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            2D BLUEPRINT
          </button>
          <button
            onClick={() => setActiveTab('3d')}
            className={`h-7 px-3.5 rounded-full uppercase tracking-wider font-stencil text-[11px] transition-all ${
              activeTab === '3d'
                ? 'bg-[#1d1d1f] text-[#efebe2] font-normal shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            3D PERSPECTIVE
          </button>
        </div>
      ) : (
        <div className="flex items-center space-x-2 text-xs font-mono text-[#1d1d1f]/70 bg-white/40 px-3.5 py-1.5 rounded-full border border-[#1d1d1f]/15">
          <Sparkles className="w-3.5 h-3.5 text-[#f0500f]" />
          <span>Generative AI Studio: 2D-to-3D Furniture + Neural Facades</span>
        </div>
      )}
    </header>
  );
}
