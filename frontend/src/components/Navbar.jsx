import React from 'react';
import { Box, Layers, Activity, Cpu, Sparkles } from 'lucide-react';

export default function Navbar({ health, latency, activeTab, setActiveTab }) {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-6 flex items-center justify-between z-30 select-none">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2.5">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 via-indigo-600 to-cyan-400 p-[2px] shadow-lg shadow-sky-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Box className="w-5 h-5 text-sky-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                ArchiGen 3D
              </h1>
              <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
                AI HomeByMe
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Deep Learning Floorplan-to-3D & Facade Synthesis
            </p>
          </div>
        </div>

        {/* View Switcher */}
        <div className="ml-8 hidden md:flex items-center p-1 rounded-lg bg-slate-950 border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setActiveTab('split')}
            className={`px-3 py-1.5 rounded-md transition-all ${
              activeTab === 'split'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Split View (2D + 3D)
          </button>
          <button
            onClick={() => setActiveTab('2d')}
            className={`px-3 py-1.5 rounded-md transition-all ${
              activeTab === '2d'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2D Floorplan Analysis
          </button>
          <button
            onClick={() => setActiveTab('3d')}
            className={`px-3 py-1.5 rounded-md transition-all ${
              activeTab === '3d'
                ? 'bg-sky-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            3D Textured Interactive Scene
          </button>
        </div>
      </div>

      {/* Status Badges */}
      <div className="flex items-center space-x-3 text-xs">
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span>Backend:</span>
          <span className="font-semibold text-emerald-400">
            {health?.status === 'healthy' ? 'Online' : 'Checking...'}
          </span>
        </div>

        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span>GPU:</span>
          <span className="font-semibold text-indigo-400">
            RTX 3050 (CUDA)
          </span>
        </div>

        {latency !== null && (
          <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Inference:</span>
            <span className="font-semibold text-amber-400">{latency} ms</span>
          </div>
        )}
      </div>
    </header>
  );
}
