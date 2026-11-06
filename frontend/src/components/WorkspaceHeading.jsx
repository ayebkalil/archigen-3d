import React from 'react';
import { Columns2, Layers, Box } from 'lucide-react';

export default function WorkspaceHeading({ activeTab, setActiveTab, isAnalyzing, hasLayout }) {
  return (
    <div className="workspace-heading">
      <div>
        <div className="workspace-eyebrow">
          <span>ARCHIGEN 3D</span>
          <span className="dot">·</span>
          <span>NEURAL 2D-TO-3D STUDIO</span>
          <span className="dot">·</span>
          <span>WORKSPACE 01</span>
        </div>
        <h1>
          FLOORPLAN <span>WORKSPACE.</span>
        </h1>
        <p>Refine your plan. Explore the space it becomes.</p>
      </div>
      <div className="heading-controls">
        <span className="workspace-note" role="status">
          {isAnalyzing
            ? 'Detecting rooms and walls…'
            : hasLayout
              ? 'Select a room to start furnishing'
              : 'Upload a plan to get started'}
        </span>
        <div className="view-switch" aria-label="Workspace view">
          {[
            ['split', Columns2, 'Split view'],
            ['2d', Layers, '2D plan'],
            ['3d', Box, '3D house'],
          ].map(([id, Icon, label]) => (
            <button
              key={id}
              data-testid={`view-${id}`}
              aria-pressed={activeTab === id}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
