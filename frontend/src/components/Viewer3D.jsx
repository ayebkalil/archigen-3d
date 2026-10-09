import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw, Box, Eye, Layers, Sun } from 'lucide-react';

export default function Viewer3D({ layout, frontWallId, facadeTextureUrl, numFloors = 1 }) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  const [wireframe, setWireframe] = useState(false);
  const [showFloors, setShowFloors] = useState(true);
  const [viewPreset, setViewPreset] = useState('perspective'); // 'perspective' | 'top'

  useEffect(() => {
    if (!containerRef.current) return;

    // 1. Scene Setup
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    scene.fog = new THREE.FogExp2(0x090d16, 0.02);
    sceneRef.current = scene;

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(15, 20, 25);
    cameraRef.current = camera;

    // 3. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.05; // Prevent dipping below ground
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.HemisphereLight(0xffffff, 0x1e293b, 0.7);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.2);
    sunLight.position.set(20, 35, 15);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.bias = -0.0001;
    scene.add(sunLight);

    // 6. Ground Grid
    const grid = new THREE.GridHelper(50, 50, 0x38bdf8, 0x1e293b);
    grid.position.y = -0.01;
    scene.add(grid);

    // Animation Loop
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!containerRef.current || !renderer || !camera) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, []);

  // Update Scene Meshes when Layout / Facade Texture changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !layout) return;

    // Remove existing building meshes (tagged with isBuildingMesh)
    const toRemove = [];
    scene.traverse((obj) => {
      if (obj.userData?.isBuildingMesh) {
        toRemove.push(obj);
      }
    });
    toRemove.forEach((obj) => {
      scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
        else obj.material.dispose();
      }
    });

    const walls = layout.walls || [];
    const rooms = layout.rooms || [];
    const wallHeight = (layout.walls?.[0]?.height_m || 2.8) * numFloors;

    // Center building at origin (0, 0, 0)
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    walls.forEach((w) => {
      w.polygon_meters?.forEach(([x, z]) => {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
      });
    });

    const centerX = (minX + maxX) / 2 || 0;
    const centerZ = (minZ + maxZ) / 2 || 0;

    // Optional Facade Texture
    let facadeTexture = null;
    if (facadeTextureUrl) {
      const loader = new THREE.TextureLoader();
      facadeTexture = loader.load(facadeTextureUrl);
      facadeTexture.wrapS = THREE.ClampToEdgeWrapping;
      facadeTexture.wrapT = THREE.ClampToEdgeWrapping;
    }

    // A. Extrude Walls
    walls.forEach((wall) => {
      const pts = wall.polygon_meters;
      if (!pts || pts.length < 3) return;

      const shape = new THREE.Shape();
      shape.moveTo(pts[0][0] - centerX, pts[0][1] - centerZ);
      for (let i = 1; i < pts.length; i++) {
        shape.lineTo(pts[i][0] - centerX, pts[i][1] - centerZ);
      }
      shape.closePath();

      const extrudeSettings = {
        steps: 1,
        depth: wallHeight,
        bevelEnabled: false,
      };

      const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      // Orient geometry so Y is Up
      geometry.rotateX(-Math.PI / 2);

      let material;
      const isFront = wall.id === frontWallId;

      if (isFront && facadeTexture) {
        material = new THREE.MeshStandardMaterial({
          map: facadeTexture,
          roughness: 0.7,
          metalness: 0.1,
          wireframe: wireframe,
        });
      } else if (wall.is_exterior) {
        material = new THREE.MeshStandardMaterial({
          color: isFront ? 0xf59e0b : 0x0284c7,
          roughness: 0.5,
          metalness: 0.2,
          wireframe: wireframe,
        });
      } else {
        material = new THREE.MeshStandardMaterial({
          color: 0x64748b,
          roughness: 0.9,
          wireframe: wireframe,
        });
      }

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = { isBuildingMesh: true, wallId: wall.id };
      scene.add(mesh);
    });

    // B. Floor Slabs for Rooms
    if (showFloors) {
      rooms.forEach((room) => {
        const pts = room.polygon_meters;
        if (!pts || pts.length < 3) return;

        const shape = new THREE.Shape();
        shape.moveTo(pts[0][0] - centerX, pts[0][1] - centerZ);
        for (let i = 1; i < pts.length; i++) {
          shape.lineTo(pts[i][0] - centerX, pts[i][1] - centerZ);
        }
        shape.closePath();

        const floorGeo = new THREE.ShapeGeometry(shape);
        floorGeo.rotateX(-Math.PI / 2);

        const floorMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.8,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });

        const floorMesh = new THREE.Mesh(floorGeo, floorMat);
        floorMesh.position.y = 0.02;
        floorMesh.receiveShadow = true;
        floorMesh.userData = { isBuildingMesh: true };
        scene.add(floorMesh);
      });
    }

    // Position camera to view whole building nicely
    const maxDim = Math.max(maxX - minX, maxZ - minZ) || 15;
    if (controlsRef.current && cameraRef.current) {
      controlsRef.current.target.set(0, wallHeight / 2, 0);
      cameraRef.current.position.set(maxDim * 1.2, maxDim * 1.5, maxDim * 1.4);
      controlsRef.current.update();
    }
  }, [layout, frontWallId, facadeTextureUrl, numFloors, wireframe, showFloors]);

  // View Presets
  const setCameraTopDown = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, 30, 0.1);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
    setViewPreset('top');
  };

  const setCameraPerspective = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(15, 20, 25);
    controlsRef.current.target.set(0, 2, 0);
    controlsRef.current.update();
    setViewPreset('perspective');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 relative overflow-hidden">
      {/* 3D Viewport Toolbar */}
      <div className="h-12 border-b border-slate-800 bg-slate-900/60 px-4 flex items-center justify-between z-10">
        <div className="flex items-center space-x-2 text-xs text-slate-300">
          <Box className="w-3.5 h-3.5 text-sky-400" />
          <span className="font-semibold text-slate-100">3D Extruded Scene</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">
            {facadeTextureUrl ? 'Generated Facade Texture Active' : 'Procedural Material'}
          </span>
        </div>

        {/* View Controls */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            onClick={() => setWireframe(!wireframe)}
            className={`px-2.5 py-1 rounded border transition-colors ${
              wireframe
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            Wireframe
          </button>
          <button
            onClick={() => setShowFloors(!showFloors)}
            className={`px-2.5 py-1 rounded border transition-colors ${
              showFloors
                ? 'bg-sky-500/10 border-sky-500/30 text-sky-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            Floors
          </button>
          <button
            onClick={viewPreset === 'top' ? setCameraPerspective : setCameraTopDown}
            className="px-2.5 py-1 rounded border border-slate-800 bg-slate-900 text-slate-300 hover:bg-slate-800 transition-colors"
          >
            {viewPreset === 'top' ? 'Perspective View' : 'Top-Down View'}
          </button>
        </div>
      </div>

      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="flex-1 w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Instructions */}
      <div className="absolute bottom-4 right-4 bg-slate-900/80 backdrop-blur border border-slate-800 rounded-lg p-2.5 text-[11px] text-slate-400 shadow-xl pointer-events-none select-none">
        <div className="flex items-center space-x-3">
          <span>🖱️ Left-Click + Drag: Rotate</span>
          <span>·</span>
          <span>Right-Click + Drag: Pan</span>
          <span>·</span>
          <span>Scroll: Zoom</span>
        </div>
      </div>
    </div>
  );
}
