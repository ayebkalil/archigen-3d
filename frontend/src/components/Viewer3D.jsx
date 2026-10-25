import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw, Box, Eye, Layers, Sun } from 'lucide-react';

export default function Viewer3D({ layout, frontWallId, facadeTextureUrl, numFloors = 1, isSplit = false }) {
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
    scene.background = new THREE.Color(0xefebe2);
    scene.fog = new THREE.FogExp2(0xefebe2, 0.018);
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
    renderer.toneMappingExposure = 1.15;

    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // Prevent dipping below ground
    controlsRef.current = controls;

    // 5. Lighting (Warm Natural Architectural Studio Light)
    const ambientLight = new THREE.HemisphereLight(0xfffdf7, 0xd4cebf, 0.95);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff8eb, 1.35);
    sunLight.position.set(22, 38, 18);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.bias = -0.0001;
    scene.add(sunLight);

    // 6. Architectural Ground Grid
    const grid = new THREE.GridHelper(50, 50, 0xb8452a, 0xd8d2c4);
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
      // Shared unmirrored transform: X_shape = x - cx, Y_shape = -(y - cy)
      // With rotateX(-Math.PI / 2), this yields X_3d = x - cx, Z_3d = y - cy (det = +1)
      shape.moveTo(pts[0][0] - centerX, -(pts[0][1] - centerZ));
      for (let i = 1; i < pts.length; i++) {
        shape.lineTo(pts[i][0] - centerX, -(pts[i][1] - centerZ));
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
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });
      } else if (wall.is_exterior) {
        material = new THREE.MeshStandardMaterial({
          color: isFront ? 0xf59e0b : 0x0284c7,
          roughness: 0.5,
          metalness: 0.2,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });
      } else {
        material = new THREE.MeshStandardMaterial({
          color: 0x64748b,
          roughness: 0.9,
          side: THREE.DoubleSide,
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
        shape.moveTo(pts[0][0] - centerX, -(pts[0][1] - centerZ));
        for (let i = 1; i < pts.length; i++) {
          shape.lineTo(pts[i][0] - centerX, -(pts[i][1] - centerZ));
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
    cameraRef.current.position.set(0, 30, 0);
    cameraRef.current.up.set(0, 0, -1);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
    setViewPreset('top');
  };

  const setCameraPerspective = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(15, 20, 25);
    cameraRef.current.up.set(0, 1, 0);
    controlsRef.current.target.set(0, 2, 0);
    controlsRef.current.update();
    setViewPreset('perspective');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#efebe2] relative overflow-hidden select-none">
      {/* 3D Viewport Toolbar */}
      <div className="h-12 border-b border-[#1d1d1f]/15 bg-[#f7f4ed]/90 backdrop-blur-md px-4 flex items-center justify-between z-10 text-[#1d1d1f] select-none">
        <div className="flex items-center space-x-2.5 shrink-0">
          <Box className="w-3.5 h-3.5 text-[#f0500f] shrink-0" />
          <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] whitespace-nowrap">
            3D PERSPECTIVE MODEL
          </span>
          <span className="text-[#1d1d1f]/20">|</span>
          <span className="font-mono text-[9.5px] text-[#1d1d1f]/70 uppercase tracking-wider whitespace-nowrap">
            {facadeTextureUrl ? 'cGAN Neural Facade' : isSplit ? 'Watertight BIM' : 'Watertight BIM Geometry & Extrusions'}
          </span>
        </div>

        {/* View Controls */}
        <div className="flex items-center bg-white/80 border border-[#1d1d1f]/20 rounded-full p-1 space-x-1 shadow-sm shrink-0 font-stencil text-[10px] uppercase tracking-wider">
          <button
            onClick={() => setWireframe(!wireframe)}
            className={`h-7 px-3 rounded-full border transition-all whitespace-nowrap shrink-0 ${
              wireframe
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/70 border-[#1d1d1f]/20 text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-white'
            }`}
          >
            WIREFRAME
          </button>
          <button
            onClick={() => setShowFloors(!showFloors)}
            className={`h-7 px-3 rounded-full border transition-all whitespace-nowrap shrink-0 ${
              showFloors
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/70 border-[#1d1d1f]/20 text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-white'
            }`}
          >
            FLOORS
          </button>
          <button
            onClick={viewPreset === 'top' ? setCameraPerspective : setCameraTopDown}
            className="h-7 px-3.5 rounded-full border border-[#1d1d1f]/25 bg-white/80 text-[#1d1d1f] hover:bg-[#1d1d1f] hover:text-[#efebe2] transition-all whitespace-nowrap shrink-0"
          >
            {viewPreset === 'top' ? 'PERSPECTIVE VIEW' : 'TOP-DOWN VIEW'}
          </button>
        </div>
      </div>

      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="flex-1 w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Instructions */}
      <div className="absolute bottom-4 right-4 bg-[#f7f4ed]/90 backdrop-blur-md border border-[#1d1d1f]/20 rounded-full px-4 py-2 font-mono text-[9.5px] uppercase tracking-wider text-[#1d1d1f]/70 shadow-lg pointer-events-none select-none">
        <div className="flex items-center space-x-3">
          <span>Left-Click: Rotate</span>
          <span className="opacity-40">·</span>
          <span>Right-Click: Pan</span>
          <span className="opacity-40">·</span>
          <span>Scroll: Zoom</span>
        </div>
      </div>
    </div>
  );
}
