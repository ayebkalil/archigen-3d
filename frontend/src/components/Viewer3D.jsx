import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RotateCcw, Box, Eye, Layers, Sun, Sparkles } from 'lucide-react';

export default function Viewer3D({
  layout,
  frontWallId,
  facadeTextureUrl,
  numFloors = 1,
  isSplit = false,
  selectedRoom = null,
  onSelectRoom = null,
  furnishingData = {},
  onClearRoomFurnishings = null
}) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const furnitureGroupRef = useRef(null);
  const gltfLoaderRef = useRef(new GLTFLoader());

  const [wireframe, setWireframe] = useState(false);
  const [showFloors, setShowFloors] = useState(true);
  const [viewPreset, setViewPreset] = useState('perspective'); // 'perspective' | 'top'
  const [showFurniture, setShowFurniture] = useState(true);

  // Store layout centers for furniture positioning
  const centersRef = useRef({ centerX: 0, centerZ: 0 });

  // 1. Scene Setup
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xefebe2);
    scene.fog = new THREE.FogExp2(0xefebe2, 0.018);
    sceneRef.current = scene;

    // Furniture container group
    const furnitureGroup = new THREE.Group();
    furnitureGroup.name = "furnitureGroup";
    scene.add(furnitureGroup);
    furnitureGroupRef.current = furnitureGroup;

    // Camera Setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(15, 20, 25);
    cameraRef.current = camera;

    // WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // Prevent dipping below ground
    controlsRef.current = controls;

    // Lighting (Warm Natural Architectural Studio Light)
    const ambientLight = new THREE.HemisphereLight(0xfffdf7, 0xd4cebf, 0.95);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xfff8eb, 1.35);
    sunLight.position.set(22, 38, 18);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.bias = -0.0001;
    scene.add(sunLight);

    // Architectural Ground Grid
    const grid = new THREE.GridHelper(50, 50, 0xb8452a, 0xd8d2c4);
    grid.position.y = -0.01;
    scene.add(grid);

    // Raycaster for room clicking in 3D
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handleClick = (e) => {
      if (!containerRef.current || !cameraRef.current || !sceneRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(sceneRef.current.children, true);

      for (const hit of intersects) {
        if (hit.object.userData?.isFloorMesh && hit.object.userData?.room) {
          if (onSelectRoom) {
            onSelectRoom(hit.object.userData.room);
          }
          break;
        }
      }
    };

    renderer.domElement.addEventListener('pointerdown', handleClick);

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
      if (renderer.domElement) {
        renderer.domElement.removeEventListener('pointerdown', handleClick);
      }
      renderer.dispose();
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, []);

  // Update Building Meshes when Layout / Facade Texture / selectedRoom changes
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
    centersRef.current = { centerX, centerZ };

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

        const isSelected = selectedRoom?.id === room.id;
        const floorMat = new THREE.MeshStandardMaterial({
          color: isSelected ? 0x0284c7 : 0x1e293b,
          emissive: isSelected ? 0x0369a1 : 0x000000,
          emissiveIntensity: isSelected ? 0.35 : 0.0,
          roughness: 0.8,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });

        const floorMesh = new THREE.Mesh(floorGeo, floorMat);
        floorMesh.position.y = 0.02;
        floorMesh.receiveShadow = true;
        floorMesh.userData = { isBuildingMesh: true, isFloorMesh: true, room: room };
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
  }, [layout, frontWallId, facadeTextureUrl, numFloors, wireframe, showFloors, selectedRoom]);

  // Update 3D Furniture Assets when furnishingData changes
  useEffect(() => {
    const furnitureGroup = furnitureGroupRef.current;
    if (!furnitureGroup) return;

    // Helper to completely free WebGL memory for an object
    const disposeHierarchy = (obj) => {
      obj.traverse((child) => {
        if (child.isMesh) {
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach((m) => {
                if (m.map) m.map.dispose();
                m.dispose();
              });
            } else {
              if (child.material.map) child.material.map.dispose();
              child.material.dispose();
            }
          }
        }
      });
    };

    // Remove all old furniture meshes and dispose WebGL memory
    while (furnitureGroup.children.length > 0) {
      const child = furnitureGroup.children[0];
      furnitureGroup.remove(child);
      disposeHierarchy(child);
    }

    if (!showFurniture || !furnishingData) return;

    const { centerX, centerZ } = centersRef.current;
    const loader = gltfLoaderRef.current;

    // Iterate through furnished rooms
    Object.entries(furnishingData).forEach(([roomId, designResult]) => {
      if (!designResult || !designResult.is_valid || !designResult.items) return;

      const roomItems = designResult.items;
      roomItems.forEach((item) => {
        if (!item.glb) return;

        loader.load(
          item.glb,
          (gltf) => {
            const model = gltf.scene;

            // Shared unmirrored transform: X_three = x_plan - centerX, Z_three = y_plan - centerZ
            const posX = item.position_m[0] - centerX;
            const posZ = item.position_m[1] - centerZ;
            model.position.set(posX, 0.0, posZ);

            // Rotation convention: rotation.y = (180.0 - alpha_deg) * (pi / 180)
            model.rotation.y = (180.0 - item.rotation_deg) * (Math.PI / 180.0);

            model.traverse((child) => {
              if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
              }
            });

            model.userData = {
              isFurnitureMesh: true,
              roomId: roomId,
              itemId: item.item_id
            };

            furnitureGroup.add(model);
          },
          undefined,
          (err) => {
            console.error('Failed to load furniture asset:', item.glb, err);
            // Fallback placeholder box
            const [w, d, h] = item.dimensions_m || [1, 1, 1];
            const boxGeo = new THREE.BoxGeometry(w, h, d);
            const boxMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8 });
            const boxMesh = new THREE.Mesh(boxGeo, boxMat);
            boxMesh.position.set(item.position_m[0] - centerX, h / 2, item.position_m[1] - centerZ);
            boxMesh.userData = { isFurnitureMesh: true, roomId, itemId: item.item_id };
            furnitureGroup.add(boxMesh);
          }
        );
      });
    });
  }, [furnishingData, showFurniture, layout]);

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

        {/* Viewport Control Buttons */}
        <div className="flex items-center space-x-2">
          {/* Furniture Toggle */}
          <button
            onClick={() => setShowFurniture(!showFurniture)}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
              showFurniture
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle placed 3D furniture models"
          >
            <Sparkles className="w-3 h-3 text-[#b8452a]" />
            <span>Furniture</span>
          </button>

          {/* Top Down View */}
          <button
            onClick={setCameraTopDown}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
              viewPreset === 'top'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Top</span>
          </button>

          {/* Perspective View */}
          <button
            onClick={setCameraPerspective}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
              viewPreset === 'perspective'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
          >
            <Eye className="w-3 h-3" />
            <span>3D</span>
          </button>

          {/* Wireframe Toggle */}
          <button
            onClick={() => setWireframe(!wireframe)}
            className={`px-2 py-1 rounded text-xs font-mono transition-colors border ${
              wireframe
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle Wireframe"
          >
            WF
          </button>
        </div>
      </div>

      {/* 3D WebGL Canvas Container */}
      <div ref={containerRef} className="flex-1 w-full h-full relative cursor-grab active:cursor-grabbing" />
    </div>
  );
}
