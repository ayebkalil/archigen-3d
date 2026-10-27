import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { RotateCcw, Box, Eye, Layers, Sun, Sparkles, Compass, Maximize2, Activity } from 'lucide-react';

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
  const [viewPreset, setViewPreset] = useState('perspective'); // 'perspective' | 'isometric' | 'top' | 'room'
  const [showFurniture, setShowFurniture] = useState(true);
  const [currentFps, setCurrentFps] = useState(60);
  const [graphicsQuality, setGraphicsQuality] = useState('high'); // 'high' | 'low'

  // Store layout centers for furniture positioning
  const centersRef = useRef({ centerX: 0, centerZ: 0 });

  // PBR texture cache ref
  const pbrTexturesRef = useRef({
    loaded: false,
    oak: null,
    marble: null,
    stucco: null
  });

  const sunLightRef = useRef(null);
  const shadowPlaneRef = useRef(null);
  const envMapRef = useRef(null);

  // 1. Scene Setup & Lifecycle
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    // Warm architectural cream canvas background
    scene.background = new THREE.Color(0xf1ede4);
    scene.fog = new THREE.FogExp2(0xf1ede4, 0.012);
    sceneRef.current = scene;

    // Furniture container group
    const furnitureGroup = new THREE.Group();
    furnitureGroup.name = "furnitureGroup";
    scene.add(furnitureGroup);
    furnitureGroupRef.current = furnitureGroup;

    // Camera Setup (Natural 45° Architectural FOV)
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(16, 19, 23);
    cameraRef.current = camera;

    // WebGL Renderer with ACES Filmic Tone Mapping and Soft Shadows
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // Prevent dipping below ground
    controlsRef.current = controls;

    // Environment Map (IBL Realistic Studio Reflections)
    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    pmremGenerator.compileEquirectangularShader();
    new RGBELoader().load(
      '/showcase/sky/qwantani_late_afternoon_puresky_1k.hdr',
      (hdrTexture) => {
        const envMap = pmremGenerator.fromEquirectangular(hdrTexture).texture;
        envMapRef.current = envMap;
        if (graphicsQuality === 'high') {
          scene.environment = envMap;
        }
        hdrTexture.dispose();
        pmremGenerator.dispose();
      },
      undefined,
      (err) => {
        console.warn('HDRI environment map fallback to procedural illumination:', err);
      }
    );

    // Warm Architectural Studio Lighting
    const hemiLight = new THREE.HemisphereLight(0xfffdf7, 0xdcd6cb, 0.85);
    scene.add(hemiLight);

    // Directional Key Sun Light
    const sunLight = new THREE.DirectionalLight(0xfff8eb, 1.45);
    sunLight.position.set(24, 40, 20);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.radius = 2.4; // Soft penumbra
    sunLight.shadow.bias = -0.00015;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 130;
    sunLight.shadow.camera.left = -28;
    sunLight.shadow.camera.right = 28;
    sunLight.shadow.camera.top = 28;
    sunLight.shadow.camera.bottom = -28;
    scene.add(sunLight);
    sunLightRef.current = sunLight;

    // Secondary Soft Cool Rim/Fill Light (ambient shadow illumination)
    const rimLight = new THREE.DirectionalLight(0xdbeafe, 0.40);
    rimLight.position.set(-22, 16, -20);
    scene.add(rimLight);

    // Ground Contact Shadow Receiver Plane
    const shadowPlaneGeo = new THREE.PlaneGeometry(80, 80);
    const shadowPlaneMat = new THREE.ShadowMaterial({ opacity: 0.22 });
    const shadowPlane = new THREE.Mesh(shadowPlaneGeo, shadowPlaneMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.005;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);
    shadowPlaneRef.current = shadowPlane;

    // Architectural Ground Grid
    const grid = new THREE.GridHelper(50, 50, 0xb8452a, 0xd8d2c4);
    grid.position.y = -0.01;
    scene.add(grid);

    // Load PBR Textures
    const texLoader = new THREE.TextureLoader();
    const loadTex = (url, isColor = false) => {
      const tex = texLoader.load(url);
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      if (isColor) tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    };

    pbrTexturesRef.current = {
      loaded: true,
      oak: {
        map: loadTex('/showcase/tex-1k/oak_wood_planks/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/oak_wood_planks/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/oak_wood_planks/arm.jpg')
      },
      marble: {
        map: loadTex('/showcase/tex-1k/marble_01/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/marble_01/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/marble_01/arm.jpg')
      },
      stucco: {
        map: loadTex('/showcase/tex-1k/white_stucco/diff.jpg', true),
        normalMap: loadTex('/showcase/tex-1k/white_stucco/nor.jpg'),
        arm: loadTex('/showcase/tex-1k/white_stucco/arm.jpg')
      }
    };

    window.THREE = THREE;
    window.__ARCHIGEN_VIEWER__ = {
      scene,
      camera,
      controls,
      renderer,
      pbrTexturesRef,
      sunLightRef,
      shadowPlaneRef,
      envMapRef,
      setGraphicsQuality: (q) => setGraphicsQuality(q),
      getGraphicsQuality: () => graphicsQuality
    };

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

    // Animation & Live FPS Measurement Loop
    let animationFrameId;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    const animate = (now) => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);

      frameCount++;
      if (now - lastFpsTime >= 500) {
        const measured = Math.round((frameCount * 1000) / (now - lastFpsTime));
        setCurrentFps(measured);
        frameCount = 0;
        lastFpsTime = now;
      }
    };
    animate(performance.now());

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

  // Handle Graphics Quality Mode (High: PBR, HDRI, Soft Shadows vs Low: Fast unshadowed diffuse for low-spec/CPU)
  useEffect(() => {
    if (!rendererRef.current || !sceneRef.current) return;
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const isHigh = graphicsQuality === 'high';

    renderer.shadowMap.enabled = isHigh;
    if (sunLightRef.current) {
      sunLightRef.current.castShadow = isHigh;
    }
    if (shadowPlaneRef.current) {
      shadowPlaneRef.current.visible = isHigh;
    }
    if (envMapRef.current) {
      scene.environment = isHigh ? envMapRef.current : null;
    }
    renderer.toneMapping = isHigh ? THREE.ACESFilmicToneMapping : THREE.LinearToneMapping;
    renderer.toneMappingExposure = isHigh ? 1.05 : 1.0;
    renderer.shadowMap.needsUpdate = true;
  }, [graphicsQuality]);

  // Update Building Meshes when Layout / Facade Texture / selectedRoom / graphicsQuality changes
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
      facadeTexture.colorSpace = THREE.SRGBColorSpace;
      facadeTexture.wrapS = THREE.ClampToEdgeWrapping;
      facadeTexture.wrapT = THREE.ClampToEdgeWrapping;
    }

    const pbr = pbrTexturesRef.current;

    // A. Extrude Walls with PBR Plaster Stucco Material
    walls.forEach((wall) => {
      const pts = wall.polygon_meters;
      if (!pts || pts.length < 3) return;

      const shape = new THREE.Shape();
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
          roughness: 0.65,
          metalness: 0.05,
          envMapIntensity: 0.9,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });
      } else if (graphicsQuality === 'high' && pbr.stucco?.map) {
        // High quality PBR Stucco with Normal & AO
        const sMap = pbr.stucco.map.clone();
        sMap.repeat.set(3, 1.5);
        sMap.needsUpdate = true;
        const sNor = pbr.stucco.normalMap.clone();
        sNor.repeat.set(3, 1.5);
        sNor.needsUpdate = true;
        const sArm = pbr.stucco.arm.clone();
        sArm.repeat.set(3, 1.5);
        sArm.needsUpdate = true;

        material = new THREE.MeshStandardMaterial({
          map: sMap,
          normalMap: sNor,
          normalScale: new THREE.Vector2(0.45, 0.45),
          roughnessMap: sArm,
          aoMap: sArm,
          aoMapIntensity: 1.1,
          roughness: 0.85,
          metalness: 0.02,
          envMapIntensity: 0.8,
          color: wall.is_exterior
            ? (isFront ? 0xf59e0b : 0xe2e8f0)
            : 0xf8fafc,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });
      } else {
        material = new THREE.MeshStandardMaterial({
          color: wall.is_exterior ? (isFront ? 0xf59e0b : 0x94a3b8) : 0xe2e8f0,
          roughness: 0.85,
          metalness: 0.05,
          side: THREE.DoubleSide,
          wireframe: wireframe,
        });
      }

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = (graphicsQuality === 'high');
      mesh.receiveShadow = (graphicsQuality === 'high');
      mesh.userData = { isBuildingMesh: true, wallId: wall.id };
      scene.add(mesh);
    });

    // B. Floor Slabs for Rooms with Room-Specific PBR Materials
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

        // Approximate room bounds for texture repeat scale
        let rMinX = Infinity, rMaxX = -Infinity, rMinZ = Infinity, rMaxZ = -Infinity;
        pts.forEach(([x, z]) => {
          if (x < rMinX) rMinX = x;
          if (x > rMaxX) rMaxX = x;
          if (z < rMinZ) rMinZ = z;
          if (z > rMaxZ) rMaxZ = z;
        });
        const rW = Math.max(2, rMaxX - rMinX);
        const rD = Math.max(2, rMaxZ - rMinZ);

        const isSelected = selectedRoom?.id === room.id;
        const normType = (room.room_type || '').toLowerCase();
        const isWetArea = normType.includes('bath') || normType.includes('toilet') || normType.includes('kitchen');

        let floorMat;

        if (graphicsQuality === 'high' && isWetArea && pbr.marble?.map) {
          // Luxury Architectural Polished Marble Tile PBR
          const mMap = pbr.marble.map.clone();
          mMap.repeat.set(Math.max(1, rW / 1.6), Math.max(1, rD / 1.6));
          mMap.needsUpdate = true;
          const mNor = pbr.marble.normalMap.clone();
          mNor.repeat.set(Math.max(1, rW / 1.6), Math.max(1, rD / 1.6));
          mNor.needsUpdate = true;
          const mArm = pbr.marble.arm.clone();
          mArm.repeat.set(Math.max(1, rW / 1.6), Math.max(1, rD / 1.6));
          mArm.needsUpdate = true;

          floorMat = new THREE.MeshStandardMaterial({
            map: mMap,
            normalMap: mNor,
            normalScale: new THREE.Vector2(0.5, 0.5),
            roughnessMap: mArm,
            aoMap: mArm,
            aoMapIntensity: 1.25,
            roughness: 0.30,
            metalness: 0.05,
            envMapIntensity: 1.15,
            color: isSelected ? 0xd0e8ff : 0xffffff,
            emissive: isSelected ? 0x0284c7 : 0x000000,
            emissiveIntensity: isSelected ? 0.28 : 0.0,
            side: THREE.DoubleSide,
            wireframe: wireframe,
          });
        } else if (graphicsQuality === 'high' && pbr.oak?.map) {
          // Warm Architectural Oak Wood Planks PBR
          const oMap = pbr.oak.map.clone();
          oMap.repeat.set(Math.max(1, rW / 1.8), Math.max(1, rD / 1.8));
          oMap.needsUpdate = true;
          const oNor = pbr.oak.normalMap.clone();
          oNor.repeat.set(Math.max(1, rW / 1.8), Math.max(1, rD / 1.8));
          oNor.needsUpdate = true;
          const oArm = pbr.oak.arm.clone();
          oArm.repeat.set(Math.max(1, rW / 1.8), Math.max(1, rD / 1.8));
          oArm.needsUpdate = true;

          floorMat = new THREE.MeshStandardMaterial({
            map: oMap,
            normalMap: oNor,
            normalScale: new THREE.Vector2(0.75, 0.75),
            roughnessMap: oArm,
            aoMap: oArm,
            aoMapIntensity: 1.15,
            roughness: 0.70,
            metalness: 0.02,
            envMapIntensity: 0.95,
            color: isSelected ? 0xd0e8ff : 0xffffff,
            emissive: isSelected ? 0x0284c7 : 0x000000,
            emissiveIntensity: isSelected ? 0.28 : 0.0,
            side: THREE.DoubleSide,
            wireframe: wireframe,
          });
        } else {
          // Low Quality: Fast untextured architectural tones
          floorMat = new THREE.MeshStandardMaterial({
            color: isSelected ? 0xd0e8ff : (isWetArea ? 0xe2e8f0 : 0xdfd3c3),
            emissive: isSelected ? 0x0284c7 : 0x000000,
            emissiveIntensity: isSelected ? 0.28 : 0.0,
            roughness: 0.85,
            metalness: 0.0,
            side: THREE.DoubleSide,
            wireframe: wireframe,
          });
        }

        const floorMesh = new THREE.Mesh(floorGeo, floorMat);
        floorMesh.position.y = 0.02;
        floorMesh.receiveShadow = (graphicsQuality === 'high');
        floorMesh.userData = { isBuildingMesh: true, isFloorMesh: true, room: room };
        scene.add(floorMesh);
      });
    }

    // Default framing
    const maxDim = Math.max(maxX - minX, maxZ - minZ) || 15;
    if (controlsRef.current && cameraRef.current && viewPreset === 'perspective') {
      controlsRef.current.target.set(0, wallHeight / 3, 0);
      cameraRef.current.position.set(maxDim * 1.15, maxDim * 1.35, maxDim * 1.35);
      controlsRef.current.update();
    }
  }, [layout, frontWallId, facadeTextureUrl, numFloors, wireframe, showFloors, selectedRoom, graphicsQuality]);

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
            const isRug = item.category === 'rug' || item.item_id?.includes('rug');
            // Floor slab is at y = 0.02.
            // Furniture sits on top of floor slab (y = 0.02).
            // Rug is elevated to y = 0.023 with polygonOffset to strictly eliminate z-fighting with floor meshes.
            const posY = isRug ? 0.023 : 0.02;
            model.position.set(posX, posY, posZ);

            // Rotation convention: rotation.y = (180.0 - alpha_deg) * (pi / 180)
            model.rotation.y = (180.0 - item.rotation_deg) * (Math.PI / 180.0);

            model.traverse((child) => {
              if (child.isMesh) {
                child.castShadow = (graphicsQuality === 'high' && !isRug);
                child.receiveShadow = (graphicsQuality === 'high');
                if (child.material) {
                  child.material.envMapIntensity = (graphicsQuality === 'high' ? 1.0 : 0.0);
                  if (isRug) {
                    child.material.polygonOffset = true;
                    child.material.polygonOffsetFactor = -2.0;
                    child.material.polygonOffsetUnits = -4.0;
                  }
                }
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
  }, [furnishingData, showFurniture, layout, graphicsQuality]);

  // View Presets: 3-4 Distinct Architectural Camera Modes
  const setCameraPerspective = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(16, 18, 22);
    cameraRef.current.up.set(0, 1, 0);
    controlsRef.current.target.set(0, 1.5, 0);
    controlsRef.current.update();
    setViewPreset('perspective');
  };

  const setCameraIsometric = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    // True 45° architectural axonometric view
    cameraRef.current.position.set(22, 22, 22);
    cameraRef.current.up.set(0, 1, 0);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
    setViewPreset('isometric');
  };

  const setCameraTopDown = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, 32, 0);
    cameraRef.current.up.set(0, 0, -1);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
    setViewPreset('top');
  };

  const setCameraRoomFocus = () => {
    if (!cameraRef.current || !controlsRef.current || !selectedRoom) return;
    const pts = selectedRoom.polygon_meters || [];
    if (!pts.length) return;
    let rx = 0, rz = 0;
    pts.forEach(([x, z]) => { rx += x; rz += z; });
    rx /= pts.length;
    rz /= pts.length;
    const { centerX, centerZ } = centersRef.current;
    const targetX = rx - centerX;
    const targetZ = rz - centerZ;
    controlsRef.current.target.set(targetX, 0.8, targetZ);
    cameraRef.current.position.set(targetX + 5.0, 4.2, targetZ + 5.0);
    cameraRef.current.up.set(0, 1, 0);
    controlsRef.current.update();
    setViewPreset('room');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#f1ede4] relative overflow-hidden select-none font-mono">
      {/* 3D Viewport Toolbar */}
      <div className="h-12 border-b border-[#1d1d1f]/15 bg-[#f7f4ed]/90 backdrop-blur-md px-4 flex items-center justify-between z-10 text-[#1d1d1f] select-none">
        <div className="flex items-center space-x-2.5 shrink-0">
          <Box className="w-3.5 h-3.5 text-[#b8452a] shrink-0" />
          <span className="font-semibold text-xs uppercase tracking-wider text-[#1d1d1f] whitespace-nowrap">
            3D PBR STUDIO
          </span>
          <span className="text-[#1d1d1f]/20">|</span>
          <div className="flex items-center gap-1.5 text-[9.5px] text-[#1d1d1f]/80 uppercase tracking-wider">
            <span>HDRI Environment</span>
            <span>&bull;</span>
            <span>Soft Shadows (PCF)</span>
            <span>&bull;</span>
            <span>ACES Tone Mapping</span>
          </div>
        </div>

        {/* Viewport Control Buttons & Camera Presets */}
        <div className="flex items-center space-x-1.5">
          {/* Real-time FPS Monitor */}
          <div className="flex items-center gap-1 px-2 py-0.5 rounded-sm bg-[#1d1d1f]/5 border border-[#1d1d1f]/10 text-[9.5px] text-[#2a7a6e] font-semibold" title="Live rendering frame rate on local GPU">
            <Activity className="w-3 h-3 text-[#2a7a6e] animate-pulse" />
            <span>{currentFps} FPS</span>
          </div>

          <div className="h-4 w-px bg-[#1d1d1f]/15 mx-1" />

          {/* Furniture Toggle */}
          <button
            onClick={() => setShowFurniture(!showFurniture)}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              showFurniture
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle placed 3D furniture models"
          >
            <Sparkles className="w-3 h-3 text-[#b8452a]" />
            <span>Furniture</span>
          </button>

          {/* Preset 1: Perspective */}
          <button
            onClick={setCameraPerspective}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'perspective'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="3D Perspective View"
          >
            <Eye className="w-3 h-3" />
            <span>3D</span>
          </button>

          {/* Preset 2: Isometric / Axonometric */}
          <button
            onClick={setCameraIsometric}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'isometric'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="45° Axonometric / Isometric View"
          >
            <Compass className="w-3 h-3" />
            <span>Axo</span>
          </button>

          {/* Preset 3: Top-Down Plan */}
          <button
            onClick={setCameraTopDown}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'top'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Top-Down Plan View"
          >
            <Layers className="w-3 h-3" />
            <span>Plan</span>
          </button>

          {/* Preset 4: Room Focus (only enabled when room selected) */}
          {selectedRoom && (
            <button
              onClick={setCameraRoomFocus}
              className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
                viewPreset === 'room'
                  ? 'bg-[#b8452a] text-[#efebe2] border-[#b8452a]'
                  : 'bg-white/60 text-[#b8452a] border-[#b8452a]/30 hover:bg-[#b8452a]/10'
              }`}
              title="Focus Camera Directly into Selected Room"
            >
              <Maximize2 className="w-3 h-3" />
              <span>Focus</span>
            </button>
          )}

          {/* Graphics Quality Toggle (High: PBR/HDRI/Shadows vs Low: Fast unshadowed diffuse for low-spec/CPU) */}
          <button
            onClick={() => setGraphicsQuality(graphicsQuality === 'high' ? 'low' : 'high')}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              graphicsQuality === 'high'
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle graphics quality (High: PBR & Soft Shadows / Low: Fast unshadowed diffuse for CPU)"
          >
            <Sun className="w-3 h-3 text-[#b8452a]" />
            <span>{graphicsQuality === 'high' ? 'PBR' : 'Low'}</span>
          </button>

          {/* Wireframe Toggle */}
          <button
            onClick={() => setWireframe(!wireframe)}
            className={`px-1.5 py-1 rounded-sm text-[10px] font-medium transition-colors border ${
              wireframe
                ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                : 'bg-white/60 text-[#1d1d1f]/60 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle Wireframe overlay"
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
