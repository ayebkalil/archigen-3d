import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { RotateCcw, Box, Eye, Layers, Sun, Sparkles, Compass, Maximize2, Activity, Home } from 'lucide-react';

export default function Viewer3D({
  layout,
  frontWallId,
  facadeTextureUrl,
  numFloors = 1,
  isSplit = false,
  selectedRoom = null,
  onSelectRoom = null,
  furnishingData = {},
  onClearRoomFurnishings = null,
  facadeFocusTrigger = 0
}) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const furnitureGroupRef = useRef(null);
  const roofGroupRef = useRef(null);
  const gltfLoaderRef = useRef(new GLTFLoader());

  const [wireframe, setWireframe] = useState(false);
  const [showFloors, setShowFloors] = useState(true);
  const [showRoof, setShowRoof] = useState(false);
  const [roofStyle, setRoofStyle] = useState('slab'); // 'slab' | 'pitched'
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

    // Roof container group (can be activated / deactivated)
    const roofGroup = new THREE.Group();
    roofGroup.name = "roofGroup";
    roofGroup.visible = false;
    scene.add(roofGroup);
    roofGroupRef.current = roofGroup;

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

  // Handle Roof Visibility Toggle
  useEffect(() => {
    if (roofGroupRef.current) {
      roofGroupRef.current.visible = showRoof;
    }
  }, [showRoof]);

  // Helper: Remove consecutive duplicate or degenerate points
  const cleanPolygonPoints = (pts) => {
    if (!pts || pts.length < 3) return pts || [];
    const result = [];
    for (let i = 0; i < pts.length; i++) {
      const curr = pts[i];
      const prev = result[result.length - 1];
      if (!prev || Math.hypot(curr[0] - prev[0], curr[1] - prev[1]) > 0.02) {
        result.push(curr);
      }
    }
    if (result.length > 2 && Math.hypot(result[0][0] - result[result.length - 1][0], result[0][1] - result[result.length - 1][1]) < 0.02) {
      result.pop();
    }
    return result;
  };

  // Helper: Buffer a 2D polygon outwards by offsetMeters (e.g. 0.22m for overhang eaves)
  const bufferPolygon2D = (pts, offsetMeters = 0.22) => {
    if (!pts || pts.length < 3) return pts;
    const n = pts.length;
    const result = [];
    for (let i = 0; i < n; i++) {
      const prev = pts[(i - 1 + n) % n];
      const curr = pts[i];
      const next = pts[(i + 1) % n];

      const dx1 = curr[0] - prev[0];
      const dz1 = curr[1] - prev[1];
      const len1 = Math.hypot(dx1, dz1) || 1e-4;
      const norm1X = -dz1 / len1;
      const norm1Z = dx1 / len1;

      const dx2 = next[0] - curr[0];
      const dz2 = next[1] - curr[1];
      const len2 = Math.hypot(dx2, dz2) || 1e-4;
      const norm2X = -dz2 / len2;
      const norm2Z = dx2 / len2;

      let bisectX = norm1X + norm2X;
      let bisectZ = norm1Z + norm2Z;
      const bisectLen = Math.hypot(bisectX, bisectZ) || 1e-4;
      bisectX /= bisectLen;
      bisectZ /= bisectLen;

      const dot = norm1X * bisectX + norm1Z * bisectZ;
      const scale = Math.min(2.0, Math.max(0.5, 1.0 / Math.max(0.2, dot)));

      result.push([
        curr[0] + bisectX * offsetMeters * scale,
        curr[1] + bisectZ * offsetMeters * scale
      ]);
    }
    return result;
  };

  // Helper: Group rooms by proximity and compute convex hull envelope per building unit
  const computeFallbackEnvelopes = (rooms) => {
    if (!rooms || rooms.length === 0) return [];
    const clusters = [];
    rooms.forEach((r) => {
      const pts = r.polygon_meters;
      if (!pts || pts.length < 3) return;
      let cx = 0, cz = 0;
      pts.forEach(p => { cx += p[0]; cz += p[1]; });
      cx /= pts.length; cz /= pts.length;

      let found = null;
      for (const cl of clusters) {
        for (const rm of cl) {
          if (Math.hypot(cx - rm.cx, cz - rm.cz) < 8.0) {
            found = cl;
            break;
          }
        }
        if (found) break;
      }
      if (found) {
        found.push({ room: r, cx, cz, pts });
      } else {
        clusters.push([{ room: r, cx, cz, pts }]);
      }
    });

    const cross2D = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);

    const envelopes = [];
    clusters.forEach((cl) => {
      const allPts = [];
      cl.forEach(item => item.pts.forEach(p => allPts.push([p[0], p[1]])));
      if (allPts.length < 3) return;

      allPts.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]);
      const lower = [];
      for (let p of allPts) {
        while (lower.length >= 2 && cross2D(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
          lower.pop();
        }
        lower.push(p);
      }
      const upper = [];
      for (let i = allPts.length - 1; i >= 0; i--) {
        const p = allPts[i];
        while (upper.length >= 2 && cross2D(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
          upper.pop();
        }
        upper.push(p);
      }
      upper.pop();
      lower.pop();
      const hull = lower.concat(upper);
      if (hull.length >= 3) {
        envelopes.push(hull);
      }
    });
    return envelopes;
  };

  // Helper: Snap opening (door / window) onto closest wall segment and compute orientation
  const findWallSegmentForOpening = (posM, walls) => {
    if (!posM || !walls || walls.length === 0) {
      return {
        center: posM || [0, 0],
        angleRad: 0,
        wallThickness: 0.18,
        wallId: null,
        isExterior: false
      };
    }

    const px = posM[0];
    const pz = posM[1];

    let bestLineDist = Infinity;
    let bestSeg = null;

    for (let wIdx = 0; wIdx < walls.length; wIdx++) {
      const wall = walls[wIdx];
      const pts = wall.polygon_meters;
      if (!pts || pts.length < 2) continue;

      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const dx = p2[0] - p1[0];
        const dz = p2[1] - p1[1];
        const L = Math.hypot(dx, dz);
        if (L < 0.15) continue; // Skip tiny end-caps

        // Distance from point to infinite line of segment
        const lineDist = Math.abs((px - p1[0]) * dz - (pz - p1[1]) * dx) / L;
        const t = ((px - p1[0]) * dx + (pz - p1[1]) * dz) / (L * L);
        const segEndDist = Math.max(0.0, -t * L, (t - 1.0) * L);

        if (lineDist < bestLineDist && segEndDist < 2.2) {
          bestLineDist = lineDist;
          const projX = p1[0] + t * dx;
          const projZ = p1[1] + t * dz;
          const angleRad = Math.atan2(-dz, dx);
          bestSeg = {
            center: [projX, projZ],
            angleRad: angleRad,
            wallThickness: wall.is_exterior ? 0.22 : 0.18,
            wallId: wall.id,
            isExterior: Boolean(wall.is_exterior)
          };
        }
      }
    }

    if (!bestSeg) {
      return {
        center: [px, pz],
        angleRad: 0,
        wallThickness: 0.18,
        wallId: null,
        isExterior: false
      };
    }

    return bestSeg;
  };

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

    // Shared Stucco / Wall Material for Wall Bodies, Corner Posts, Lintels & Sills
    let defaultWallMat;
    if (graphicsQuality === 'high' && pbr.stucco?.map) {
      const sMap = pbr.stucco.map.clone();
      sMap.repeat.set(1.5, 1);
      sMap.needsUpdate = true;
      const sNor = pbr.stucco.normalMap.clone();
      sNor.repeat.set(1.5, 1);
      sNor.needsUpdate = true;
      const sArm = pbr.stucco.arm.clone();
      sArm.repeat.set(1.5, 1);
      sArm.needsUpdate = true;
      defaultWallMat = new THREE.MeshStandardMaterial({
        map: sMap,
        normalMap: sNor,
        normalScale: new THREE.Vector2(0.45, 0.45),
        roughnessMap: sArm,
        aoMap: sArm,
        aoMapIntensity: 1.1,
        roughness: 0.85,
        metalness: 0.02,
        color: 0xf8fafc,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });
    } else {
      defaultWallMat = new THREE.MeshStandardMaterial({
        color: 0xe2e8f0,
        roughness: 0.85,
        metalness: 0.05,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });
    }

    // A. Extrude Walls with PBR Plaster Stucco Material (Cleaned & Solid Geometry)
    walls.forEach((wall) => {
      const pts = cleanPolygonPoints(wall.polygon_meters);
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

    // A2. Sims-Style Seamless Corner Junction Posts & Welding Columns
    // Physically and visually caps intersecting wall corners so there are ZERO cracks, air gaps, or disjointed slivers
    const junctionPositions = [];
    for (let i = 0; i < walls.length; i++) {
      const pts1 = walls[i].polygon_meters || [];
      for (let j = i + 1; j < walls.length; j++) {
        const pts2 = walls[j].polygon_meters || [];
        for (const p1 of pts1) {
          for (const p2 of pts2) {
            const dist = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
            if (dist < 0.38) {
              const jX = (p1[0] + p2[0]) / 2 - centerX;
              const jZ = (p1[1] + p2[1]) / 2 - centerZ;
              const alreadyHas = junctionPositions.some(jp => Math.hypot(jp.x - jX, jp.z - jZ) < 0.18);
              if (!alreadyHas) {
                junctionPositions.push({
                  x: jX,
                  z: jZ,
                  isExterior: Boolean(walls[i].is_exterior || walls[j].is_exterior)
                });
              }
            }
          }
        }
      }
    }

    junctionPositions.forEach((j) => {
      const postThick = j.isExterior ? 0.24 : 0.18;
      const postGeo = new THREE.BoxGeometry(postThick, wallHeight, postThick);
      const postMesh = new THREE.Mesh(postGeo, defaultWallMat);
      postMesh.position.set(j.x, wallHeight / 2, j.z);
      postMesh.castShadow = (graphicsQuality === 'high');
      postMesh.receiveShadow = (graphicsQuality === 'high');
      postMesh.userData = { isBuildingMesh: true, type: 'corner_junction' };
      scene.add(postMesh);
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

      // Ground Foundation Slab (Architectural Base Grounding for Building Units)
      const foundationEnvelopes = (layout.envelope && layout.envelope.length > 0)
        ? layout.envelope
        : computeFallbackEnvelopes(rooms);

      const foundationMat = new THREE.MeshStandardMaterial({
        color: 0xd9d3c7, // Architectural concrete foundation base
        roughness: 0.90,
        metalness: 0.02,
        side: THREE.DoubleSide
      });

      foundationEnvelopes.forEach((env) => {
        const cleanEnv = cleanPolygonPoints(env);
        if (cleanEnv.length < 3) return;
        const bufferedBase = bufferPolygon2D(cleanEnv, 0.12);

        const baseShape = new THREE.Shape();
        baseShape.moveTo(bufferedBase[0][0] - centerX, -(bufferedBase[0][1] - centerZ));
        for (let i = 1; i < bufferedBase.length; i++) {
          baseShape.lineTo(bufferedBase[i][0] - centerX, -(bufferedBase[i][1] - centerZ));
        }
        baseShape.closePath();

        const baseGeo = new THREE.ExtrudeGeometry(baseShape, {
          depth: 0.08,
          bevelEnabled: false
        });
        baseGeo.rotateX(-Math.PI / 2);

        const baseMesh = new THREE.Mesh(baseGeo, foundationMat);
        baseMesh.position.y = -0.06;
        baseMesh.receiveShadow = (graphicsQuality === 'high');
        baseMesh.userData = { isBuildingMesh: true, type: 'foundation_slab' };
        scene.add(baseMesh);
      });
    }

    const scaleMperPx = layout.metadata?.scale_meters_per_pixel || 0.03;
    const doors = layout.doors || [];
    const windows = layout.windows || [];

    // D. 3D Doors & Wall Lintels (Green Dots on Blueprint)
    doors.forEach((door, dIdx) => {
      const posM = door.position_meters || (door.position_px ? [door.position_px[0] * scaleMperPx, door.position_px[1] * scaleMperPx] : null);
      if (!posM) return;

      const segInfo = findWallSegmentForOpening(posM, walls);
      const [snappedXPlan, snappedZPlan] = segInfo.center;
      const snappedX = snappedXPlan - centerX;
      const snappedZ = snappedZPlan - centerZ;
      const angleRad = segInfo.angleRad;
      const wallThick = segInfo.wallThickness || 0.18;

      const doorWidth = Math.max(0.70, Math.min(door.width_m || 0.90, 1.80));
      const doorHeight = Math.min(2.15, wallHeight - 0.20);
      const lintelHeight = wallHeight - doorHeight;

      // 1. Wall Lintel above door (fills empty space with solid wall up to ceiling, embedded into flanking walls)
      if (lintelHeight > 0.05) {
        const lintelGeo = new THREE.BoxGeometry(doorWidth + 0.10, lintelHeight, wallThick + 0.02);
        const lintelMesh = new THREE.Mesh(lintelGeo, defaultWallMat);
        lintelMesh.position.set(
          snappedX,
          doorHeight + lintelHeight / 2,
          snappedZ
        );
        lintelMesh.rotation.y = angleRad;
        lintelMesh.castShadow = (graphicsQuality === 'high');
        lintelMesh.receiveShadow = (graphicsQuality === 'high');
        lintelMesh.userData = { isBuildingMesh: true, type: 'door_lintel', doorId: door.id || `door_${dIdx}` };
        scene.add(lintelMesh);
      }

      // 2. 3D Architectural Door (Frame, Hinged Leaf, Lever Handles, Floor Threshold)
      const doorGroup = new THREE.Group();
      doorGroup.position.set(snappedX, 0, snappedZ);
      doorGroup.rotation.y = angleRad;

      // Frame Material (Architectural Charcoal Wood)
      const frameMat = new THREE.MeshStandardMaterial({
        color: 0x221f1d,
        roughness: 0.65,
        metalness: 0.08
      });
      const frameThick = 0.045;
      const frameDepth = wallThick + 0.02;

      // Left Jamb
      const jambLGeo = new THREE.BoxGeometry(frameThick, doorHeight, frameDepth);
      const jambL = new THREE.Mesh(jambLGeo, frameMat);
      jambL.position.set(-doorWidth / 2 + frameThick / 2, doorHeight / 2, 0);
      jambL.castShadow = (graphicsQuality === 'high');
      jambL.receiveShadow = (graphicsQuality === 'high');
      doorGroup.add(jambL);

      // Right Jamb
      const jambRGeo = new THREE.BoxGeometry(frameThick, doorHeight, frameDepth);
      const jambR = new THREE.Mesh(jambRGeo, frameMat);
      jambR.position.set(doorWidth / 2 - frameThick / 2, doorHeight / 2, 0);
      jambR.castShadow = (graphicsQuality === 'high');
      jambR.receiveShadow = (graphicsQuality === 'high');
      doorGroup.add(jambR);

      // Top Header Casing
      const headGeo = new THREE.BoxGeometry(doorWidth, frameThick, frameDepth);
      const headMesh = new THREE.Mesh(headGeo, frameMat);
      headMesh.position.set(0, doorHeight - frameThick / 2, 0);
      headMesh.castShadow = (graphicsQuality === 'high');
      headMesh.receiveShadow = (graphicsQuality === 'high');
      doorGroup.add(headMesh);

      // Hinged Door Leaf (Warm Oak Finish)
      const leafWidth = doorWidth - frameThick * 2;
      const leafHeight = doorHeight - frameThick;
      const leafThick = 0.04;
      const leafMat = new THREE.MeshStandardMaterial({
        color: 0x9b6b43, // Architectural warm oak
        roughness: 0.55,
        metalness: 0.05
      });

      const hingePivot = new THREE.Group();
      hingePivot.position.set(-doorWidth / 2 + frameThick, 0, 0);
      hingePivot.rotation.y = -0.35; // ~20° slightly ajar for realistic look

      const leafGeo = new THREE.BoxGeometry(leafWidth, leafHeight, leafThick);
      const leafMesh = new THREE.Mesh(leafGeo, leafMat);
      leafMesh.position.set(leafWidth / 2, leafHeight / 2 + 0.01, 0);
      leafMesh.castShadow = (graphicsQuality === 'high');
      leafMesh.receiveShadow = (graphicsQuality === 'high');
      hingePivot.add(leafMesh);

      // Polished Brass Lever Handle
      const handleMat = new THREE.MeshStandardMaterial({
        color: 0xd4af37, // Polished brass hardware
        metalness: 0.85,
        roughness: 0.20
      });
      const leverGeo = new THREE.BoxGeometry(0.10, 0.02, 0.02);
      const leverF = new THREE.Mesh(leverGeo, handleMat);
      leverF.position.set(leafWidth - 0.07, 0.98, leafThick / 2 + 0.02);
      hingePivot.add(leverF);

      const leverB = new THREE.Mesh(leverGeo, handleMat);
      leverB.position.set(leafWidth - 0.07, 0.98, -leafThick / 2 - 0.02);
      hingePivot.add(leverB);

      doorGroup.add(hingePivot);

      // Floor threshold strip
      const threshGeo = new THREE.BoxGeometry(doorWidth, 0.015, wallThick);
      const threshMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.5, roughness: 0.4 });
      const threshMesh = new THREE.Mesh(threshGeo, threshMat);
      threshMesh.position.set(0, 0.008, 0);
      doorGroup.add(threshMesh);

      doorGroup.userData = { isBuildingMesh: true, type: 'door', doorId: door.id || `door_${dIdx}` };
      scene.add(doorGroup);
    });

    // E. 3D Windows, Wall Sills & Lintels (Blue Squares on Blueprint)
    windows.forEach((win, wIdx) => {
      const posM = win.position_meters || (win.position_px ? [win.position_px[0] * scaleMperPx, win.position_px[1] * scaleMperPx] : null);
      if (!posM) return;

      const segInfo = findWallSegmentForOpening(posM, walls);
      const [snappedXPlan, snappedZPlan] = segInfo.center;
      const snappedX = snappedXPlan - centerX;
      const snappedZ = snappedZPlan - centerZ;
      const angleRad = segInfo.angleRad;
      const wallThick = segInfo.wallThickness || 0.20;

      const winWidth = Math.max(0.60, Math.min(win.width_m || 1.10, 3.20));
      const sillHeight = Math.min(0.95, wallHeight * 0.35);
      const winHeight = Math.min(1.30, wallHeight - sillHeight - 0.35);
      const winTop = sillHeight + winHeight;
      const lintelHeight = wallHeight - winTop;

      // 1. Wall Sill below window (fills space from ground to sill, embedded into flanking walls)
      if (sillHeight > 0.05) {
        const sillGeo = new THREE.BoxGeometry(winWidth + 0.12, sillHeight, wallThick + 0.02);
        const sillMesh = new THREE.Mesh(sillGeo, defaultWallMat);
        sillMesh.position.set(snappedX, sillHeight / 2, snappedZ);
        sillMesh.rotation.y = angleRad;
        sillMesh.castShadow = (graphicsQuality === 'high');
        sillMesh.receiveShadow = (graphicsQuality === 'high');
        sillMesh.userData = { isBuildingMesh: true, type: 'window_sill_wall', winId: win.id || `win_${wIdx}` };
        scene.add(sillMesh);
      }

      // 2. Wall Lintel above window (fills space from window top to ceiling, embedded into flanking walls)
      if (lintelHeight > 0.05) {
        const lintelGeo = new THREE.BoxGeometry(winWidth + 0.12, lintelHeight, wallThick + 0.02);
        const lintelMesh = new THREE.Mesh(lintelGeo, defaultWallMat);
        lintelMesh.position.set(snappedX, winTop + lintelHeight / 2, snappedZ);
        lintelMesh.rotation.y = angleRad;
        lintelMesh.castShadow = (graphicsQuality === 'high');
        lintelMesh.receiveShadow = (graphicsQuality === 'high');
        lintelMesh.userData = { isBuildingMesh: true, type: 'window_lintel_wall', winId: win.id || `win_${wIdx}` };
        scene.add(lintelMesh);
      }

      // 3. 3D Architectural Window (Outer Frame, Mullion, Exterior Sill Shelf, Reflective Glass Panes)
      const winGroup = new THREE.Group();
      winGroup.position.set(snappedX, 0, snappedZ);
      winGroup.rotation.y = angleRad;

      const frameMat = new THREE.MeshStandardMaterial({
        color: 0x1e242b, // Architectural dark bronze / black aluminum
        metalness: 0.65,
        roughness: 0.35
      });
      const frameThick = 0.04;
      const frameDepth = wallThick * 0.70;

      // Frame profiles (bottom, top, left, right)
      const botGeo = new THREE.BoxGeometry(winWidth, frameThick, frameDepth);
      const botMesh = new THREE.Mesh(botGeo, frameMat);
      botMesh.position.set(0, sillHeight + frameThick / 2, 0);
      winGroup.add(botMesh);

      const topGeo = new THREE.BoxGeometry(winWidth, frameThick, frameDepth);
      const topMesh = new THREE.Mesh(topGeo, frameMat);
      topMesh.position.set(0, winTop - frameThick / 2, 0);
      winGroup.add(topMesh);

      const leftGeo = new THREE.BoxGeometry(frameThick, winHeight, frameDepth);
      const leftMesh = new THREE.Mesh(leftGeo, frameMat);
      leftMesh.position.set(-winWidth / 2 + frameThick / 2, sillHeight + winHeight / 2, 0);
      winGroup.add(leftMesh);

      const rightGeo = new THREE.BoxGeometry(frameThick, winHeight, frameDepth);
      const rightMesh = new THREE.Mesh(rightGeo, frameMat);
      rightMesh.position.set(winWidth / 2 - frameThick / 2, sillHeight + winHeight / 2, 0);
      winGroup.add(rightMesh);

      // Center Vertical Mullion
      const mullGeo = new THREE.BoxGeometry(0.035, winHeight - frameThick * 2, frameDepth);
      const mullMesh = new THREE.Mesh(mullGeo, frameMat);
      mullMesh.position.set(0, sillHeight + winHeight / 2, 0);
      winGroup.add(mullMesh);

      // Exterior Sill Shelf (protruding drip lip)
      const shelfGeo = new THREE.BoxGeometry(winWidth + 0.08, 0.035, wallThick + 0.06);
      const shelfMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.50, metalness: 0.30 });
      const shelfMesh = new THREE.Mesh(shelfGeo, shelfMat);
      shelfMesh.position.set(0, sillHeight - 0.015, 0.02);
      winGroup.add(shelfMesh);

      // Glass Panes (Translucent PBR Glass)
      const glassMat = (graphicsQuality === 'high')
        ? new THREE.MeshPhysicalMaterial({
            color: 0xe0f2fe,
            transparent: true,
            opacity: 0.35,
            roughness: 0.05,
            metalness: 0.1,
            transmission: 0.85,
            ior: 1.5,
            reflectivity: 0.9,
            envMapIntensity: 1.5,
            side: THREE.DoubleSide,
            depthWrite: false
          })
        : new THREE.MeshStandardMaterial({
            color: 0x93c5fd,
            transparent: true,
            opacity: 0.40,
            roughness: 0.1,
            metalness: 0.1,
            side: THREE.DoubleSide
          });

      const paneWidth = (winWidth - frameThick * 3) / 2;
      const paneHeight = winHeight - frameThick * 2;
      const paneGeo = new THREE.BoxGeometry(paneWidth, paneHeight, 0.01);

      const paneL = new THREE.Mesh(paneGeo, glassMat);
      paneL.position.set(-winWidth / 4 + frameThick / 4, sillHeight + winHeight / 2, 0);
      winGroup.add(paneL);

      const paneR = new THREE.Mesh(paneGeo, glassMat);
      paneR.position.set(winWidth / 4 - frameThick / 4, sillHeight + winHeight / 2, 0);
      winGroup.add(paneR);

      winGroup.userData = { isBuildingMesh: true, type: 'window', winId: win.id || `win_${wIdx}` };
      scene.add(winGroup);
    });

    // F. 3D Unified Architectural Roof (The Sims-Style Continuous Building Envelope)
    if (roofGroupRef.current) {
      const roofGroup = roofGroupRef.current;
      while (roofGroup.children.length > 0) {
        const child = roofGroup.children[0];
        roofGroup.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
          else child.material.dispose();
        }
      }

      // High-End Architectural Roof Materials
      const roofSlabMat = new THREE.MeshStandardMaterial({
        color: 0x222a36, // Deep architectural slate / dark anthracite
        roughness: 0.65,
        metalness: 0.12,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });

      const pitchedRoofMat = new THREE.MeshStandardMaterial({
        color: 0x2a3342, // Rich dark architectural shingles
        roughness: 0.70,
        metalness: 0.08,
        side: THREE.DoubleSide,
        wireframe: wireframe
      });

      const soffitMat = new THREE.MeshStandardMaterial({
        color: 0xf8fafc, // Clean off-white ceiling soffit underneath
        roughness: 0.90,
        metalness: 0.02,
        side: THREE.DoubleSide
      });

      // 1. Obtain Unified Building Envelope Polygons (from backend layout or fallback cluster hulls)
      let envelopeList = (layout.envelope && layout.envelope.length > 0)
        ? layout.envelope
        : computeFallbackEnvelopes(rooms);

      envelopeList.forEach((rawEnvPts, envIdx) => {
        if (!rawEnvPts || rawEnvPts.length < 3) return;

        // Clean & ensure valid points
        let envPts = cleanPolygonPoints(rawEnvPts);
        if (envPts.length < 3) return;

        // Check winding order (compute signed area in 2D plan space)
        let signedArea = 0;
        for (let i = 0; i < envPts.length; i++) {
          const p1 = envPts[i];
          const p2 = envPts[(i + 1) % envPts.length];
          signedArea += (p2[0] - p1[0]) * (p2[1] + p1[1]);
        }
        if (signedArea < 0) {
          envPts.reverse();
        }

        // Buffer outwards by 0.22m to create realistic architectural overhang eaves
        const bufferedPts = bufferPolygon2D(envPts, 0.22);

        // A. Ceiling Soffit Plate at wallHeight (closes rooms from below)
        const soffitShape = new THREE.Shape();
        soffitShape.moveTo(envPts[0][0] - centerX, -(envPts[0][1] - centerZ));
        for (let i = 1; i < envPts.length; i++) {
          soffitShape.lineTo(envPts[i][0] - centerX, -(envPts[i][1] - centerZ));
        }
        soffitShape.closePath();

        const soffitGeo = new THREE.ShapeGeometry(soffitShape);
        soffitGeo.rotateX(-Math.PI / 2);
        const soffitMesh = new THREE.Mesh(soffitGeo, soffitMat);
        soffitMesh.position.y = wallHeight - 0.005;
        soffitMesh.userData = { isBuildingMesh: true, isRoofMesh: true };
        roofGroup.add(soffitMesh);

        if (roofStyle === 'pitched') {
          // B1. Classic Sims Pitched Hipped Gable Roof
          let cX = 0, cZ = 0;
          bufferedPts.forEach(pt => { cX += pt[0]; cZ += pt[1]; });
          cX /= bufferedPts.length;
          cZ /= bufferedPts.length;

          let maxR = 0;
          bufferedPts.forEach(pt => {
            const r = Math.hypot(pt[0] - cX, pt[1] - cZ);
            if (r > maxR) maxR = r;
          });

          const pitchHeight = Math.min(2.5, Math.max(1.3, maxR * 0.35));
          const apexY = wallHeight + pitchHeight;
          const baseY = wallHeight;

          const apexX = cX - centerX;
          const apexZ = cZ - centerZ;

          const positions = [];
          const uvs = [];
          const numPts = bufferedPts.length;

          for (let i = 0; i < numPts; i++) {
            const p1 = bufferedPts[i];
            const p2 = bufferedPts[(i + 1) % numPts];
            const x1 = p1[0] - centerX;
            const z1 = p1[1] - centerZ;
            const x2 = p2[0] - centerX;
            const z2 = p2[1] - centerZ;

            // Outer triangle face
            positions.push(x1, baseY, z1);
            positions.push(x2, baseY, z2);
            positions.push(apexX, apexY, apexZ);
            uvs.push(0, 0, 1, 0, 0.5, 1);

            // Inward-facing backface
            positions.push(apexX, apexY, apexZ);
            positions.push(x2, baseY, z2);
            positions.push(x1, baseY, z1);
            uvs.push(0.5, 1, 1, 0, 0, 0);
          }

          const pitchedGeo = new THREE.BufferGeometry();
          pitchedGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
          pitchedGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
          pitchedGeo.computeVertexNormals();

          const pitchedMesh = new THREE.Mesh(pitchedGeo, pitchedRoofMat);
          pitchedMesh.castShadow = (graphicsQuality === 'high');
          pitchedMesh.receiveShadow = (graphicsQuality === 'high');
          pitchedMesh.userData = { isBuildingMesh: true, isRoofMesh: true };
          roofGroup.add(pitchedMesh);

        } else {
          // B2. Modern Architectural Slab Roof with Overhang Eaves & Fascia Trim
          const roofShape = new THREE.Shape();
          roofShape.moveTo(bufferedPts[0][0] - centerX, -(bufferedPts[0][1] - centerZ));
          for (let i = 1; i < bufferedPts.length; i++) {
            roofShape.lineTo(bufferedPts[i][0] - centerX, -(bufferedPts[i][1] - centerZ));
          }
          roofShape.closePath();

          const fasciaThick = 0.22;
          const roofGeo = new THREE.ExtrudeGeometry(roofShape, {
            depth: fasciaThick,
            bevelEnabled: true,
            bevelThickness: 0.03,
            bevelSize: 0.03,
            bevelSegments: 2
          });
          roofGeo.rotateX(-Math.PI / 2);

          const slabMesh = new THREE.Mesh(roofGeo, roofSlabMat);
          slabMesh.position.y = wallHeight;
          slabMesh.castShadow = (graphicsQuality === 'high');
          slabMesh.receiveShadow = (graphicsQuality === 'high');
          slabMesh.userData = { isBuildingMesh: true, isRoofMesh: true };
          roofGroup.add(slabMesh);
        }
      });

      roofGroup.visible = showRoof;
    }

    // Default framing
    const maxDim = Math.max(maxX - minX, maxZ - minZ) || 15;
    if (controlsRef.current && cameraRef.current && viewPreset === 'perspective') {
      controlsRef.current.target.set(0, wallHeight / 3, 0);
      cameraRef.current.position.set(maxDim * 1.15, maxDim * 1.35, maxDim * 1.35);
      controlsRef.current.update();
    }
  }, [layout, frontWallId, facadeTextureUrl, numFloors, wireframe, showFloors, selectedRoom, graphicsQuality, roofStyle]);

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

            // Apply compact scale if specified
            if (item.scale && Array.isArray(item.scale) && item.scale.length === 3) {
              model.scale.set(item.scale[0], item.scale[1], item.scale[2]);
            }

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

  const setCameraFacadeFocus = () => {
    if (!cameraRef.current || !controlsRef.current || !layout) return;
    const walls = layout.walls || [];
    const targetWall = walls.find(w => w.id === frontWallId) || walls.find(w => w.is_front) || walls.find(w => w.is_exterior) || walls[0];
    if (!targetWall || !targetWall.polygon_meters || targetWall.polygon_meters.length < 2) return;

    const { centerX = 0, centerZ = 0 } = centersRef.current || {};
    const pts = targetWall.polygon_meters;
    let wx = 0, wz = 0;
    pts.forEach(([x, z]) => { wx += x; wz += z; });
    wx /= pts.length;
    wz /= pts.length;

    const targetX = wx - centerX;
    const targetZ = wz - centerZ;

    // Outward normal computation
    const p1 = pts[0];
    const p2 = pts[1];
    const dx = p2[0] - p1[0];
    const dz = p2[1] - p1[1];
    const len = Math.hypot(dx, dz) || 1;
    let nx = -dz / len;
    let nz = dx / len;

    // Ensure normal points outward from center
    if (nx * targetX + nz * targetZ < 0) {
      nx = -nx;
      nz = -nz;
    }

    const camDist = Math.max(len * 1.3, 7.5);
    cameraRef.current.position.set(targetX + nx * camDist, (numFloors * 2.8) * 0.6, targetZ + nz * camDist);
    cameraRef.current.up.set(0, 1, 0);
    controlsRef.current.target.set(targetX, (numFloors * 2.8) * 0.45, targetZ);
    controlsRef.current.update();
    setViewPreset('facade');
  };

  useEffect(() => {
    if (facadeFocusTrigger > 0) {
      setCameraFacadeFocus();
    }
  }, [facadeFocusTrigger]);

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

          {/* Roof Activation / Deactivation Toggle */}
          <button
            onClick={() => setShowRoof(!showRoof)}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              showRoof
                ? 'bg-[#b8452a] text-[#efebe2] border-[#b8452a] shadow-sm'
                : 'bg-white/60 text-[#1d1d1f]/70 border-[#1d1d1f]/20 hover:border-[#1d1d1f]'
            }`}
            title="Toggle 3D Building Roof Enclosure (Activate / Deactivate)"
          >
            <Home className="w-3 h-3" />
            <span>Roof: {showRoof ? 'ON' : 'OFF'}</span>
          </button>

          {/* Roof Style Selector: Modern Flat Slab vs Classic Sims Pitched Hip */}
          {showRoof && (
            <div className="flex items-center space-x-1 bg-white/80 px-1 py-0.5 rounded border border-[#1d1d1f]/20 shadow-xs">
              <button
                onClick={() => setRoofStyle('slab')}
                className={`px-1.5 py-0.5 text-[9.5px] rounded transition-colors ${
                  roofStyle === 'slab' ? 'bg-[#1d1d1f] text-[#efebe2] font-semibold' : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f]'
                }`}
                title="Modern Architectural Flat Slab Roof with Overhang Eaves & Fascia Band"
              >
                Flat
              </button>
              <button
                onClick={() => setRoofStyle('pitched')}
                className={`px-1.5 py-0.5 text-[9.5px] rounded transition-colors ${
                  roofStyle === 'pitched' ? 'bg-[#b8452a] text-[#efebe2] font-semibold' : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f]'
                }`}
                title="Classic Sims Pitched Hipped Gable Roof"
              >
                Pitched
              </button>
            </div>
          )}

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

          {/* Preset 5: Facade Focus */}
          <button
            onClick={setCameraFacadeFocus}
            className={`flex items-center space-x-1 px-2 py-1 rounded-sm text-[10.5px] font-medium transition-colors border ${
              viewPreset === 'facade'
                ? 'bg-[#f0500f] text-white border-[#f0500f] shadow-sm'
                : 'bg-white/60 text-[#f0500f] border-[#f0500f]/30 hover:bg-[#f0500f]/10'
            }`}
            title="Focus camera directly on Front Facade wall"
          >
            <Sparkles className="w-3 h-3 text-[#f0500f]" />
            <span>Facade</span>
          </button>

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
