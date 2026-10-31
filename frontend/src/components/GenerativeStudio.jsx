import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { 
  Sparkles, 
  Box, 
  Layers, 
  Download, 
  Check, 
  RefreshCw, 
  Upload, 
  Palette, 
  Eye, 
  RotateCcw, 
  Maximize2, 
  Sliders, 
  Building2, 
  BookmarkCheck, 
  Trash2, 
  ExternalLink,
  Info
} from 'lucide-react';

// Preset sample furniture images (Data URI high-quality synthetic silhouettes)
const FURNITURE_PRESETS = [
  {
    id: 'preset_chair_1',
    name: 'Modern Accent Armchair',
    category: 'chair',
    desc: 'Sculptural curved lounge chair with warm terracotta fabric',
    height_m: 0.85,
    preview_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23f7f4ed"/><rect x="55" y="35" width="90" height="85" rx="15" fill="%23c85a2b"/><rect x="40" y="105" width="120" height="28" rx="8" fill="%23b8452a"/><rect x="52" y="133" width="12" height="48" rx="4" fill="%232b2b2b"/><rect x="136" y="133" width="12" height="48" rx="4" fill="%232b2b2b"/></svg>'
  },
  {
    id: 'preset_table_1',
    name: 'Scandinavian Dining Table',
    category: 'table',
    desc: 'Minimalist solid oak dining table with rounded beveled edges',
    height_m: 0.75,
    preview_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23f7f4ed"/><rect x="25" y="60" width="150" height="24" rx="6" fill="%23b38054"/><rect x="38" y="84" width="14" height="88" rx="3" fill="%238a5b32"/><rect x="148" y="84" width="14" height="88" rx="3" fill="%238a5b32"/></svg>'
  },
  {
    id: 'preset_sofa_1',
    name: 'Tufted Velvet Sofa',
    category: 'sofa',
    desc: 'Deep olive green 3-seater contemporary living room sofa',
    height_m: 0.80,
    preview_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23f7f4ed"/><rect x="20" y="55" width="160" height="60" rx="12" fill="%233b5a45"/><rect x="20" y="100" width="160" height="35" rx="6" fill="%232e4836"/><rect x="30" y="135" width="12" height="30" rx="3" fill="%231a1a1a"/><rect x="158" y="135" width="12" height="30" rx="3" fill="%231a1a1a"/></svg>'
  },
  {
    id: 'preset_bed_1',
    name: 'Minimalist Platform Bed',
    category: 'bed',
    desc: 'Low-profile upholstered king platform bed with integrated headboard',
    height_m: 0.65,
    preview_url: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23f7f4ed"/><rect x="25" y="45" width="150" height="55" rx="8" fill="%23485563"/><rect x="30" y="95" width="140" height="45" rx="6" fill="%23e8e4db"/><rect x="35" y="140" width="130" height="18" rx="4" fill="%232d3436"/></svg>'
  }
];

const FACADE_STYLES = [
  { id: 'Modern Minimalist', label: 'Modern Minimalist', desc: 'White stucco, floor-to-ceiling glass, black aluminum' },
  { id: 'Mediterranean Stone', label: 'Mediterranean Stone', desc: 'Warm sandstone masonry, terracotta tiles, rustic arches' },
  { id: 'Traditional Brick', label: 'Traditional Brick', desc: 'Exposed red clay brickwork, framed structural lintels' },
  { id: 'Lush Vegetation', label: 'Lush Vegetation', desc: 'Biophilic vertical greening, climbing ivy, living facade' },
];

const CMP_LEGEND = [
  { label: 'Wall', color: '#0030ff', desc: 'Primary facade surface' },
  { label: 'Window', color: '#0080ff', desc: 'Glazed openings' },
  { label: 'Door', color: '#00cfff', desc: 'Entrance / egress' },
  { label: 'Window Sill', color: '#ff5000', desc: 'Bottom stone sill' },
  { label: 'Window Lintel', color: '#ffff00', desc: 'Top structural lintel' },
  { label: 'Cornice / Trim', color: '#0000de', desc: 'Roofline & base molding' },
];

export default function GenerativeStudio({
  layout,
  frontWallId,
  onApplyFacadeTexture,
  onNavigateToFloorplan
}) {
  const [studioTab, setStudioTab] = useState('furniture'); // 'furniture' | 'facade'

  // --- 2D-to-3D Furniture State ---
  const [furnitureImageBase64, setFurnitureImageBase64] = useState(FURNITURE_PRESETS[0].preview_url);
  const [furnitureName, setFurnitureName] = useState(FURNITURE_PRESETS[0].name);
  const [furnitureCategory, setFurnitureCategory] = useState(FURNITURE_PRESETS[0].category);
  const [removeBg, setRemoveBg] = useState(true);
  const [targetHeightM, setTargetHeightM] = useState(FURNITURE_PRESETS[0].height_m);
  const [isGenerating3D, setIsGenerating3D] = useState(false);
  const [generatedModel, setGeneratedModel] = useState(null);
  const [customLibrary, setCustomLibrary] = useState([]);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState(null);
  const [wireframe3D, setWireframe3D] = useState(false);
  const [autoRotate3D, setAutoRotate3D] = useState(true);

  // --- Facade Synthesis State ---
  const [selectedStyle, setSelectedStyle] = useState('Modern Minimalist');
  const [numFloors, setNumFloors] = useState(1);
  const [isGeneratingFacade, setIsGeneratingFacade] = useState(false);
  const [facadeResult, setFacadeResult] = useState(null);
  const [appliedNotice, setAppliedNotice] = useState(false);

  const fileInputRef = useRef(null);
  const canvas3DRef = useRef(null);
  const scene3DRef = useRef(null);
  const renderer3DRef = useRef(null);
  const modelGroupRef = useRef(null);
  const controls3DRef = useRef(null);
  const autoRotateRef = useRef(autoRotate3D);

  useEffect(() => {
    autoRotateRef.current = autoRotate3D;
  }, [autoRotate3D]);

  // Load saved custom library on mount
  useEffect(() => {
    axios.get('/furniture/custom-library')
      .then(res => {
        if (res.data) {
          setCustomLibrary(Object.values(res.data));
        }
      })
      .catch(err => console.log('Custom library empty or uninitialized:', err));
  }, []);

  // --- Initialize Dedicated 3D Furniture Viewport ---
  useEffect(() => {
    if (studioTab !== 'furniture') return;
    const container = canvas3DRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf1ede4);
    scene3DRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, (width && height) ? (width / height) : 1.33, 0.05, 50);
    camera.position.set(1.4, 1.2, 1.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    // Safely remove any existing canvas from previous renders
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.appendChild(renderer.domElement);
    renderer3DRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.02;
    controls.minDistance = 0.4;
    controls.maxDistance = 5.0;
    controls3DRef.current = controls;

    // Lighting setup
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0xd0c8b8, 0.9);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(3, 5, 2.5);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.bias = -0.0001;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xbad2e8, 0.4);
    fillLight.position.set(-3, 2, -2);
    scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.65);
    backLight.position.set(0, 3, -4);
    scene.add(backLight);

    // Grid Floor
    const grid = new THREE.GridHelper(3.0, 30, 0x1d1d1f, 0x1d1d1f);
    grid.material.opacity = 0.12;
    grid.material.transparent = true;
    grid.position.y = 0.001;
    scene.add(grid);

    // Shadow Catcher Plane
    const shadowGeo = new THREE.PlaneGeometry(6, 6);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.15 });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.receiveShadow = true;
    scene.add(shadowPlane);

    const modelGroup = new THREE.Group();
    scene.add(modelGroup);
    modelGroupRef.current = modelGroup;

    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (controls3DRef.current) {
        if (autoRotateRef.current && modelGroup.children.length > 0) {
          modelGroup.rotation.y += 0.006;
        }
        controls.update();
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !renderer) return;
      const w = container.clientWidth || 600;
      const h = container.clientHeight || 450;
      if (h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement && renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [studioTab]);

  // Load GLB model into 3D viewer when generatedModel updates
  useEffect(() => {
    const modelGroup = modelGroupRef.current;
    if (!modelGroup || !generatedModel?.model_glb_base64) return;

    modelGroup.clear();

    const loader = new GLTFLoader();
    loader.load(
      generatedModel.model_glb_base64,
      (gltf) => {
        const scene = gltf.scene;
        scene.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.geometry) {
              child.geometry.computeVertexNormals();
            }
            if (child.material) {
              child.material.side = THREE.DoubleSide;
              child.material.color.set(0xffffff);
              const hasVertexColors = Boolean(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
              child.material.vertexColors = hasVertexColors;
              child.material.roughness = generatedModel.roughness !== undefined ? generatedModel.roughness : 0.35;
              child.material.metalness = 0.05;
              child.material.flatShading = false;
              if (wireframe3D) {
                child.material.wireframe = true;
              }
              if (child.material.map) {
                if (THREE.SRGBColorSpace) {
                  child.material.map.colorSpace = THREE.SRGBColorSpace;
                }
                child.material.map.needsUpdate = true;
              }
              child.material.needsUpdate = true;
            }
          }
        });

        // Center on floor (y=0)
        const bbox = new THREE.Box3().setFromObject(scene);
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        scene.position.x = -center.x;
        scene.position.y = -bbox.min.y;
        scene.position.z = -center.z;

        modelGroup.add(scene);

        // Frame camera comfortably
        const maxDim = Math.max(size.x, size.y, size.z) || 1.0;
        if (controls3DRef.current) {
          controls3DRef.current.target.set(0, size.y * 0.5, 0);
          controls3DRef.current.update();
        }
      },
      undefined,
      (err) => console.error('GLB preview load error:', err)
    );
  }, [generatedModel, wireframe3D]);

  // Handle Preset Selection
  const handleSelectPreset = (preset) => {
    setFurnitureImageBase64(preset.preview_url);
    setFurnitureName(preset.name);
    setFurnitureCategory(preset.category);
    setTargetHeightM(preset.height_m);
  };

  // Handle Image Upload with client-side downscaling (max 512px) for instant, responsive processing
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (uploadEvt) => {
      const rawDataUrl = uploadEvt.target.result;
      const cleanName = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
      setFurnitureName(cleanName.charAt(0).toUpperCase() + cleanName.slice(1));

      // Downscale to max 512x512 to prevent browser freezing and GPU memory bloat
      const img = new Image();
      img.onload = () => {
        try {
          const maxDim = 512;
          let w = img.width;
          let h = img.height;
          if (w > maxDim || h > maxDim) {
            if (w > h) {
              h = Math.round((h * maxDim) / w);
              w = maxDim;
            } else {
              w = Math.round((w * maxDim) / h);
              h = maxDim;
            }
          }
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          const optimizedDataUrl = canvas.toDataURL('image/png');
          setFurnitureImageBase64(optimizedDataUrl);
        } catch (resizeErr) {
          console.warn('Canvas resize fallback:', resizeErr);
          setFurnitureImageBase64(rawDataUrl);
        }
      };
      img.onerror = () => {
        setFurnitureImageBase64(rawDataUrl);
      };
      img.src = rawDataUrl;
    };
    reader.readAsDataURL(file);
  };

  // Trigger 3D Furniture Generation
  const handleGenerate3DFurniture = async () => {
    if (!furnitureImageBase64) return;
    setIsGenerating3D(true);
    setSaveSuccessMsg(null);
    try {
      const payload = {
        image_base64: furnitureImageBase64,
        name: furnitureName,
        category: furnitureCategory,
        remove_bg: removeBg,
        target_height_m: targetHeightM
      };

      const res = await axios.post('/furniture/generate-3d', payload);
      if (res.data?.status === 'success') {
        setGeneratedModel(res.data);
      }
    } catch (err) {
      console.error('Furniture generation error:', err);
      alert(err.response?.data?.detail || '3D generation failed. Please try another image.');
    } finally {
      setIsGenerating3D(false);
    }
  };

  // Save 3D Model to Room Design Library
  const handleSaveToLibrary = async () => {
    if (!generatedModel) return;
    try {
      const payload = {
        id: generatedModel.id,
        name: generatedModel.name,
        category: generatedModel.category,
        dimensions_m: generatedModel.dimensions_m,
        bounds: generatedModel.bounds,
        glb_base64: generatedModel.model_glb_base64
      };

      const res = await axios.post('/furniture/save-to-library', payload);
      if (res.data?.status === 'success') {
        setSaveSuccessMsg(`Saved "${generatedModel.name}" to Room Design Library!`);
        setCustomLibrary(prev => [...prev.filter(item => item.id !== generatedModel.id), res.data.item]);
      }
    } catch (err) {
      console.error('Save error:', err);
      alert('Failed to save model to catalog.');
    }
  };

  // Download GLB
  const handleDownloadGLB = () => {
    if (!generatedModel?.model_glb_base64) return;
    const a = document.createElement('a');
    a.href = generatedModel.model_glb_base64;
    a.download = `${generatedModel.id}.glb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Trigger Facade Generation
  const handleGenerateFacade = async (overrideStyle = null) => {
    const targetStyle = overrideStyle || selectedStyle;
    setIsGeneratingFacade(true);
    setAppliedNotice(false);
    try {
      const payload = {
        layout: layout,
        wall_id: frontWallId,
        style: targetStyle,
        num_floors: numFloors
      };

      const res = await axios.post('/facade/generate', payload);
      if (res.data?.status === 'success') {
        setFacadeResult(res.data);
      }
    } catch (err) {
      console.error('Facade generation error:', err);
      alert(err.response?.data?.detail || 'Facade generation failed.');
    } finally {
      setIsGeneratingFacade(false);
    }
  };

  // Apply Facade to Active 3D Building Floorplan
  const handleApplyFacadeTo3D = () => {
    if (!facadeResult?.texture_base64) return;
    if (onApplyFacadeTexture) {
      onApplyFacadeTexture(facadeResult.texture_base64, facadeResult);
      setAppliedNotice(true);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#efebe2] overflow-hidden select-none font-mono text-[#1d1d1f]">
      
      {/* Studio Header & Tab Navigation */}
      <div className="h-14 border-b border-[#1d1d1f]/15 bg-[#e5e0d4]/90 backdrop-blur-md px-6 flex items-center justify-between z-10 shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-[#f0500f] text-white flex items-center justify-center shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-stencil text-sm uppercase tracking-wider text-[#1d1d1f]">
                GENERATIVE AI STUDIO
              </span>
              <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-[#f0500f]/15 text-[#f0500f] border border-[#f0500f]/30">
                RTX 3050 CUDA
              </span>
            </div>
            <p className="text-[10px] text-[#1d1d1f]/60 tracking-wider">
              2D Image-to-3D Furniture Synthesizer & Conditional GAN Facade Engine
            </p>
          </div>
        </div>

        {/* Sub-Studio Switcher */}
        <div className="flex items-center p-1 rounded-full border border-[#1d1d1f]/20 bg-white/60">
          <button
            onClick={() => setStudioTab('furniture')}
            className={`flex items-center space-x-1.5 h-8 px-4 rounded-full font-stencil text-xs uppercase tracking-wider transition-all ${
              studioTab === 'furniture'
                ? 'bg-[#1d1d1f] text-[#efebe2] shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <Box className="w-3.5 h-3.5 text-[#f0500f]" />
            <span>2D-to-3D Furniture</span>
          </button>

          <button
            onClick={() => {
              setStudioTab('facade');
              if (!facadeResult) {
                handleGenerateFacade();
              }
            }}
            className={`flex items-center space-x-1.5 h-8 px-4 rounded-full font-stencil text-xs uppercase tracking-wider transition-all ${
              studioTab === 'facade'
                ? 'bg-[#1d1d1f] text-[#efebe2] shadow-sm'
                : 'text-[#1d1d1f]/70 hover:text-[#1d1d1f] hover:bg-black/5'
            }`}
          >
            <Building2 className="w-3.5 h-3.5 text-[#f0500f]" />
            <span>Neural Facade Studio</span>
          </button>
        </div>

        {/* Action: Return to Floorplan */}
        {onNavigateToFloorplan && (
          <button
            onClick={onNavigateToFloorplan}
            className="flex items-center space-x-1.5 h-8 px-3.5 rounded-full border border-[#1d1d1f]/30 bg-white hover:bg-[#1d1d1f] hover:text-[#efebe2] transition-all font-stencil text-xs uppercase tracking-wider shadow-xs"
          >
            <Eye className="w-3.5 h-3.5 text-[#f0500f]" />
            <span>VIEW 3D FLOORPLAN</span>
          </button>
        )}
      </div>

      {/* =========================================================================
          TAB 1: 2D-TO-3D FURNITURE SYNTHESIZER
         ========================================================================= */}
      {studioTab === 'furniture' && (
        <div className="flex-1 flex overflow-hidden">
          
          {/* Left Controls & Image Upload */}
          <aside className="w-96 border-r border-[#1d1d1f]/15 bg-[#efebe2] p-5 flex flex-col space-y-4 overflow-y-auto shrink-0">
            
            <div>
              <label className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] block mb-2">
                01 // FURNITURE IMAGE INPUT
              </label>

              {/* Upload Dropzone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="w-full h-44 rounded-xl border-2 border-dashed border-[#1d1d1f]/30 hover:border-[#f0500f] bg-white/70 hover:bg-white transition-all cursor-pointer flex flex-col items-center justify-center p-3 relative overflow-hidden group shadow-xs"
              >
                {furnitureImageBase64 ? (
                  <div className="relative w-full h-full flex items-center justify-center">
                    <img
                      src={generatedModel?.preview_image_base64 || furnitureImageBase64}
                      alt="Input Furniture"
                      className="max-h-full max-w-full object-contain rounded drop-shadow-md"
                    />
                    {generatedModel?.preview_image_base64 && (
                      <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-emerald-700/90 text-white text-[9px] font-mono font-bold tracking-wider">
                        AI ISOLATED (BG REMOVED)
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center space-y-1.5 text-center">
                    <Upload className="w-6 h-6 text-[#f0500f]" />
                    <span className="font-stencil text-xs uppercase text-[#1d1d1f]">DROP 2D FURNITURE PHOTO</span>
                    <span className="text-[10px] text-[#1d1d1f]/50">PNG, JPG, WebP</span>
                  </div>
                )}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white text-xs font-stencil">
                  CLICK TO REPLACE IMAGE
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>

            {/* Quick Test Preset Templates */}
            <div>
              <span className="text-[10px] font-bold text-[#1d1d1f]/60 uppercase tracking-wider block mb-1.5">
                Or Pick a Curated Preset Template:
              </span>
              <div className="grid grid-cols-2 gap-2">
                {FURNITURE_PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleSelectPreset(p)}
                    className={`p-2 rounded-lg border text-left flex items-center space-x-2 transition-all ${
                      furnitureName === p.name
                        ? 'bg-white border-[#f0500f] shadow-xs ring-1 ring-[#f0500f]'
                        : 'bg-white/40 border-[#1d1d1f]/15 hover:bg-white hover:border-[#1d1d1f]/40'
                    }`}
                  >
                    <img src={p.preview_url} alt={p.name} className="w-8 h-8 rounded shrink-0 object-contain" />
                    <div className="overflow-hidden">
                      <div className="text-[10px] font-bold text-[#1d1d1f] truncate">{p.name}</div>
                      <div className="text-[9px] text-[#1d1d1f]/50 capitalize">{p.category}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Metadata Parameters */}
            <div className="space-y-3 bg-[#e5e0d4]/50 p-3.5 rounded-xl border border-[#1d1d1f]/15 text-xs">
              <div>
                <label className="text-[10px] text-[#1d1d1f]/70 uppercase block mb-1">Furniture Name</label>
                <input
                  type="text"
                  value={furnitureName}
                  onChange={(e) => setFurnitureName(e.target.value)}
                  className="w-full h-8 px-2.5 rounded bg-white border border-[#1d1d1f]/20 text-[#1d1d1f] font-mono text-xs focus:outline-none focus:border-[#f0500f]"
                />
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="text-[10px] text-[#1d1d1f]/70 uppercase block mb-1">Category</label>
                  <select
                    value={furnitureCategory}
                    onChange={(e) => setFurnitureCategory(e.target.value)}
                    className="w-full h-8 px-2 rounded bg-white border border-[#1d1d1f]/20 text-[#1d1d1f] font-mono text-xs"
                  >
                    <option value="chair">Chair / Armchair</option>
                    <option value="table">Table / Desk</option>
                    <option value="sofa">Sofa / Couch</option>
                    <option value="bed">Bed</option>
                    <option value="storage">Cabinet / Wardrobe</option>
                    <option value="decor">Decor / Plant</option>
                    <option value="lighting">Floor Lamp</option>
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-[#1d1d1f]/70 uppercase block mb-1">Height (m)</label>
                  <input
                    type="number"
                    step="0.05"
                    value={targetHeightM}
                    onChange={(e) => setTargetHeightM(parseFloat(e.target.value) || 0.85)}
                    className="w-full h-8 px-2 rounded bg-white border border-[#1d1d1f]/20 text-[#1d1d1f] font-mono text-xs"
                  />
                </div>
              </div>

              <label className="flex items-center space-x-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={removeBg}
                  onChange={(e) => setRemoveBg(e.target.checked)}
                  className="rounded text-[#f0500f] focus:ring-[#f0500f]"
                />
                <span className="text-[11px] text-[#1d1d1f]/80">AI Background Removal (u2netp)</span>
              </label>
            </div>

            {/* Synthesize Button */}
            <button
              onClick={handleGenerate3DFurniture}
              disabled={isGenerating3D}
              className="w-full h-11 px-5 rounded-full bg-[#f0500f] hover:bg-[#d8450a] text-white font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#f0500f]/20 disabled:opacity-50 mt-auto"
            >
              {isGenerating3D ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>SYNTHESIZING 3D MODEL...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>SYNTHESIZE 3D MODEL →</span>
                </>
              )}
            </button>
          </aside>

          {/* Center 3D Viewport & Model Inspector */}
          <main className="flex-1 flex flex-col h-full bg-[#f1ede4] relative">
            
            {/* Viewport Top Toolbar */}
            <div className="h-10 border-b border-[#1d1d1f]/10 bg-[#f7f4ed]/80 backdrop-blur-sm px-4 flex items-center justify-between z-10 text-xs">
              <div className="flex items-center space-x-2">
                <Box className="w-3.5 h-3.5 text-[#f0500f]" />
                <span className="font-semibold text-xs uppercase tracking-wider text-[#1d1d1f]">
                  3D MODEL ORBIT VIEWPORT
                </span>
                {generatedModel && (
                  <span className="px-2 py-0.5 rounded text-[10px] bg-[#f0500f]/10 text-[#f0500f] font-bold">
                    {generatedModel.name}
                  </span>
                )}
              </div>

              {/* Viewport Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setAutoRotate3D(!autoRotate3D)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono border transition-all ${
                    autoRotate3D
                      ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                      : 'bg-white text-[#1d1d1f]/70 border-[#1d1d1f]/20'
                  }`}
                >
                  Rotate: {autoRotate3D ? 'ON' : 'OFF'}
                </button>

                <button
                  onClick={() => setWireframe3D(!wireframe3D)}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono border transition-all ${
                    wireframe3D
                      ? 'bg-[#1d1d1f] text-[#efebe2] border-[#1d1d1f]'
                      : 'bg-white text-[#1d1d1f]/70 border-[#1d1d1f]/20'
                  }`}
                >
                  Wireframe: {wireframe3D ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>

            {/* Three.js Canvas Container & Viewport Overlays */}
            <div className="flex-1 w-full h-full relative overflow-hidden">
              {/* Dedicated Three.js Canvas Mount - completely empty so React never touches its DOM children */}
              <div ref={canvas3DRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

              {/* Sibling React Overlay: Empty State */}
              {!generatedModel && !isGenerating3D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center pointer-events-none select-none">
                  <div className="w-16 h-16 rounded-2xl bg-white/70 border border-[#1d1d1f]/15 flex items-center justify-center mb-3 shadow-sm">
                    <Box className="w-8 h-8 text-[#f0500f]" />
                  </div>
                  <h3 className="font-stencil text-base uppercase text-[#1d1d1f] tracking-wider mb-1">
                    READY FOR 3D SYNTHESIS
                  </h3>
                  <p className="font-mono text-xs text-[#1d1d1f]/60 max-w-sm">
                    Choose a preset furniture piece on the left or upload your own 2D photo, then click "SYNTHESIZE 3D MODEL".
                  </p>
                </div>
              )}

              {/* Sibling React Overlay: Generating Loader */}
              {isGenerating3D && (
                <div className="absolute inset-0 bg-[#efebe2]/85 backdrop-blur-xs flex flex-col items-center justify-center z-20 pointer-events-none select-none">
                  <div className="w-16 h-16 rounded-2xl bg-white/90 border border-[#1d1d1f]/20 flex items-center justify-center mb-4 shadow-lg animate-pulse">
                    <RefreshCw className="w-8 h-8 text-[#f0500f] animate-spin" />
                  </div>
                  <h3 className="font-stencil text-base uppercase text-[#1d1d1f] tracking-wider mb-2">
                    SYNTHESIZING 3D GEOMETRY...
                  </h3>
                  <p className="font-mono text-xs text-[#1d1d1f]/60 max-w-sm text-center">
                    Extracting alpha mask with u2netp, building volumetric boundary hull, and generating textured GLB mesh.
                  </p>
                </div>
              )}
            </div>

            {/* Model Action & Telemetry Floating Bar */}
            {generatedModel && (
              <div className="p-4 bg-white/95 border-t border-[#1d1d1f]/15 flex flex-wrap items-center justify-between gap-3 shadow-lg">
                <div className="flex items-center space-x-4 text-xs font-mono">
                  <div>
                    <span className="text-[#1d1d1f]/50 block text-[9px] uppercase">Dimensions (W×D×H)</span>
                    <span className="font-bold text-[#1d1d1f]">
                      {generatedModel.dimensions_m?.[0]}m × {generatedModel.dimensions_m?.[1]}m × {generatedModel.dimensions_m?.[2]}m
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#1d1d1f]/10" />
                  <div>
                    <span className="text-[#1d1d1f]/50 block text-[9px] uppercase">Mesh Density</span>
                    <span className="font-bold text-[#1d1d1f]">
                      {generatedModel.num_vertices?.toLocaleString()} verts / {generatedModel.num_faces?.toLocaleString()} faces
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#1d1d1f]/10" />
                  <div>
                    <span className="text-[#1d1d1f]/50 block text-[9px] uppercase">Surface Texture</span>
                    <span className="font-bold text-[#1d1d1f]">
                      {generatedModel.roughness !== undefined 
                        ? (generatedModel.roughness < 0.40 ? `Smooth Polished (${generatedModel.roughness})` : `Matte Textured (${generatedModel.roughness})`)
                        : 'Auto PBR'}
                    </span>
                  </div>
                  <div className="h-6 w-px bg-[#1d1d1f]/10" />
                  <div>
                    <span className="text-[#1d1d1f]/50 block text-[9px] uppercase">Synthesis Engine</span>
                    <span className="font-bold text-[#f0500f]">{generatedModel.engine}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2.5">
                  {saveSuccessMsg && (
                    <span className="text-xs text-emerald-700 font-bold flex items-center space-x-1 animate-in fade-in">
                      <BookmarkCheck className="w-4 h-4 text-emerald-600" />
                      <span>{saveSuccessMsg}</span>
                    </span>
                  )}

                  <button
                    onClick={handleSaveToLibrary}
                    className="h-9 px-4 rounded-full bg-[#1d1d1f] hover:bg-black text-[#efebe2] font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-sm"
                  >
                    <BookmarkCheck className="w-3.5 h-3.5 text-[#f0500f]" />
                    <span>SAVE TO ROOM DESIGN LIBRARY</span>
                  </button>

                  <button
                    onClick={handleDownloadGLB}
                    className="h-9 px-4 rounded-full bg-white hover:bg-[#efebe2] border border-[#1d1d1f]/30 text-[#1d1d1f] font-stencil text-xs uppercase tracking-wider flex items-center space-x-1.5 transition-all shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5 text-[#f0500f]" />
                    <span>DOWNLOAD .GLB</span>
                  </button>
                </div>
              </div>
            )}

            {/* Custom Saved Library Shelf */}
            {customLibrary.length > 0 && (
              <div className="px-5 py-3 bg-[#e5e0d4]/40 border-t border-[#1d1d1f]/10 flex items-center space-x-3 overflow-x-auto">
                <span className="text-[10px] font-stencil uppercase tracking-wider text-[#1d1d1f]/60 shrink-0">
                  SAVED CATALOG ({customLibrary.length}):
                </span>
                <div className="flex items-center space-x-2">
                  {customLibrary.map(item => (
                    <div
                      key={item.id}
                      className="px-2.5 py-1 rounded bg-white border border-[#1d1d1f]/20 flex items-center space-x-2 shrink-0 shadow-2xs"
                    >
                      <span className="w-2 h-2 rounded-full bg-[#f0500f]" />
                      <span className="text-xs font-mono font-medium text-[#1d1d1f]">{item.name}</span>
                      <span className="text-[9px] text-[#1d1d1f]/50 capitalize">({item.category})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </main>
        </div>
      )}

      {/* =========================================================================
          TAB 2: NEURAL FACADE SYNTHESIS STUDIO
         ========================================================================= */}
      {studioTab === 'facade' && (
        <div className="flex-1 flex overflow-hidden">
          
          {/* Left: Facade Style & Wall Selection Controls */}
          <aside className="w-96 border-r border-[#1d1d1f]/15 bg-[#efebe2] p-5 flex flex-col space-y-5 overflow-y-auto shrink-0">
            <div>
              <label className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] flex items-center space-x-1.5 mb-1.5">
                <Building2 className="w-3.5 h-3.5 text-[#f0500f]" />
                <span>01 // TARGET WALL SPECS</span>
              </label>
              <div className="p-3 bg-white rounded-xl border border-[#1d1d1f]/15 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-[#1d1d1f]/60">Target Wall:</span>
                  <span className="font-bold">{frontWallId || 'Exterior Front Wall'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#1d1d1f]/60">Floor Stories:</span>
                  <div className="flex items-center space-x-1">
                    {[1, 2, 3].map(fl => (
                      <button
                        key={fl}
                        onClick={() => setNumFloors(fl)}
                        className={`w-6 h-6 rounded text-xs font-mono font-bold transition-all ${
                          numFloors === fl ? 'bg-[#f0500f] text-white' : 'bg-[#efebe2] text-[#1d1d1f]/70'
                        }`}
                      >
                        {fl}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Architectural Styles Selector */}
            <div className="space-y-2">
              <label className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] flex items-center space-x-1.5">
                <Palette className="w-3.5 h-3.5 text-[#f0500f]" />
                <span>02 // ARCHITECTURAL STYLE</span>
              </label>

              <div className="space-y-2">
                {FACADE_STYLES.map(st => (
                  <div
                    key={st.id}
                    onClick={() => {
                      setSelectedStyle(st.id);
                      handleGenerateFacade(st.id);
                    }}
                    className={`p-3 rounded-xl border cursor-pointer transition-all ${
                      selectedStyle === st.id
                        ? 'bg-white border-[#f0500f] shadow-sm ring-1 ring-[#f0500f]'
                        : 'bg-white/50 border-[#1d1d1f]/15 hover:bg-white hover:border-[#1d1d1f]/40'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-stencil text-xs tracking-wider uppercase text-[#1d1d1f]">{st.label}</span>
                      {selectedStyle === st.id && <Check className="w-3.5 h-3.5 text-[#f0500f]" />}
                    </div>
                    <p className="font-mono text-[9.5px] text-[#1d1d1f]/60 mt-0.5 leading-tight">{st.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={() => handleGenerateFacade(selectedStyle)}
              disabled={isGeneratingFacade}
              className="w-full h-11 px-5 rounded-full bg-[#f0500f] hover:bg-[#d8450a] text-white font-stencil text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-md shadow-[#f0500f]/20 disabled:opacity-50 mt-auto"
            >
              {isGeneratingFacade ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>SYNTHESIZING cGAN TEXTURE...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>RE-GENERATE FACADE TEXTURE →</span>
                </>
              )}
            </button>
          </aside>

          {/* Center: Side-by-Side Semantic Sketch vs Realistic Output */}
          <main className="flex-1 flex flex-col h-full bg-[#f6f3eb] p-6 overflow-y-auto">
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-[460px]">
              
              {/* LEFT: Input Semantic Layout Scrap */}
              <div className="bg-white rounded-2xl border border-[#1d1d1f]/20 p-5 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#0030ff]" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f]">
                      01 // INPUT SEMANTIC SKETCH ("THE SCRAP")
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#1d1d1f]/50">CMP 12-Class Palette</span>
                </div>

                <p className="text-[11px] text-[#1d1d1f]/70 mb-3 font-mono">
                  Procedural architectural scrap automatically projected from the exterior wall, doors, and window geometry.
                </p>

                <div className="flex-1 w-full bg-[#0030ff]/10 rounded-xl overflow-hidden border border-[#1d1d1f]/10 flex items-center justify-center p-3 min-h-[280px]">
                  {facadeResult?.sketch_base64 ? (
                    <img
                      src={facadeResult.sketch_base64}
                      alt="Semantic Sketch"
                      className="max-h-full max-w-full object-contain rounded shadow-inner"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  ) : (
                    <div className="text-xs text-[#1d1d1f]/50 font-mono text-center">
                      Click "RE-GENERATE FACADE TEXTURE" to synthesize
                    </div>
                  )}
                </div>

                {/* CMP Color Palette Legend */}
                <div className="mt-4 pt-3 border-t border-[#1d1d1f]/10">
                  <span className="font-mono text-[10px] font-bold text-[#1d1d1f]/60 uppercase tracking-wider block mb-1.5">
                    CMP Semantic Colormap Legend:
                  </span>
                  <div className="grid grid-cols-3 gap-2 text-[10.5px] font-mono">
                    {CMP_LEGEND.map(item => (
                      <div key={item.label} className="flex items-center space-x-1.5 bg-[#efebe2]/60 px-2 py-1 rounded">
                        <span
                          className="w-3 h-3 rounded-sm border border-black/20 shrink-0"
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="truncate text-[#1d1d1f]/80">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* RIGHT: Synthesized Realistic Facade */}
              <div className="bg-white rounded-2xl border border-[#1d1d1f]/20 p-5 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#f0500f] animate-pulse" />
                    <span className="font-stencil text-xs uppercase tracking-wider text-[#1d1d1f]">
                      02 // SYNTHESIZED REALISTIC FACADE (cGAN)
                    </span>
                  </div>
                  <span className="px-2.5 py-0.5 rounded text-[10px] font-bold bg-[#f0500f]/10 text-[#f0500f]">
                    {selectedStyle}
                  </span>
                </div>

                <p className="text-[11px] text-[#1d1d1f]/70 mb-3 font-mono">
                  Photorealistic building materials synthesized by Pix2Pix conditional GAN with architectural enhancements.
                </p>

                <div className="flex-1 w-full bg-[#1d1d1f]/5 rounded-xl overflow-hidden border border-[#1d1d1f]/10 flex items-center justify-center p-3 min-h-[280px]">
                  {isGeneratingFacade ? (
                    <div className="flex flex-col items-center justify-center space-y-2 text-[#f0500f]">
                      <RefreshCw className="w-8 h-8 animate-spin" />
                      <span className="font-mono text-xs">Neural Inference in Progress...</span>
                    </div>
                  ) : facadeResult?.texture_base64 ? (
                    <img
                      src={facadeResult.texture_base64}
                      alt="Realistic Facade"
                      className="max-h-full max-w-full object-contain rounded shadow-lg"
                    />
                  ) : (
                    <div className="text-xs text-[#1d1d1f]/50 font-mono text-center">
                      No texture generated yet
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-3 border-t border-[#1d1d1f]/10 flex items-center justify-between text-xs font-mono text-[#1d1d1f]/60">
                  <span>Aspect ratio preserved · {facadeResult?.model_info?.resolution || 256}px</span>
                  <span className="text-[#f0500f] font-semibold">{facadeResult?.device || 'ONNX CPU / PyTorch CUDA'}</span>
                </div>
                {facadeResult?.model_info?.model === 'facade_residual_film_512_v1' && !facadeResult.model_info.style_conditioned && (
                  <p className="mt-2 text-xs text-amber-800">Experimental CMP model: neutral appearance. The selected architectural style has not been trained.</p>
                )}
              </div>

            </div>

            {/* Facade Action Bar */}
            <div className="mt-6 p-4 bg-white rounded-2xl border border-[#1d1d1f]/15 flex flex-wrap items-center justify-between gap-3 shadow-md">
              <div className="flex items-center space-x-2 text-xs font-mono text-[#1d1d1f]/70">
                <Info className="w-4 h-4 text-[#f0500f]" />
                <span>Apply this synthesized texture directly to the exterior walls of your 3D building floorplan.</span>
              </div>

              <div className="flex items-center space-x-3">
                {appliedNotice && (
                  <span className="text-xs font-bold text-emerald-700 flex items-center space-x-1 animate-in fade-in">
                    <Check className="w-4 h-4 text-emerald-600" />
                    <span>Applied to 3D Floorplan!</span>
                  </span>
                )}

                <button
                  onClick={handleApplyFacadeTo3D}
                  disabled={!facadeResult?.texture_base64}
                  className="h-10 px-5 rounded-full bg-[#1d1d1f] hover:bg-black text-[#efebe2] font-stencil text-xs uppercase tracking-wider flex items-center space-x-2 transition-all shadow-sm disabled:opacity-40"
                >
                  <Eye className="w-3.5 h-3.5 text-[#f0500f]" />
                  <span>APPLY TO 3D BUILDING</span>
                </button>

                {facadeResult?.texture_base64 && (
                  <a
                    href={facadeResult.texture_base64}
                    download={`facade_${selectedStyle.toLowerCase().replace(/\s+/g, '_')}.png`}
                    className="h-10 px-4 rounded-full bg-white hover:bg-[#efebe2] border border-[#1d1d1f]/30 font-stencil text-xs uppercase tracking-wider text-[#1d1d1f] flex items-center space-x-1.5 transition-all shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5 text-[#f0500f]" />
                    <span>DOWNLOAD PNG</span>
                  </a>
                )}
              </div>
            </div>

          </main>
        </div>
      )}

    </div>
  );
}
