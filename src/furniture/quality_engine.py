"""Opt-in TripoSR reconstruction with bounded extraction and neural texture baking."""
import base64
import gc
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import trimesh

from .quality_geometry import prepare_input, extract_surface, to_room_coordinates, bake_neural_atlas


def gpu_usage():
    line = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    return tuple(int(v.strip()) for v in line.split(','))


class QualityFurnitureEngine:
    def __init__(self, use_gpu=True):
        import torch
        from omegaconf import OmegaConf
        self.torch = torch
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.device_name = f'TripoSR quality / {self.device}'
        self.lock = threading.Lock()
        self.chunk_size = 8192
        self.peak_device_mib = 0
        if self.device.type == 'cuda':
            total, used = gpu_usage()
            owned = torch.cuda.memory_reserved()/1024**2
            limit = min(total, 3500)-used+owned-384
            if limit-owned < 1500:
                raise RuntimeError('Insufficient GPU headroom for furniture quality mode; stop other GPU jobs')
            torch.cuda.set_per_process_memory_fraction(limit/total)
        root = Path(__file__).resolve().parents[2] / 'TripoSR'
        if not (root / 'model.ckpt').is_file():
            raise FileNotFoundError('TripoSR/model.ckpt is required; quality mode never substitutes an extrusion')
        sys.path.insert(0, str(root)) if str(root) not in sys.path else None
        from tsr.system import TSR
        cfg = OmegaConf.load(root / 'config.yaml')
        OmegaConf.resolve(cfg)
        self.model = TSR(cfg)
        state = torch.load(root / 'model.ckpt', map_location='cpu', weights_only=True)
        # Do not hide missing weights behind strict=False / tokenizer API changes.
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError('TripoSR checkpoint does not match this runtime. Use the isolated '
                               '.venv-furniture-quality environment and requirements-furniture-quality.txt') from exc
        del state
        gc.collect()
        self.model.eval()
        if self.device.type == 'cuda':
            # Keep the large image encoder/transformer weights in FP16 as well as
            # their activations. Density/color decoding stays FP32 for stability.
            self.model.half()
            self.model.decoder.float()
        self.model.to(self.device)
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.check_budget()

    def check_budget(self):
        if self.device.type == 'cuda':
            _, used = gpu_usage()
            self.peak_device_mib = max(used, self.peak_device_mib)
            if used > 3500:
                raise RuntimeError('Device GPU usage exceeded 3500 MiB; generation stopped')

    def query(self, scene, positions, key):
        torch = self.torch
        values = []
        # Density exp activation remains FP32 to avoid FP16 overflow.
        with torch.inference_mode():
            for start in range(0, len(positions), self.chunk_size):
                points = torch.as_tensor(positions[start:start+self.chunk_size], dtype=torch.float32, device=self.device)
                result = self.model.renderer.query_triplane(self.model.decoder, points, scene.float())[key]
                values.append(result.float().cpu().numpy())
        return np.concatenate(values)

    def density_volume(self, scene, resolution):
        radius = self.model.renderer.cfg.radius
        axis = np.linspace(-radius, radius, resolution, dtype=np.float32)
        volume = np.empty((resolution, resolution, resolution), np.float32)
        # Generate only two X slabs at once; the full grid never lives on the GPU.
        for start in range(0, resolution, 2):
            grid = np.stack(np.meshgrid(axis[start:start+2], axis, axis, indexing='ij'), -1)
            volume[start:start+2] = self.query(scene, grid.reshape(-1, 3), 'density_act').reshape(grid.shape[:-1])
            if start % 32 == 0:
                self.check_budget()
                print(f'Furniture density: {start+2}/{resolution} slabs', flush=True)
        return volume

    def generate_from_image(self, image, name='Custom Furniture', category='chair', remove_bg=True, target_height_m=None):
        from .image_to_3d import CATEGORY_DIMENSIONS
        torch = self.torch
        category = (category or 'chair').lower().strip()
        height = CATEGORY_DIMENSIONS.get(category, [.6, .6, .85])[2] if target_height_m is None else target_height_m
        if not np.isfinite(height) or not .05 <= height <= 10:
            raise ValueError('Target height must be between 0.05 and 10 metres')
        preset = os.environ.get('ARCHIGEN_FURNITURE_DETAIL', 'balanced')
        if preset not in ('balanced', 'high'):
            raise ValueError('ARCHIGEN_FURNITURE_DETAIL must be balanced or high')
        resolution, atlas_resolution = (384, 2048) if preset == 'high' else (256, 1024)
        with self.lock:
            started = time.monotonic()
            self.peak_device_mib = 0
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Floorplan/ONNX buffers may have grown since startup. Account for
                # non-PyTorch allocations again before admitting this generation.
                total, used = gpu_usage()
                owned = torch.cuda.memory_reserved()/1024**2
                limit = min(total, 3500)-used+owned-384
                if limit <= owned:
                    raise RuntimeError('Insufficient GPU headroom; close other GPU jobs and retry')
                torch.cuda.set_per_process_memory_fraction(limit/total)
            self.check_budget()
            scene = None
            try:
                prepared, preview, mask_source = prepare_input(image, remove_bg)
                with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                    scene = self.model([prepared], device=self.device)[0].float()
                if not torch.isfinite(scene).all():
                    raise RuntimeError('Nonfinite TripoSR scene representation')
                self.check_budget()
                density = self.density_volume(scene, resolution)
                raw = extract_surface(density, self.model.renderer.cfg.radius)
                del density
                geometry = {'watertight': bool(raw.is_watertight),
                            'components': int(len(trimesh.graph.connected_components(raw.face_adjacency, nodes=np.arange(len(raw.faces))))),
                            'faces_before_uv_seams': int(len(raw.faces))}
                print(f'Furniture: baking {atlas_resolution}px neural atlas', flush=True)
                canonical, faces, uv, texture, coverage = bake_neural_atlas(
                    raw, lambda points: self.query(scene, points, 'color'), atlas_resolution)
                vertices = to_room_coordinates(canonical, height)
                roughness = {'chair': .6, 'sofa': .75, 'bed': .75, 'table': .45, 'storage': .5}.get(category, .55)
                material = trimesh.visual.material.PBRMaterial(
                    name='Neural surface color', baseColorTexture=texture, metallicFactor=0.,
                    roughnessFactor=roughness, doubleSided=False)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                    visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)
                # UV seam vertices inherit smooth normals from the canonical surface,
                # avoiding per-chart shading discontinuities.
                # xatlas duplicates vertices but preserves their exact coordinates.
                from scipy.spatial import cKDTree
                _, mapping = cKDTree(raw.vertices).query(canonical, k=1)
                mesh.vertex_normals = raw.vertex_normals[mapping][:, [1, 2, 0]]
                glb = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)
                self.check_budget()
                bounds = mesh.bounds
                buffer = io.BytesIO(); preview.save(buffer, format='PNG')
                quality = {
                    'preset': preset, 'density_resolution': resolution, 'texture_resolution': atlas_resolution,
                    'texture_source': 'TripoSR neural color field sampled on unwrapped surface',
                    'mask_source': mask_source, 'atlas_coverage': coverage, **geometry,
                    'scale_source': 'user height' if target_height_m is not None else 'category height assumption',
                    'roughness_source': 'category preset; not measured PBR',
                    'checkpoint_loading': 'strict', 'elapsed_seconds': time.monotonic()-started,
                    'precision': 'FP16 encoder/transformer; FP32 field decoder' if self.device.type == 'cuda' else 'FP32',
                    'peak_device_sampled_mib': self.peak_device_mib,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2 if self.device.type == 'cuda' else 0,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2 if self.device.type == 'cuda' else 0,
                    'limitations': ['Hidden surfaces are inferred from one image',
                                    'Higher mesh/atlas resolution does not add missing model knowledge',
                                    'Metric scale needs a measured dimension'],
                }
                return {'status': 'success', 'id': f'custom_{category}_{uuid.uuid4().hex[:6]}',
                        'name': name, 'category': category,
                        'dimensions_m': np.round(mesh.extents[[0, 2, 1]], 3).tolist(),
                        'bounds': {'min': np.round(bounds[0], 3).tolist(), 'max': np.round(bounds[1], 3).tolist()},
                        'num_vertices': len(vertices), 'num_faces': len(faces), 'roughness': roughness,
                        'engine': 'TripoSR Quality: neural surface atlas', 'device': self.device_name,
                        'quality': quality, 'glb_bytes': glb,
                        'glb_base64': 'data:model/gltf-binary;base64,'+base64.b64encode(glb).decode(),
                        'preview_image_base64': 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}
            finally:
                del scene
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
