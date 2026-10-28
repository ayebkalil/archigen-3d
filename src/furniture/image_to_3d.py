"""
Industrialized Image-to-3D Furniture Generation Engine for ArchiGen 3D.
Optimized for RTX 3050 (4GB VRAM) laptop architecture.

Features:
1. Automated AI background removal (u2netp / rembg) with OpenCV GrabCut adaptive fallback.
2. Intelligent alpha thresholding to eliminate floor drop-shadows and gradient walls.
3. Category-aware Volumetric Surface Synthesizer (sculpted depth profiles for chairs, tables, sofas, beds, cabinets).
4. Volume-preserving Taubin smoothing filter to eliminate voxel stepping.
5. Direct binary GLB export with UV coordinates, vertex normals, and PBR textures.
6. Automatic physical scaling and bounding-box calibration for Room Design Engine.
"""

import io
import math
import uuid
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
import trimesh
from PIL import Image

# Standard architectural furniture bounding boxes (Width, Depth, Height in meters)
CATEGORY_DIMENSIONS = {
    "chair": [0.60, 0.60, 0.85],
    "table": [1.20, 0.80, 0.75],
    "sofa": [1.80, 0.85, 0.80],
    "bed": [1.60, 2.00, 0.65],
    "storage": [0.90, 0.45, 1.20],
    "decor": [0.40, 0.40, 0.60],
    "lighting": [0.45, 0.45, 1.40],
}


class ImageTo3DGenerator:
    _instance = None

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.tsr_model = None
        self.device_name = "CPU"
        self.rembg_session = None

        # 1. Initialize GPU if available
        try:
            import torch
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
            else:
                self.device = torch.device("cpu")
                self.device_name = "CPU"
        except Exception:
            self.device = None
            self.device_name = "CPU"

        # 2. Try loading TripoSR if checkpoint exists
        self._init_triposr()

    def _init_triposr(self):
        """Attempts to load TripoSR onto GPU if local model checkpoint is available."""
        try:
            import torch
            import sys
            triposr_dir = Path(__file__).resolve().parent.parent.parent / "TripoSR"
            if triposr_dir.exists() and str(triposr_dir) not in sys.path:
                sys.path.insert(0, str(triposr_dir))

            from tsr.system import TSR

            ckpt_path = triposr_dir / "model.ckpt"
            cfg_path = triposr_dir / "config.yaml"

            # Check if model.ckpt exists and has full size (~1.6GB)
            if ckpt_path.exists() and ckpt_path.stat().st_size > 1.5e9 and cfg_path.exists():
                print(f"[IMAGE-TO-3D] Loading TripoSR from {ckpt_path} onto {self.device}...")
                model = TSR.from_pretrained(
                    str(triposr_dir),
                    config_name="config.yaml",
                    weight_name="model.ckpt"
                )
                model.renderer.set_chunk_size(4096)
                if self.device is not None:
                    model.to(self.device)
                model.eval()
                self.tsr_model = model
                print("[IMAGE-TO-3D] TripoSR neural model active on GPU!")
            else:
                print("[IMAGE-TO-3D] High-precision Architectural Volumetric Synthesizer active.")
        except Exception as e:
            print(f"[IMAGE-TO-3D] TripoSR initialization deferred: {e}")

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        High-precision background removal.
        Uses u2netp neural segmenter with OpenCV GrabCut adaptive fallback.
        Completely eliminates background walls and soft floor shadows.
        """
        # 1. If image is already transparent PNG (> 15% transparent pixels), keep it
        if img.mode == "RGBA":
            alpha = np.array(img)[:, :, 3]
            if (alpha < 30).mean() > 0.15:
                # Filter residual soft floor shadow
                arr = np.array(img)
                arr[:, :, 3] = np.where(arr[:, :, 3] < 80, 0, arr[:, :, 3])
                return Image.fromarray(arr, mode="RGBA")

        # 2. Primary: Lightweight u2netp (~4.5MB ONNX)
        try:
            import rembg
            if self.rembg_session is None:
                self.rembg_session = rembg.new_session("u2netp")
            clean_rgba = rembg.remove(img, session=self.rembg_session, post_process_mask=True)
            arr = np.array(clean_rgba)
            # Filter low-opacity drop shadows on the floor
            arr[:, :, 3] = np.where(arr[:, :, 3] < 85, 0, arr[:, :, 3])
            alpha_test = arr[:, :, 3]
            # Ensure meaningful foreground was detected
            if 0.03 < (alpha_test > 100).mean() < 0.90:
                return Image.fromarray(arr, mode="RGBA")
        except Exception as e:
            print(f"[IMAGE-TO-3D] rembg failed, falling back to OpenCV GrabCut: {e}")

        # 3. Secondary: OpenCV GrabCut + morphological cleanup (runs 100% locally)
        try:
            rgb = np.array(img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            # Margin for bounding box
            rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
            cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = fg_mask
            return Image.fromarray(rgba, mode="RGBA")
        except Exception as e:
            print(f"[IMAGE-TO-3D] GrabCut fallback failed: {e}")
            return img.convert("RGBA")

    def estimate_surface_roughness(self, img_rgba: Image.Image) -> float:
        """
        Estimates physical PBR roughness (0.18 - 0.85) from high-frequency image texture.
        Smooth polished surfaces (lacquered wood, marble, metal) -> 0.18 - 0.35.
        Rough coarse surfaces (rustic wood, canvas, stone, fabric) -> 0.65 - 0.85.
        """
        try:
            arr = np.array(img_rgba)
            alpha = arr[:, :, 3] > 80
            if not alpha.any():
                return 0.35
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = float(np.var(lap[alpha]))
            norm = np.clip((var - 120.0) / 600.0, 0.0, 1.0)
            return round(float(0.20 + norm * 0.60), 2)
        except Exception:
            return 0.35

    def _get_category_depth_profile(
        self,
        category: str,
        y_vals: np.ndarray,
        norm_dist: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes category-specific architectural depth envelope and center offset.
        Prevents chairs and tables from extruding into bulky blocks.
        """
        cat = (category or "chair").lower()
        if "chair" in cat or "stool" in cat:
            # y_vals: 1.0 (top backrest) to 0.0 (bottom legs)
            depth = np.where(
                y_vals < 0.45,
                np.clip(norm_dist * 1.5, 0.12, 0.75),             # Thin cylindrical legs
                np.where(
                    y_vals < 0.60,
                    np.clip(norm_dist * 1.8 + 0.35, 0.40, 1.0),   # Seat cushion / plate
                    np.clip(norm_dist * 0.70 + 0.10, 0.10, 0.45)  # Thin backrest slats
                )
            )
            # Tilt backrest slightly backward along Z axis
            z_offset = np.where(y_vals > 0.58, -0.25 * (y_vals - 0.58) / 0.42, 0.0)
            return depth, z_offset

        elif "table" in cat or "desk" in cat:
            # Tabletop on top, thin legs / posts below
            depth = np.where(
                y_vals > 0.82,
                np.clip(norm_dist * 1.5 + 0.50, 0.70, 1.0),       # Tabletop surface
                np.clip(norm_dist * 1.4, 0.10, 0.60)              # Legs / pedestal
            )
            return depth, np.zeros_like(y_vals)

        elif "sofa" in cat or "couch" in cat:
            depth = np.where(
                y_vals < 0.55,
                np.clip(norm_dist * 1.6 + 0.40, 0.50, 1.0),       # Deep base & seat
                np.clip(norm_dist * 1.0 + 0.20, 0.20, 0.60)       # Back cushions
            )
            z_offset = np.where(y_vals > 0.55, -0.20 * (y_vals - 0.55) / 0.45, 0.0)
            return depth, z_offset

        elif "bed" in cat:
            depth = np.where(
                y_vals > 0.65,
                np.clip(norm_dist * 0.60 + 0.15, 0.10, 0.40),     # Headboard
                np.clip(norm_dist * 1.80 + 0.40, 0.60, 1.0)       # Mattress & frame
            )
            z_offset = np.where(y_vals > 0.65, -0.40, 0.0)
            return depth, z_offset

        elif "storage" in cat or "cabinet" in cat or "wardrobe" in cat:
            depth = np.clip(norm_dist * 1.20 + 0.50, 0.60, 1.0)
            return depth, np.zeros_like(y_vals)

        else:
            # Generic item (decor, plant, lighting)
            depth = np.clip(norm_dist * 1.40 + 0.20, 0.20, 0.90)
            return depth, np.zeros_like(y_vals)

    def _generate_volumetric_mesh(
        self,
        img_rgba: Image.Image,
        target_dims: List[float],
        category: str = "chair"
    ) -> trimesh.Trimesh:
        """
        High-precision architectural volumetric 3D reconstruction.
        - Distance-field local thickness calibration.
        - Category-aware structural depth envelope.
        - Watertight marching cubes isosurface.
        - Taubin volume-preserving smoothing filter.
        - Front & back planar UV mapping.
        """
        arr = np.array(img_rgba)
        alpha = (arr[:, :, 3] > 80).astype(np.uint8)

        # Find tight bounding box of isolated object
        y_indices, x_indices = np.where(alpha > 0)
        if len(y_indices) == 0:
            return trimesh.creation.box(extents=target_dims)

        y0, y1 = max(0, y_indices.min() - 2), min(arr.shape[0] - 1, y_indices.max() + 2)
        x0, x1 = max(0, x_indices.min() - 2), min(arr.shape[1] - 1, x_indices.max() + 2)
        cropped_rgba = img_rgba.crop((x0, y0, x1 + 1, y1 + 1))
        cropped_alpha = alpha[y0:y1 + 1, x0:x1 + 1]

        # Resolution for voxel marching cubes
        res = 72
        dist = cv2.distanceTransform(cropped_alpha, cv2.DIST_L2, 5)
        norm_dist = cv2.resize(dist, (res, res))
        norm_dist = norm_dist / max(1.0, norm_dist.max())
        v_mask = cv2.resize(cropped_alpha, (res, res), interpolation=cv2.INTER_NEAREST)

        # 3D Grid: Z x Y x X
        z_vals = np.linspace(-1.0, 1.0, res, dtype=np.float32)[:, None, None]
        y_vals = np.linspace(1.0, 0.0, res, dtype=np.float32)[None, :, None]

        # Apply category depth profile
        depth_profile, z_offset = self._get_category_depth_profile(category, y_vals, norm_dist[None, :, :])
        z_dist = np.abs(z_vals - z_offset)
        vol = np.where(
            (v_mask[None, :, :] > 0) & (z_dist < depth_profile),
            depth_profile - z_dist,
            -1.0
        ).astype(np.float32)

        # Marching cubes extraction
        verts, faces = None, None
        try:
            import mcubes
            vol_contiguous = np.ascontiguousarray(vol, dtype=np.float32)
            verts, faces = mcubes.marching_cubes(vol_contiguous, 0.0)
        except Exception:
            from skimage.measure import marching_cubes as sk_mc
            verts, faces, _, _ = sk_mc(vol, level=0.0)

        if len(verts) == 0 or len(faces) == 0:
            return trimesh.creation.box(extents=target_dims)

        # Coordinates mapping to physical architectural scale
        t_w, t_d, t_h = target_dims
        vx = (verts[:, 2] / (res - 1.0) - 0.5) * t_w
        vy = ((res - 1.0 - verts[:, 1]) / (res - 1.0)) * t_h  # Height upward
        vz = (verts[:, 0] / (res - 1.0) - 0.5) * t_d          # Depth forward

        v_scaled = np.column_stack([vx, vy, vz])
        mesh = trimesh.Trimesh(vertices=v_scaled, faces=faces, process=True)
        mesh.fix_normals()

        # Apply Taubin smoothing to eliminate voxel stepping
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=10)
            mesh.fix_normals()
        except Exception:
            pass

        # Projected UV mapping
        uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
        uvs[:, 0] = np.clip(verts[:, 2] / (res - 1.0), 0.0, 1.0)
        uvs[:, 1] = np.clip((res - 1.0 - verts[:, 1]) / (res - 1.0), 0.0, 1.0)

        # Diffuse texture embedding with PBR material
        rgb_tex = cropped_rgba.convert("RGB")
        roughness_val = self.estimate_surface_roughness(cropped_rgba)
        pbr_mat = trimesh.visual.material.PBRMaterial(
            name=f"{category}_material",
            baseColorTexture=rgb_tex,
            metallicFactor=0.04,
            roughnessFactor=roughness_val,
            doubleSided=True
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr_mat)

        return mesh

    def generate_from_image(
        self,
        image: Image.Image,
        name: str = "Custom Furniture",
        category: str = "chair",
        remove_bg: bool = True,
        target_height_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end 2D-to-3D generation.
        Returns GLB bytes, base64 payload, and catalog placement metadata.
        """
        category_clean = category.lower().strip()
        base_dims = CATEGORY_DIMENSIONS.get(category_clean, [0.60, 0.60, 0.85]).copy()
        if target_height_m and target_height_m > 0.2:
            scale_ratio = target_height_m / base_dims[2]
            base_dims = [base_dims[0] * scale_ratio, base_dims[1] * scale_ratio, target_height_m]

        # 1. Background removal
        if remove_bg:
            processed_img = self.remove_background(image)
        else:
            processed_img = image.convert("RGBA")

        # Ensure image is in RGBA format
        if processed_img.mode != "RGBA":
            processed_img = processed_img.convert("RGBA")

        # 2. Surface roughness estimation and tight texture cropping with color-bleed background
        roughness_val = self.estimate_surface_roughness(processed_img)

        arr_clean = np.array(processed_img)
        alpha_clean = arr_clean[:, :, 3] > 60
        if alpha_clean.any():
            mean_color = arr_clean[alpha_clean, :3].mean(axis=0).astype(np.uint8)
        else:
            mean_color = np.array([160, 140, 120], dtype=np.uint8)

        # Fill transparent background with mean material tone so no UV sampling ever hits black
        solid_rgb = np.where(alpha_clean[:, :, None], arr_clean[:, :, :3], mean_color)

        y_idxs, x_idxs = np.where(alpha_clean)
        if len(y_idxs) > 0:
            x0, x1 = max(0, x_idxs.min() - 2), min(arr_clean.shape[1] - 1, x_idxs.max() + 2)
            y0, y1 = max(0, y_idxs.min() - 2), min(arr_clean.shape[0] - 1, y_idxs.max() + 2)
            cropped_rgb = Image.fromarray(solid_rgb[y0:y1 + 1, x0:x1 + 1])
        else:
            cropped_rgb = Image.fromarray(solid_rgb)

        # 3. Neural TripoSR vs Volumetric Generation
        engine_used = "Architectural Surface Synthesizer"
        mesh = None

        if self.tsr_model is not None:
            try:
                import torch
                from tsr.utils import resize_foreground

                with torch.no_grad():
                    # TripoSR image preprocessing
                    img_fg = resize_foreground(processed_img, 0.85)
                    img_arr = np.array(img_fg).astype(np.float32) / 255.0
                    img_arr = img_arr[:, :, :3] * img_arr[:, :, 3:4] + (1 - img_arr[:, :, 3:4]) * 0.5
                    input_img = Image.fromarray((img_arr * 255.0).astype(np.uint8))

                    scene_codes = self.tsr_model([input_img], device=self.device)
                    # High-definition resolution 192 captures delicate slats and spindles with ~15k-25k vertices
                    meshes = self.tsr_model.extract_mesh(scene_codes, has_vertex_color=True, resolution=192)
                    raw_mesh = meshes[0]

                    # Gentle Taubin smoothing: eliminates voxel stepping while preserving thin slats & sturdy legs
                    try:
                        trimesh.smoothing.filter_taubin(raw_mesh, lamb=0.35, nu=-0.37, iterations=3)
                        raw_mesh.fix_normals()
                    except Exception as e:
                        print(f"[IMAGE-TO-3D] Taubin smoothing fallback: {e}")

                    # Verified Axis Mapping to Three.js coordinates:
                    # In TripoSR marching cubes:
                    # Axis 0: Vertical Height (floor to top)
                    # Axis 1: Horizontal Width (left to right)
                    # Axis 2: Depth (front to back)
                    # Three.js coordinate system:
                    # X: Width, Y: Height (floor is Y=0), Z: Depth
                    verts = raw_mesh.vertices.copy()
                    b_min = verts.min(axis=0)
                    b_max = verts.max(axis=0)

                    vx = verts[:, 1] - (b_min[1] + b_max[1]) * 0.5
                    vy = verts[:, 0] - b_min[0]
                    vz = verts[:, 2] - (b_min[2] + b_max[2]) * 0.5

                    v_scaled = np.column_stack([vx, vy, vz])

                    # Proportional height scaling: preserves true physical proportions and leg thickness
                    t_w, t_d, t_h = base_dims
                    curr_span = v_scaled.max(axis=0) - v_scaled.min(axis=0)
                    scale_factor = t_h / max(curr_span[1], 0.01)
                    v_scaled *= scale_factor

                    # Ground feet alignment (pitch correction for elevated camera in 2D photos)
                    z_med = np.median(v_scaled[:, 2])
                    front_mask = v_scaled[:, 2] > (z_med + 0.05)
                    back_mask = v_scaled[:, 2] < (z_med - 0.05)
                    if front_mask.any() and back_mask.any():
                        front_min = np.percentile(v_scaled[front_mask, 1], 1)
                        back_min = np.percentile(v_scaled[back_mask, 1], 1)
                        front_z = np.median(v_scaled[front_mask & (v_scaled[:, 1] < front_min + 0.05), 2])
                        back_z = np.median(v_scaled[back_mask & (v_scaled[:, 1] < back_min + 0.05), 2])
                        delta_y = back_min - front_min
                        delta_z = back_z - front_z
                        if abs(delta_z) > 0.05:
                            th_feet = -math.atan(delta_y / delta_z)
                            th_feet = np.clip(th_feet, -math.radians(20.0), math.radians(20.0))
                            cx, sx = math.cos(th_feet), math.sin(th_feet)
                            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])
                            v_scaled = v_scaled @ Rx.T

                    # Top surface leveling for tables/desks
                    if "table" in category_clean or "desk" in category_clean:
                        for _ in range(4):
                            top_cutoff = v_scaled[:, 1].max() - 0.10 * (v_scaled[:, 1].max() - v_scaled[:, 1].min())
                            top_idx = np.where(v_scaled[:, 1] > top_cutoff)[0]
                            if len(top_idx) < 30:
                                break
                            top_pts = v_scaled[top_idx]
                            A = np.column_stack([top_pts[:, 0], top_pts[:, 2], np.ones(len(top_pts))])
                            a, b, _ = np.linalg.lstsq(A, top_pts[:, 1], rcond=None)[0]
                            if math.hypot(a, b) < 0.005:
                                break
                            th_x = -math.atan(b)
                            th_z = math.atan(a)
                            cx, sx = math.cos(th_x), math.sin(th_x)
                            cz, sz = math.cos(th_z), math.sin(th_z)
                            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])
                            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
                            v_scaled = v_scaled @ (Rz @ Rx).T

                    # Ground alignment: floor is Y = 0
                    v_scaled[:, 1] -= v_scaled[:, 1].min()

                    # Firm floor contact: soft clamp for feet within 2.5cm of ground
                    near_floor = v_scaled[:, 1] < 0.025
                    v_scaled[near_floor, 1] = 0.0

                    # Perspective UV Projection with camera elevation angle (20 deg)
                    alpha_rad = math.radians(20.0)
                    u_cam = v_scaled[:, 0]
                    v_cam = v_scaled[:, 1] * math.cos(alpha_rad) - v_scaled[:, 2] * math.sin(alpha_rad)

                    u = np.clip((u_cam - u_cam.min()) / (u_cam.max() - u_cam.min() + 1e-6), 0.01, 0.99)
                    # IN GLTF 2.0: v=0 is TOP of image, v=1 is BOTTOM of image
                    v = np.clip(1.0 - (v_cam - v_cam.min()) / (v_cam.max() - v_cam.min() + 1e-6), 0.01, 0.99)
                    uvs = np.column_stack([u, v]).astype(np.float32)

                    # Physically Based Rendering (PBR) Material with DoubleSided geometry
                    pbr_mat = trimesh.visual.material.PBRMaterial(
                        name=f"{category_clean}_material",
                        baseColorTexture=cropped_rgb,
                        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                        metallicFactor=0.04,
                        roughnessFactor=roughness_val,
                        doubleSided=True
                    )
                    mesh = trimesh.Trimesh(
                        vertices=v_scaled,
                        faces=raw_mesh.faces,
                        visual=trimesh.visual.TextureVisuals(uv=uvs, material=pbr_mat),
                        process=True
                    )
                    mesh.fix_normals()

                    engine_used = f"TripoSR Neural 3D ({torch.cuda.get_device_name(0)})"
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[IMAGE-TO-3D] TripoSR inference fallback: {e}")
                mesh = None

        if mesh is None:
            mesh = self._generate_volumetric_mesh(processed_img, base_dims, category=category_clean)

        # 4. Export to binary GLB with explicit vertex normals
        glb_bytes = trimesh.exchange.gltf.export_glb(mesh, include_normals=True)

        # 5. Preview base64 serialization
        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        preview_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        glb_base64 = f"data:model/gltf-binary;base64,{base64.b64encode(glb_bytes).decode('ascii')}"

        # 6. Metadata for Room Design placement
        item_id = f"custom_{category_clean}_{uuid.uuid4().hex[:6]}"
        bounds = mesh.bounds
        dimensions_m = [
            round(float(bounds[1][0] - bounds[0][0]), 3),
            round(float(bounds[1][2] - bounds[0][2]), 3),
            round(float(bounds[1][1] - bounds[0][1]), 3)
        ]

        return {
            "status": "success",
            "id": item_id,
            "name": name,
            "category": category_clean,
            "dimensions_m": dimensions_m,
            "bounds": {
                "min": [round(float(x), 3) for x in bounds[0]],
                "max": [round(float(x), 3) for x in bounds[1]]
            },
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "roughness": roughness_val,
            "glb_bytes": glb_bytes,
            "glb_base64": glb_base64,
            "preview_image_base64": preview_base64,
            "engine": engine_used,
            "device": self.device_name
        }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
