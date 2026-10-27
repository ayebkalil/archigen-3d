"""
Production Floorplan Inference, Vectorization, 3D Extrusion, and Facade Bridge Engine.
Performs:
1. Deep-learning floorplan segmentation using MultiTask UNet++:
   - Structural elements (walls, internal/external, doors, windows) -> 5 classes
   - Semantic room types (living room, bedroom, kitchen, bath, etc.) -> 12 classes
2. Geometric vectorization & rectilinear polygon simplification.
3. Facade Bridge texture synthesis for exterior walls via Pix2Pix.
4. Three.js 3D extrusion scene generation with textured materials.
"""

import sys
from pathlib import Path

def get_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        if (parent / "models").exists() and (parent / "src").exists():
            return parent
    return Path("/app") if Path("/app").exists() else p.parent.parent.parent

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional
import io
from PIL import Image
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file
    from src.floorplan.models import MultiTaskFloorplanUNet
    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False
    torch = None

import onnxruntime as ort
from scipy.special import softmax

from src.floorplan.vectorizer import FloorplanVectorizer
from src.floorplan.extruder_3d import Floorplan3DExtruder
from src.bridge.facade_bridge import FacadeBridge

class FloorplanPipeline:
    _instance = None

    def __init__(
        self,
        multitask_weights_path: Optional[str] = None,
        baseline_weights_path: str = "models/saved/floorplan_baseline/best.safetensors",
        device: Optional[str] = None
    ):
        self.device = None
        self.baseline_model = None
        self.multitask_model = None
        self.fallback_model = None
        self.onnx_multitask = None
        self.is_multitask = False

        if HAS_TORCH:
            if device is not None:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Resolve multitask weights candidate
            if multitask_weights_path is None:
                candidates = [
                    PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth",
                    PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_best.pth",
                    PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth"
                ]
                chosen_mt = None
                for cand in candidates:
                    if cand.exists():
                        chosen_mt = cand
                        break
                multitask_weights_path = str(chosen_mt or candidates[1])
            elif not Path(multitask_weights_path).is_absolute():
                multitask_weights_path = str(PROJECT_ROOT / multitask_weights_path)

            # 1. Load Baseline ResNet-34 UNet for high-precision structural elements (Walls, Doors, Windows)
            base_file = Path(baseline_weights_path)
            if not base_file.is_absolute():
                base_file = PROJECT_ROOT / baseline_weights_path
            if base_file.exists():
                try:
                    self.baseline_model = smp.Unet(encoder_name="resnet34", classes=4).to(self.device)
                    state_dict = load_file(str(base_file))
                    self.baseline_model.load_state_dict(state_dict)
                    self.baseline_model.eval()
                    print(f"[FLOORPLAN ENGINE] Loaded baseline structure model from {base_file}")
                except Exception as e:
                    print(f"[FLOORPLAN ENGINE] Error loading baseline weights: {e}")

            # 2. Load Multi-Task UNet++ for semantic room classification (12 room types)
            mt_file = Path(multitask_weights_path)
            if mt_file.exists():
                try:
                    ckpt = torch.load(str(mt_file), map_location=self.device, weights_only=False)
                    self.multitask_model = MultiTaskFloorplanUNet(
                        encoder_name="resnet34",
                        num_struct_classes=5,
                        num_room_classes=12
                    ).to(self.device)
                    self.multitask_model.load_state_dict(ckpt["model_state_dict"])
                    self.multitask_model.eval()
                    print(f"[FLOORPLAN ENGINE] Loaded MultiTask UNet++ from {mt_file} on {self.device}")
                except Exception as e:
                    print(f"[FLOORPLAN ENGINE] Warning loading multitask weights: {e}")

            # 3. Load Fallback Routing Model (Blend a=0.3 / Challenger) for zero-detection recovery
            self.fallback_model = None
            fallback_candidates = [
                PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth",
                PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth"
            ]
            for fb_cand in fallback_candidates:
                if fb_cand.exists():
                    try:
                        ckpt_fb = torch.load(str(fb_cand), map_location=self.device, weights_only=False)
                        self.fallback_model = MultiTaskFloorplanUNet(
                            encoder_name="resnet34",
                            num_struct_classes=5,
                            num_room_classes=12
                        ).to(self.device)
                        sd_fb = ckpt_fb.get("model_state_dict", ckpt_fb)
                        self.fallback_model.load_state_dict(sd_fb)
                        self.fallback_model.eval()
                        self.fallback_model_name = fb_cand.stem
                        print(f"[FLOORPLAN ENGINE] Loaded Fallback Router model from {fb_cand.name}")
                        break
                    except Exception as e:
                        print(f"[FLOORPLAN ENGINE] Warning loading fallback model: {e}")

            self.model = self.baseline_model or self.multitask_model
            self.is_multitask = self.multitask_model is not None
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        else:
            # High-performance lightweight ONNX Runtime execution (e.g. in Docker)
            onnx_path = PROJECT_ROOT / "models/saved/floorplan_unetpp/floorplan_multitask.onnx"
            if onnx_path.exists():
                try:
                    self.onnx_multitask = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                    self.is_multitask = True
                    print(f"[FLOORPLAN ENGINE] Loaded ONNX MultiTask UNet++ from {onnx_path.name} via ONNX Runtime")
                except Exception as e:
                    print(f"[FLOORPLAN ENGINE] Error loading ONNX model: {e}")

        self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        self.vectorizer = FloorplanVectorizer()
        self.extruder = Floorplan3DExtruder()
        self.bridge = FacadeBridge(
            onnx_model_path=str(PROJECT_ROOT / "models/saved/generator_perceptual.onnx"),
            pytorch_model_path=str(PROJECT_ROOT / "models/saved/generator_perceptual_best.pth"),
            prefer_cuda=(self.device is not None and getattr(self.device, "type", None) == "cuda")
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_image(
        self,
        image_bytes: bytes,
        scale_override: Optional[float] = None,
        num_floors: int = 1,
        front_wall_id: Optional[str] = None,
        use_tta: bool = False,
        tta_mode: str = "none",
        front_only: bool = True
    ) -> Dict[str, Any]:
        """
        Takes raw image bytes, runs multi-task segmentation, vectorizes to polygons,
        synthesizes facade textures for exterior walls, and produces 3D scene data.

        Parameters:
        - use_tta: When True, applies test-time augmentation (defaults to 4_fold).
        - tta_mode: "none" (ultra-fast single pass), "4_fold" (+1.56% mIoU), or "8_fold" (+1.70% mIoU, D4 dihedral).
        - front_only: When True, generates photorealistic neural texture strictly for the front facade,
                      reducing P95 latency tail from 710ms down to ~34ms.
        """
        raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        iw, ih = raw_img.size

        # Aspect-ratio preserving letterbox pad to 512x512
        size = 512
        scale = min(size / iw, size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2

        resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(resized, (pad_x, pad_y))

        active_mode = "single_pass"

        if HAS_TORCH:
            # Tensor conversion
            tensor = TF.to_tensor(canvas).to(self.device)
            tensor = (tensor - self.mean) / self.std
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                if self.multitask_model is not None:
                    effective_tta = tta_mode
                    if use_tta and effective_tta == "none":
                        effective_tta = "4_fold"

                    if effective_tta == "none" or self.device.type != "cuda":
                        outputs = self.multitask_model(tensor)
                        prob_s = F.softmax(outputs["struct"], dim=1)
                        prob_r = F.softmax(outputs["room"], dim=1)
                        active_mode = "single_pass"
                    elif effective_tta == "4_fold":
                        # 4-Fold TTA (Original + H-Flip + V-Flip + 180 Rotation)
                        out0 = self.multitask_model(tensor)
                        prob_s = F.softmax(out0["struct"], dim=1)
                        prob_r = F.softmax(out0["room"], dim=1)

                        t_h = torch.flip(tensor, dims=[3])
                        out_h = self.multitask_model(t_h)
                        prob_s += torch.flip(F.softmax(out_h["struct"], dim=1), dims=[3])
                        prob_r += torch.flip(F.softmax(out_h["room"], dim=1), dims=[3])

                        t_v = torch.flip(tensor, dims=[2])
                        out_v = self.multitask_model(t_v)
                        prob_s += torch.flip(F.softmax(out_v["struct"], dim=1), dims=[2])
                        prob_r += torch.flip(F.softmax(out_v["room"], dim=1), dims=[2])

                        t_hv = torch.flip(tensor, dims=[2, 3])
                        out_hv = self.multitask_model(t_hv)
                        prob_s += torch.flip(F.softmax(out_hv["struct"], dim=1), dims=[2, 3])
                        prob_r += torch.flip(F.softmax(out_hv["room"], dim=1), dims=[2, 3])

                        prob_s /= 4.0
                        prob_r /= 4.0
                        active_mode = "4_fold_tta"
                    elif effective_tta == "8_fold":
                        # Dihedral D4 8-Fold TTA (4 flips/rotations + transpose symmetries)
                        transforms = [
                            (lambda t: t, lambda p: p),
                            (lambda t: torch.flip(t, dims=[3]), lambda p: torch.flip(p, dims=[3])),
                            (lambda t: torch.flip(t, dims=[2]), lambda p: torch.flip(p, dims=[2])),
                            (lambda t: torch.flip(t, dims=[2, 3]), lambda p: torch.flip(p, dims=[2, 3])),
                            (lambda t: t.transpose(2, 3), lambda p: p.transpose(2, 3)),
                            (lambda t: torch.flip(t, dims=[3]).transpose(2, 3), lambda p: torch.flip(p.transpose(2, 3), dims=[3])),
                            (lambda t: torch.flip(t, dims=[2]).transpose(2, 3), lambda p: torch.flip(p.transpose(2, 3), dims=[2])),
                            (lambda t: torch.flip(t, dims=[2, 3]).transpose(2, 3), lambda p: torch.flip(p.transpose(2, 3), dims=[2, 3])),
                        ]
                        accum_s = None
                        accum_r = None
                        for fwd, inv in transforms:
                            t_aug = fwd(tensor)
                            out_aug = self.multitask_model(t_aug)
                            p_s = inv(F.softmax(out_aug["struct"], dim=1))
                            p_r = inv(F.softmax(out_aug["room"], dim=1))
                            accum_s = p_s if accum_s is None else accum_s + p_s
                            accum_r = p_r if accum_r is None else accum_r + p_r

                        prob_s = accum_s / 8.0
                        prob_r = accum_r / 8.0
                        active_mode = "8_fold_tta"
                    else:
                        outputs = self.multitask_model(tensor)
                        prob_s = F.softmax(outputs["struct"], dim=1)
                        prob_r = F.softmax(outputs["room"], dim=1)
                        active_mode = "single_pass"

                    struct_pred = torch.argmax(prob_s, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    room_pred = torch.argmax(prob_r, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    # Map 5-class structure (bg, wall_int, wall_ext, door, win) to 4-class for vectorizer
                    mask_4class = np.zeros_like(struct_pred)
                    mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1
                    mask_4class[struct_pred == 3] = 2
                    mask_4class[struct_pred == 4] = 3
                elif self.baseline_model is not None:
                    logits = self.baseline_model(tensor)
                    mask_4class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    struct_pred = None
                    room_pred = None
                else:
                    mask_4class = np.zeros((size, size), dtype=np.uint8)
                    struct_pred = None
                    room_pred = None
        elif self.onnx_multitask is not None:
            # High-speed CPU inference with ONNX Runtime
            arr = np.array(canvas, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = (arr - self.mean_np) / self.std_np
            tensor_np = np.expand_dims(arr, 0)

            outs = self.onnx_multitask.run(None, {'input': tensor_np})
            prob_s = softmax(outs[0], axis=1)
            prob_r = softmax(outs[1], axis=1)

            struct_pred = np.argmax(prob_s, axis=1).squeeze(0).astype(np.uint8)
            room_pred = np.argmax(prob_r, axis=1).squeeze(0).astype(np.uint8)

            mask_4class = np.zeros_like(struct_pred)
            mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1
            mask_4class[struct_pred == 3] = 2
            mask_4class[struct_pred == 4] = 3
            active_mode = "onnx_single_pass"
        else:
            mask_4class = np.zeros((size, size), dtype=np.uint8)
            struct_pred = None
            room_pred = None

        # 2. Geometric Vectorization
        layout = self.vectorizer.vectorize(
            mask=mask_4class,
            struct_mask=struct_pred,
            room_mask=room_pred,
            scale_override_m_per_px=scale_override
        )

        model_routed = "champion_v1_1"
        fallback_triggered = False

        # Fallback Routing: triggered only on genuine structural collapses
        # (0 detected rooms, invalid building shell, or single-room artifact under 8.0 m2).
        # This isolates true model failures while preserving legitimate 1-2 room studio apartments.
        primary_rooms = layout.get("rooms", [])
        primary_area = layout.get("metadata", {}).get("total_surface_m2", 0.0)
        is_shell_valid = layout.get("metadata", {}).get("is_building_shell_valid", False)

        is_collapse = (len(primary_rooms) == 0) or (not is_shell_valid) or (len(primary_rooms) == 1 and primary_area < 8.0)

        if is_collapse and HAS_TORCH and self.fallback_model is not None:
            with torch.no_grad():
                fb_out = self.fallback_model(tensor)
                fb_prob_s = F.softmax(fb_out["struct"], dim=1)
                fb_prob_r = F.softmax(fb_out["room"], dim=1)
                fb_struct = torch.argmax(fb_prob_s, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                fb_room = torch.argmax(fb_prob_r, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                fb_m4 = np.zeros_like(fb_struct)
                fb_m4[(fb_struct == 1) | (fb_struct == 2)] = 1
                fb_m4[fb_struct == 3] = 2
                fb_m4[fb_struct == 4] = 3

                fb_layout = self.vectorizer.vectorize(
                    mask=fb_m4,
                    struct_mask=fb_struct,
                    room_mask=fb_room,
                    scale_override_m_per_px=scale_override
                )
                fb_rooms = fb_layout.get("rooms", [])
                fb_shell_valid = fb_layout.get("metadata", {}).get("is_building_shell_valid", False)

                # Adopt fallback if it found more rooms, or if primary shell was invalid and fallback produced a valid shell
                if len(fb_rooms) > len(primary_rooms) or (not is_shell_valid and fb_shell_valid):
                    layout = fb_layout
                    model_routed = getattr(self, "fallback_model_name", "fallback_adapted")
                    fallback_triggered = True

        if "metadata" not in layout:
            layout["metadata"] = {}
        layout["metadata"]["routing"] = {
            "model_used": model_routed,
            "fallback_triggered": fallback_triggered
        }

        # 3. Facade Bridge: synthesize textures for exterior walls
        facade_meta = self.bridge.process_layout_facades(
            layout=layout,
            front_wall_id=front_wall_id,
            num_floors=num_floors,
            front_only=front_only
        )

        # 4. 3D Scene Generation
        threejs_scene = self.extruder.generate_threejs_scene_data(layout)

        return {
            "layout": layout,
            "threejs_scene": threejs_scene,
            "front_wall_id": facade_meta.get("front_wall_id"),
            "exterior_walls_count": facade_meta.get("exterior_walls_count", 0),
            "inference_mode": active_mode,
            "wall_mask": (mask_4class == 1).astype(np.uint8)
        }
