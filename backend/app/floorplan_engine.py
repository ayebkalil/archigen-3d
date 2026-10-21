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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional
import io
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from safetensors.torch import load_file

from src.floorplan.models import MultiTaskFloorplanUNet
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
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.baseline_model = None
        self.multitask_model = None

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

        self.model = self.baseline_model or self.multitask_model
        self.is_multitask = self.multitask_model is not None

        self.vectorizer = FloorplanVectorizer()
        self.extruder = Floorplan3DExtruder()
        self.bridge = FacadeBridge(
            onnx_model_path=str(PROJECT_ROOT / "models/saved/generator_perceptual.onnx"),
            pytorch_model_path=str(PROJECT_ROOT / "models/saved/generator_perceptual_best.pth"),
            prefer_cuda=(self.device.type == "cuda")
        )

        # ImageNet Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

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
        front_only: bool = True
    ) -> Dict[str, Any]:
        """
        Takes raw image bytes, runs multi-task segmentation, vectorizes to polygons,
        synthesizes facade textures for exterior walls, and produces 3D scene data.

        Parameters:
        - use_tta: When True, applies 4-fold test-time augmentation (+1.56% mIoU, +0.90% Door IoU).
                   Disabled by default on CPU for ultra-low latency.
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

        # Tensor conversion
        tensor = TF.to_tensor(canvas).to(self.device)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            if self.multitask_model is not None:
                if not use_tta or self.device.type != "cuda":
                    outputs = self.multitask_model(tensor)
                    prob_s = F.softmax(outputs["struct"], dim=1)
                    prob_r = F.softmax(outputs["room"], dim=1)
                else:
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

        # 2. Geometric Vectorization
        layout = self.vectorizer.vectorize(
            mask=mask_4class,
            struct_mask=struct_pred,
            room_mask=room_pred,
            scale_override_m_per_px=scale_override
        )

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
            "inference_mode": "4_fold_tta" if (use_tta and self.device.type == "cuda") else "single_pass"
        }
