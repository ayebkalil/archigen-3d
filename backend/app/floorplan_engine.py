"""
Production Floorplan Inference, Vectorization, and 3D Extrusion Engine.
Performs:
1. Deep-learning floorplan segmentation (background, wall, door, window) using SMP UNet.
2. Geometric vectorization & rectilinear polygon simplification.
3. Three.js 3D extrusion scene generation.
"""

from pathlib import Path
from typing import Dict, Any
import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from safetensors.torch import load_file

from src.floorplan.vectorizer import FloorplanVectorizer
from src.floorplan.extruder_3d import Floorplan3DExtruder

class FloorplanPipeline:
    _instance = None

    def __init__(self, weights_path: str = "models/saved/floorplan_baseline/best.safetensors"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(encoder_name="resnet34", classes=4).to(self.device)
        
        weights_file = Path(weights_path)
        if weights_file.exists():
            state_dict = load_file(str(weights_file))
            self.model.load_state_dict(state_dict)
            print(f"[FLOORPLAN ENGINE] Loaded pretrained weights from {weights_file}")
        else:
            print(f"[FLOORPLAN ENGINE] Warning: weights not found at {weights_file}, running with base weights")

        self.model.eval()
        self.vectorizer = FloorplanVectorizer()
        self.extruder = Floorplan3DExtruder()

        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_image(self, image_bytes: bytes, scale_override: Any = None) -> Dict[str, Any]:
        """
        Takes raw image bytes, runs segmentation, vectorizes to polygons, and returns 3D scene data.
        Supports manual scale override in meters per pixel.
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
            logits = self.model(tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 2. Geometric Vectorization (with exterior detection & scale override)
        layout = self.vectorizer.vectorize(pred_mask, scale_override_m_per_px=scale_override)

        # 3. 3D Scene Generation
        threejs_scene = self.extruder.generate_threejs_scene_data(layout)

        return {
            "layout": layout,
            "threejs_scene": threejs_scene
        }

