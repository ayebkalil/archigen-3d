"""
CubiCasa5k PyTorch Dataset and Ground-Truth Rasterizer.
Parses floorplan images (F1_scaled.png) and vector annotations (model.svg)
producing 512x512 letterboxed image tensors and multi-class segmentation masks.
Classes:
  0: Background / Floor
  1: Wall
  2: Door
  3: Window
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def parse_svg_points(pts_str: str, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """Parses SVG points attribute string into scaled integer pixel coordinates."""
    clean = pts_str.replace(",", " ").split()
    pts = []
    for i in range(0, len(clean), 2):
        if i + 1 < len(clean):
            x = float(clean[i]) * scale + pad_x
            y = float(clean[i+1]) * scale + pad_y
            pts.append([int(round(x)), int(round(y))])
    return np.array(pts, dtype=np.int32)

class CubiCasaDataset(Dataset):
    """
    High-performance PyTorch Dataset for CubiCasa5k Floorplan Segmentation.
    Features:
    1. Aspect-ratio preserving letterbox pad to target image size (default 512x512).
    2. Real-time SVG polygon vector rasterization into 4-class ground truth masks.
    3. Spatial augmentations: 90 deg discrete rotations, horizontal/vertical flips.
    """
    def __init__(
        self,
        root_dir: str = "data/CubiCasa5k/cubicasa5k/cubicasa5k",
        split: str = "train",
        img_size: int = 512,
        augment: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.split_file = self.root_dir / f"{split}.txt"
        self.img_size = img_size
        self.augment = augment and (split == "train")

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file, "r") as f:
            self.samples = [line.strip().lstrip("/") for line in f if line.strip()]

        # Filter only existing directories
        self.valid_samples = [
            s for s in self.samples
            if (self.root_dir / s / "model.svg").exists() and (self.root_dir / s / "F1_scaled.png").exists()
        ]
        print(f"[CubiCasaDataset] Loaded split '{split}': {len(self.valid_samples)} valid floorplans found.")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_rel_path = self.valid_samples[idx]
        folder = self.root_dir / sample_rel_path
        img_path = folder / "F1_scaled.png"
        svg_path = folder / "model.svg"

        # 1. Load image
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        # 2. Parse SVG dimensions
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            vb = root.attrib.get("viewBox", "").split()
            if vb and len(vb) == 4:
                svg_w, svg_h = float(vb[2]), float(vb[3])
            else:
                svg_w, svg_h = float(root.attrib.get("width", iw)), float(root.attrib.get("height", ih))
        except Exception:
            svg_w, svg_h = float(iw), float(ih)

        # 3. Calculate letterbox scaling
        size = self.img_size
        scale = min(size / svg_w, size / svg_h)
        nw, nh = int(svg_w * scale), int(svg_h * scale)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2

        # Resize image
        img_resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas_img = Image.new("RGB", (size, size), (255, 255, 255))
        canvas_img.paste(img_resized, (pad_x, pad_y))

        # 4. Rasterize ground-truth mask
        # 0: Background, 1: Wall, 2: Door, 3: Window
        mask = np.zeros((size, size), dtype=np.uint8)

        for elem in root.iter():
            c = elem.attrib.get("class", "")
            for child in elem.iter():
                tag = child.tag.split("}")[-1]
                if tag == "polygon" and "points" in child.attrib:
                    pts = parse_svg_points(child.attrib["points"], scale, pad_x, pad_y)
                    if len(pts) >= 3:
                        if "Wall" in c:
                            cv2.fillPoly(mask, [pts], 1)
                        elif "Door" in c:
                            cv2.fillPoly(mask, [pts], 2)
                        elif "Window" in c:
                            cv2.fillPoly(mask, [pts], 3)

        # 5. Data Augmentation (90 deg rotations, flips)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                canvas_img = TF.hflip(canvas_img)
                mask = np.fliplr(mask)
            # Random vertical flip
            if np.random.rand() > 0.5:
                canvas_img = TF.vflip(canvas_img)
                mask = np.flipud(mask)
            # Random 90 deg rotation (preserves rectilinear orientation)
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                canvas_img = canvas_img.rotate(rot_k * 90)
                mask = np.rot90(mask, rot_k)

        # 6. Tensor conversion and normalization
        img_tensor = TF.to_tensor(canvas_img)
        img_tensor = (img_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "sample_id": sample_rel_path
        }
