"""
Restyled CubiCasa Dataset for Procedural Style Domain Adaptation.
Draws vector floorplans in 6 distinct CAD/architectural styles on the fly with 50% probability
during training, exposing the UNet++ encoder to blueprints, hatching, dimension markings,
and varied line weights while keeping 100% exact vector ground-truth masks.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, STRUCTURE_CLASSES
from src.floorplan.style_renderer import render_floorplan_style

class RestyledCubiCasaDataset(CubiCasaDataset):
    """
    Extends CubiCasaDataset to randomly render procedural styles on-the-fly.
    50% probability of standard CubiCasa rendering, 50% procedural style (CAD, hatching, B&W, etc.)
    """
    def __init__(
        self,
        p_restyle: float = 0.50,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p_restyle = p_restyle

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_rel_path = self.valid_samples[idx]
        folder = self.root_dir / sample_rel_path
        svg_path = folder / "model.svg"

        is_restyle = self.augment and (np.random.rand() < self.p_restyle)

        if is_restyle and svg_path.exists():
            try:
                canvas_img, struct_mask, room_mask = render_floorplan_style(
                    svg_path, img_size=self.img_size, style="random"
                )
                mask_4class = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                mask_4class[(struct_mask == 1) | (struct_mask == 2)] = 1
                mask_4class[struct_mask == 3] = 2
                mask_4class[struct_mask == 4] = 3
            except Exception:
                # Fallback to standard cached/rasterized sample if SVG parsing fails
                return super().__getitem__(idx)
        else:
            return super().__getitem__(idx)

        # Standard Spatial Augmentations (flips & 90 deg rotations)
        if self.augment:
            if np.random.rand() > 0.5:
                canvas_img = TF.hflip(canvas_img)
                struct_mask = np.fliplr(struct_mask)
                room_mask = np.fliplr(room_mask)
                mask_4class = np.fliplr(mask_4class)
            if np.random.rand() > 0.5:
                canvas_img = TF.vflip(canvas_img)
                struct_mask = np.flipud(struct_mask)
                room_mask = np.flipud(room_mask)
                mask_4class = np.flipud(mask_4class)
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                canvas_img = canvas_img.rotate(rot_k * 90)
                struct_mask = np.rot90(struct_mask, rot_k)
                room_mask = np.rot90(room_mask, rot_k)
                mask_4class = np.rot90(mask_4class, rot_k)

        img_tensor = TF.to_tensor(canvas_img)
        img_tensor = (img_tensor - self.mean) / self.std

        return {
            "image": img_tensor.float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_4class)).long(),
            "struct_mask": torch.from_numpy(np.ascontiguousarray(struct_mask)).long(),
            "room_mask": torch.from_numpy(np.ascontiguousarray(room_mask)).long(),
            "gt_scale": torch.tensor(0.035, dtype=torch.float32),
            "sample_id": sample_rel_path
        }
