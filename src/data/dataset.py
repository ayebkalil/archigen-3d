"""
PyTorch Dataset and Data Augmentation Pipeline for ArchiGen 3D.
Loads paired architectural images (Layout/Sketch and Real Photo)
for Pix2Pix conditional GAN training.
"""

import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class FacadesDataset(Dataset):
    """
    Dataset class for paired Facades images with Advanced Data Augmentation.
    Features:
    1. Scale jittering: Resize to 286x286 then synchronized Random Crop to 256x256.
    2. Synchronized Random Horizontal Flip.
    3. Strict isolation: Augmentation applies ONLY to training split (prevents Data Leakage).
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256, augment: bool = True, color_jitter: bool = False):
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.image_files = sorted(list(self.split_dir.glob("*.jpg")))
        self.img_size = img_size
        self.split = split
        self.augment = augment and (split == "train")
        self.color_jitter = color_jitter

        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        combined_img = Image.open(img_path).convert("RGB")
        w, h = combined_img.size

        # Split horizontally: standard pix2pix format is (photo, sketch)
        split_point = w // 2
        photo = combined_img.crop((0, 0, split_point, h))
        sketch = combined_img.crop((split_point, 0, w, h))

        if self.augment:
            # 1. Scale Jitter: dynamically scale between 1.05x and 1.20x (268px to 308px)
            scale_factor = random.uniform(1.05, 1.20)
            jitter_size = int(self.img_size * scale_factor)
            photo = photo.resize((jitter_size, jitter_size), Image.Resampling.BICUBIC)
            sketch = sketch.resize((jitter_size, jitter_size), Image.Resampling.NEAREST)

            # 2. Synchronized Random Crop back to 256x256 (identical coordinates for sketch & photo)
            top = random.randint(0, jitter_size - self.img_size)
            left = random.randint(0, jitter_size - self.img_size)
            photo = photo.crop((left, top, left + self.img_size, top + self.img_size))
            sketch = sketch.crop((left, top, left + self.img_size, top + self.img_size))

            # 3. Synchronized Horizontal Random Flip (p=0.5)
            if random.random() > 0.5:
                photo = TF.hflip(photo)
                sketch = TF.hflip(sketch)

            # 4. Subtle Color Jitter on target photo only (optional, disabled by default to prevent blur)
            if self.color_jitter and random.random() > 0.5:
                photo = TF.adjust_brightness(photo, brightness_factor=random.uniform(0.9, 1.1))
                photo = TF.adjust_contrast(photo, contrast_factor=random.uniform(0.9, 1.1))
        else:
            # Deterministic resize for val/test
            photo = photo.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            sketch = sketch.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        photo_tensor = self.normalize(TF.to_tensor(photo))
        sketch_tensor = self.normalize(TF.to_tensor(sketch))

        return {
            "sketch": sketch_tensor,   # Conditioning input (Layout)
            "photo": photo_tensor,     # Real target image
            "filename": img_path.name
        }
