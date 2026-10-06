"""
PyTorch Dataset and Data Augmentation Pipeline for ArchiGen 3D.
Loads paired architectural images (Layout/Sketch and Real Photo)
for Pix2Pix conditional GAN training.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FacadesDataset(Dataset):
    """
    Dataset class for paired Facades images.
    In the CMP Facades dataset, each image is 512x256 where:
    - Left half (256x256) is the real building photo
    - Right half (256x256) is the architectural semantic layout/sketch (or vice versa)
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.image_files = sorted(list(self.split_dir.glob("*.jpg")))
        self.img_size = img_size
        self.split = split

        # Base transform: resize to target resolution and normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

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

        # Data augmentation on train split (preventing data leakage on val/test)
        if self.split == "train":
            # Horizontal random flip
            if torch.rand(1).item() > 0.5:
                photo = transforms.functional.hflip(photo)
                sketch = transforms.functional.hflip(sketch)

        photo_tensor = self.transform(photo)
        sketch_tensor = self.transform(sketch)

        return {
            "sketch": sketch_tensor,   # Conditioning input (Layout)
            "photo": photo_tensor,     # Real target image
            "filename": img_path.name
        }
