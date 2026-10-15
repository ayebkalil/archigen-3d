"""
Geometry-Preserving CAD Image Augmentation Pipeline for Floorplans.
Applies:
1. Random Grayscale (p=0.2)
2. Inverted Blueprint Mode: Dark background with light lines (p=0.15)
3. Gaussian Blur (sigma 0.5-1.5, p=0.3)
4. JPEG Compression Artifacts (quality 35-75, p=0.3)
5. Downscale-and-Upscale Resolution Degrade (scale 0.5-0.8, p=0.3)
6. Contrast and Brightness Jitter (p=0.4)
7. Minimal 1px Line Thinning/Thickening (morphological kernel 3x3, max 1px, p=0.25)
All augmentations modify ONLY the image pixel intensities; masks remain 100% untouched.
Saves a 4x4 visual verification grid (models/samples/aug_verification_16.png)
with ground-truth structural contours overlaid to verify perfect spatial alignment.
"""

import sys
import io
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset

class CADImageAugmentor:
    """
    Geometry-preserving augmentor that attacks domain shift and thin-line CAD failures
    without introducing boundary label noise.
    """
    def __init__(self, p_total: float = 0.5):
        self.p_total = p_total

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        if random.random() > self.p_total:
            return pil_img

        img = pil_img.copy()

        # 1. Blueprint Color Inversion (white on black CAD drawing)
        if random.random() < 0.20:
            arr = np.array(img)
            # Invert RGB channels
            arr = 255 - arr
            img = Image.fromarray(arr)

        # 2. Grayscale conversion (monochrome architectural scans)
        if random.random() < 0.30:
            img = img.convert("L").convert("RGB")

        # 3. Contrast & Brightness Jitter
        if random.random() < 0.40:
            c_factor = random.uniform(0.70, 1.35)
            img = ImageEnhance.Contrast(img).enhance(c_factor)
            b_factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(b_factor)

        # 4. Downscale then Upscale (simulates low-res raster scans)
        if random.random() < 0.35:
            w, h = img.size
            scale = random.uniform(0.40, 0.75)
            dw, dh = max(int(w * scale), 64), max(int(h * scale), 64)
            img = img.resize((dw, dh), Image.Resampling.BILINEAR)
            img = img.resize((w, h), Image.Resampling.NEAREST if random.random() < 0.5 else Image.Resampling.BILINEAR)

        # 5. Gaussian Blur
        if random.random() < 0.30:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 6. JPEG Compression Artifacts
        if random.random() < 0.30:
            quality = random.randint(30, 75)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # 7. Sub-pixel 1px Line Thinning / Thickening (strictly <= 1px)
        if random.random() < 0.25:
            arr = np.array(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            if random.random() < 0.5:
                # Erode dark lines (thins black lines on white paper)
                arr = cv2.dilate(arr, kernel, iterations=1)
            else:
                # Dilate dark lines (thickens black lines on white paper)
                arr = cv2.erode(arr, kernel, iterations=1)
            img = Image.fromarray(arr)

        return img


def generate_verification_grid(num_samples: int = 16, output_path: str = "models/samples/aug_verification_16.png"):
    dataset = CubiCasaDataset(split="train", img_size=512, augment=False)
    augmentor = CADImageAugmentor(p_total=1.0) # Force augmentation for all 16 inspection samples

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("ArchiGen 3D — 16 Augmented CAD Samples with GT Mask Contour Overlays\n(Red=Walls, Green=Doors, Cyan=Windows)", fontsize=14, fontweight="bold", y=0.99)

    # Pick 16 diverse samples across styles
    indices = [i * (len(dataset) // num_samples) for i in range(num_samples)]

    for idx, (ax, data_idx) in enumerate(zip(axes.flat, indices)):
        item = dataset[data_idx]
        sample_id = item["sample_id"]
        
        # Un-normalize image
        img_tensor = item["image"] # (3, 512, 512)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        raw_tensor = (img_tensor * std + mean).clamp(0, 1)
        raw_np = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Apply CAD augmentation
        pil_img = Image.fromarray(raw_np)
        aug_pil = augmentor(pil_img)
        aug_np = np.array(aug_pil)

        # Overlay Ground-Truth Contours:
        # Walls (ext/int) in Red, Doors in Green, Windows in Cyan
        struct_mask = item["struct_mask"].numpy()
        
        overlay = aug_np.copy()
        
        # Red contours for walls (classes 1 and 2)
        wall_mask = ((struct_mask == 1) | (struct_mask == 2)).astype(np.uint8)
        cnts_w, _ = cv2.findContours(wall_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts_w, -1, (255, 30, 30), 1)

        # Green contours for doors (class 3)
        door_mask = (struct_mask == 3).astype(np.uint8)
        cnts_d, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts_d, -1, (30, 255, 30), 2)

        # Cyan contours for windows (class 4)
        win_mask = (struct_mask == 4).astype(np.uint8)
        cnts_win, _ = cv2.findContours(win_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts_win, -1, (0, 230, 255), 2)

        ax.imshow(overlay)
        style_label = sample_id.split("/")[0] if "/" in sample_id else sample_id.split("\\")[0]
        ax.set_title(f"#{idx+1}: {style_label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved 16-sample augmentation verification grid to {output_path}")

if __name__ == "__main__":
    generate_verification_grid()
