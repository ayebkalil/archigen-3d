"""
Script: verify_unsharp_mask_grid.py
Generates a side-by-side inspection grid comparing raw cGAN synthesized facades
with the contrast-adaptive unsharp masked post-processed outputs.
Saves difference maps to verify that the unsharp filter sharpens architectural lines
(window mullions, lintels, cornices) without inducing high-frequency overshoot or ringing halos.
"""

import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.architecture import UNetGenerator
from src.bridge.facade_bridge import enhance_facade_details

def create_unsharp_inspection_grid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Perceptual Generator
    weights_path = PROJECT_ROOT / "models/saved/generator_perceptual_best.pth"
    generator = UNetGenerator(in_channels=3, out_channels=3, num_filters=64, upsample_mode="nearest")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt["generator_state_dict"] if isinstance(ckpt, dict) and "generator_state_dict" in ckpt else ckpt
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()

    # Pick 4 test CMP facades
    test_dir = PROJECT_ROOT / "data/raw/facades/test"
    sample_ids = [1, 10, 25, 42]
    
    transform_in = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    results = []

    for sid in sample_ids:
        img_path = test_dir / f"{sid}.jpg"
        if not img_path.exists():
            continue
        
        full_img = Image.open(img_path).convert("RGB")
        w, h = full_img.size
        # Left half is photo, right half is CMP semantic label
        label_crop = full_img.crop((w // 2, 0, w, h))
        
        tensor_in = transform_in(label_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            tensor_out = generator(tensor_in)
        
        # De-normalize [-1, 1] -> [0, 255]
        arr_out = (tensor_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        raw_pil = Image.fromarray((arr_out * 255).astype(np.uint8))

        # Apply adaptive unsharp mask post-processing
        enhanced_pil = enhance_facade_details(raw_pil, radius=1.2, percent=125, threshold=3, contrast_factor=1.04)

        # Difference map amplified by 5x to check for ringing
        arr_raw = np.array(raw_pil, dtype=np.float32)
        arr_enh = np.array(enhanced_pil, dtype=np.float32)
        diff = np.abs(arr_enh - arr_raw) * 5.0
        diff_pil = Image.fromarray(diff.clip(0, 255).astype(np.uint8))

        results.append((label_crop.resize((256, 256)), raw_pil, enhanced_pil, diff_pil))

    # Compose 4x4 Grid
    # Columns: [Semantic Sketch, Raw cGAN Synthesis, Enhanced (Unsharp+Pop), Difference Map (5x)]
    cols = 4
    rows = len(results)
    grid_w = cols * 256 + 50
    grid_h = rows * 256 + 80
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid)

    headers = [
        "1. Input Semantic Label",
        "2. Raw cGAN Output",
        "3. Enhanced (+Unsharp/Contrast)",
        "4. Diff Map (5x Amplified)"
    ]

    for c, h_text in enumerate(headers):
        draw.text((25 + c * 256 + 10, 15), h_text, fill=(30, 41, 59))

    for r, (lbl, raw, enh, diff) in enumerate(results):
        y = 50 + r * 256
        grid.paste(lbl, (25, y))
        grid.paste(raw, (25 + 256, y))
        grid.paste(enh, (25 + 512, y))
        grid.paste(diff, (25 + 768, y))

    out_path = PROJECT_ROOT / "outputs/unsharp_mask_inspection_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Inspection grid saved successfully to: {out_path}")

if __name__ == "__main__":
    create_unsharp_inspection_grid()
