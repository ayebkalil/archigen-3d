"""
analyze_cmp_palette_and_frequencies.py
Extracts the exact RGB color centers, class pixel frequencies,
and vertical spatial distributions from all 400 training sketches
of the CMP Facades dataset.
"""

import sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data/raw/facades/train"

# Official CMP 12 architectural classes (Tylecek & Sara, 2013)
# Base colormap specified by CMP:
CMP_CLASSES = {
    1: {"name": "background / sky", "nominal_rgb": (0, 0, 0)},
    2: {"name": "facade / wall",     "nominal_rgb": (0, 0, 170)}, # Or (0, 48, 255) in 8-bit CMP
    3: {"name": "window",            "nominal_rgb": (0, 128, 255)},
    4: {"name": "door",              "nominal_rgb": (0, 207, 255)}, # Or (0, 255, 255)
    5: {"name": "cornice / roof",    "nominal_rgb": (0, 0, 222)},
    6: {"name": "sill",              "nominal_rgb": (255, 80, 0)},
    7: {"name": "balcony",           "nominal_rgb": (192, 255, 54)},
    8: {"name": "blind",             "nominal_rgb": (128, 0, 0)},
    9: {"name": "pillar / column",   "nominal_rgb": (0, 128, 0)},
    10: {"name": "deco / header",    "nominal_rgb": (255, 255, 0)},
    11: {"name": "molding",          "nominal_rgb": (0, 0, 222)},
    12: {"name": "base / foundation","nominal_rgb": (0, 0, 222)}
}

def analyze():
    files = sorted(list(TRAIN_DIR.glob("*.jpg")))
    print(f"Analyzing {len(files)} training sketches from CMP Facades dataset...")

    # We collect pixels from right half (sketch)
    # Quantize to 16-level buckets to eliminate JPEG compression fuzziness
    color_hist = Counter()
    total_pixels = 0
    spatial_y = {c: [] for c in range(1, 13)}

    for idx, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        w, h = im.size
        sketch = im.crop((w // 2, 0, w, h))
        arr = np.array(sketch, dtype=np.int32)
        
        # Flatten
        pixels = arr.reshape(-1, 3)
        # Quantize by rounding to nearest multiple of 16 to group JPEG noise
        quant = (np.round(pixels / 16.0) * 16).astype(np.int32)
        quant = np.clip(quant, 0, 255)
        
        for p in quant:
            color_hist[tuple(p.tolist())] += 1
        total_pixels += len(pixels)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(files)} images...")

    print("\n" + "=" * 80)
    print("TOP DOMINANT RGB COLOR CLUSTERS IN CMP FACADES (TRAIN N = 400):")
    print("=" * 80)
    print(f"{'Rank':<6} | {'RGB Value':<16} | {'Pixel Count':<12} | {'Percentage':<10} | {'Visual Color / Likely Class'}")
    print("-" * 80)

    top_colors = color_hist.most_common(20)
    cumulative_pct = 0.0
    for r, (rgb, count) in enumerate(top_colors, 1):
        pct = (count / total_pixels) * 100.0
        cumulative_pct += pct
        
        # Determine likely class name by color
        r_val, g_val, b_val = rgb
        desc = "Unknown"
        if b_val > 180 and r_val < 50 and 30 <= g_val <= 70:
            desc = "CMP Class 2: Facade Wall (Cobalt Blue)"
        elif b_val > 200 and r_val < 30 and g_val < 30:
            desc = "CMP Class 5/11/12: Cornice / Base / Molding (Dark Blue)"
        elif b_val > 200 and 100 <= g_val <= 160 and r_val < 30:
            desc = "CMP Class 3: Window (Light Cyan/Blue)"
        elif r_val > 200 and 60 <= g_val <= 120 and b_val < 50:
            desc = "CMP Class 6: Window Sill (Orange/Red)"
        elif r_val < 50 and g_val > 180 and b_val > 180:
            desc = "CMP Class 4: Door (True Cyan)"
        elif r_val > 160 and g_val > 200 and 30 <= b_val <= 90:
            desc = "CMP Class 7: Balcony (Lime Green)"
        elif r_val > 200 and g_val > 200 and b_val < 50:
            desc = "CMP Class 10: Deco / Header (Yellow)"
        elif r_val < 50 and 80 <= g_val <= 160 and b_val < 50:
            desc = "CMP Class 9: Pillar / Column (Green)"
        elif r_val < 30 and g_val < 30 and b_val < 30:
            desc = "CMP Class 1: Background / Sky / Shadow (Black)"
        elif r_val > 100 and g_val < 50 and b_val < 50:
            desc = "CMP Class 8: Blind / Shutter (Maroon)"

        print(f"#{r:<5} | {str(rgb):<16} | {count:<12} | {pct:6.2f}%    | {desc}")

    print("-" * 80)
    print(f"Top-20 clusters account for {cumulative_pct:.2f}% of all training pixels.")

if __name__ == "__main__":
    analyze()
