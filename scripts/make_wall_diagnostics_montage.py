"""
scripts/make_wall_diagnostics_montage.py
----------------------------------------
Combines the 5 diagnostic overlays into a clean multi-panel report image.
"""

import os
from PIL import Image

def make_montage():
    overlay_dir = "outputs/wall_offset_diagnostics"
    plans = [os.path.join(overlay_dir, f"plan_{i}_overlay.png") for i in range(5)]
    
    images = [Image.open(p) for p in plans]
    w, h = images[0].size
    
    # Create vertical stack of 5 plans or 2x3 grid
    # Let's do 5 vertical rows: each row is 1400 x 700
    # Or stack them vertically
    total_h = h * len(images)
    montage = Image.new("RGB", (w, total_h), (255, 255, 255))
    
    for i, im in enumerate(images):
        montage.paste(im, (0, i * h))
        
    out_path = "outputs/wall_offset_diagnostics/wall_offsets_5plans_montage.png"
    montage.save(out_path)
    print("Saved montage to:", out_path)

if __name__ == "__main__":
    make_montage()
