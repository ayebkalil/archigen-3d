import sys
from pathlib import Path
import types

# Create mock floortrans package
sys.path.append('scripts')
import cubicasa_svg_utils as csu
floortrans = types.ModuleType("floortrans")
floortrans_loaders = types.ModuleType("floortrans.loaders")
floortrans_loaders.svg_utils = csu
floortrans.loaders = floortrans_loaders
sys.modules["floortrans"] = floortrans
sys.modules["floortrans.loaders"] = floortrans_loaders
sys.modules["floortrans.loaders.svg_utils"] = csu

import cubicasa_house as ch
import cv2
import numpy as np

sample_path = Path('data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/6044')
img_p = sample_path / 'F1_scaled.png'
svg_p = sample_path / 'model.svg'

im = cv2.imread(str(img_p))
h, w = im.shape[:2]
print(f"Image F1_scaled shape: H={h}, W={w}")

try:
    house = ch.House(str(svg_p), h, w)
    print("House created successfully!")
    print("Walls shape:", house.walls.shape)
    print("Non-zero wall pixels:", np.count_nonzero(house.walls))
    print("Non-zero icon pixels:", np.count_nonzero(house.icons))
    
    # Save an overlay image to see if it matches!
    overlay = im.copy()
    overlay[house.walls > 0] = [0, 0, 255] # Red walls
    cv2.imwrite("scripts/debug_house_6044.png", overlay)
    print("Saved overlay to scripts/debug_house_6044.png")
except Exception as e:
    import traceback
    traceback.print_exc()
