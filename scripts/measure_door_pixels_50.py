import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.floorplan.dataset import CubiCasaDataset
import numpy as np

ds = CubiCasaDataset(split="train", img_size=512, augment=False)
door_pixels = []
win_pixels = []
wall_pixels = []

print("Measuring pixel counts across 50 samples...")
for i in range(50):
    s = ds[i]
    sm = s["struct_mask"].numpy()
    w_px = int((sm == 1).sum() + (sm == 2).sum())
    d_px = int((sm == 3).sum())
    win_px = int((sm == 4).sum())
    wall_pixels.append(w_px)
    door_pixels.append(d_px)
    win_pixels.append(win_px)

door_pixels = np.array(door_pixels)
win_pixels = np.array(win_pixels)
wall_pixels = np.array(wall_pixels)

print(f"50 Floorplans Pixel Statistics (at 512x512 = 262,144 px):")
print(f"  Walls:   mean = {wall_pixels.mean():.1f} px ({wall_pixels.mean()/262144*100:.2f}%), min = {wall_pixels.min()}, max = {wall_pixels.max()}")
print(f"  Windows: mean = {win_pixels.mean():.1f} px ({win_pixels.mean()/262144*100:.2f}%), min = {win_pixels.min()}, max = {win_pixels.max()}")
print(f"  Doors:   mean = {door_pixels.mean():.1f} px ({door_pixels.mean()/262144*100:.2f}%), min = {door_pixels.min()}, max = {door_pixels.max()}")
print(f"  Samples with 0 door pixels: {np.count_nonzero(door_pixels == 0)} / 50")
print(f"  Samples with 0 window pixels: {np.count_nonzero(win_pixels == 0)} / 50")

for i in range(15):
    print(f"Sample {i:02d}: Wall={wall_pixels[i]:5d} | Window={win_pixels[i]:4d} | Door={door_pixels[i]:4d}")
