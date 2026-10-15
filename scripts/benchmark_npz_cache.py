import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from src.floorplan.dataset import CubiCasaDataset

cache_dir = Path("data/CubiCasa5k/cached_test")
cache_dir.mkdir(parents=True, exist_ok=True)

ds = CubiCasaDataset(split="train", img_size=512, augment=False)
print("Benchmarking rasterize & cache for 10 samples...")

t0 = time.time()
for i in range(10):
    s = ds[i]
    sample_id = s["sample_id"].replace("/", "_").strip("_")
    out_file = cache_dir / f"{sample_id}.npz"
    
    # Save as uint8
    img_np = ((s["image"].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0).clip(0, 255).astype(np.uint8)
    struct_mask_np = s["struct_mask"].numpy().astype(np.uint8)
    room_mask_np = s["room_mask"].numpy().astype(np.uint8)
    mask_4class_np = s["mask"].numpy().astype(np.uint8)
    
    np.savez_compressed(
        out_file,
        image=img_np,
        struct_mask=struct_mask_np,
        room_mask=room_mask_np,
        mask=mask_4class_np
    )

t_save = time.time() - t0
print(f"Saved 10 samples in {t_save:.2f}s ({t_save/10:.3f}s per sample)")

# Check size
sizes = [f.stat().st_size for f in cache_dir.glob("*.npz")]
print(f"Mean size: {np.mean(sizes)/1024:.1f} KB per sample")

# Benchmark loading speed
t1 = time.time()
for f in cache_dir.glob("*.npz"):
    data = np.load(f)
    img = data["image"]
    sm = data["struct_mask"]
    rm = data["room_mask"]
t_load = time.time() - t1
print(f"Loaded 10 samples in {t_load*1000:.2f} ms ({t_load*1000/10:.2f} ms per sample)")

# Clean up test cache
for f in cache_dir.glob("*.npz"):
    f.unlink()
cache_dir.rmdir()
