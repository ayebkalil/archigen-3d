"""
Pre-rasterizes CubiCasa5k SVGs into compressed NPZ cache files.
Eliminates XML parsing overhead during training, providing ~10x speedup per epoch.
Outputs:
  data/CubiCasa5k/cached_512/{split}/{sample_id}.npz
Contains:
  - image: uint8 (512, 512, 3) letterbox padded
  - struct_mask: uint8 (512, 512) classes [0..4]
  - room_mask: uint8 (512, 512) classes [0..11]
  - mask: uint8 (512, 512) legacy 4-class [0..3]
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset

def cache_split(split: str = "train", max_samples: int = None, img_size: int = 512):
    cache_root = Path("data/CubiCasa5k/cached_512") / split
    cache_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[CACHE BUILDER] Processing split '{split}'...")
    dataset = CubiCasaDataset(split=split, img_size=img_size, augment=False)
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    start_time = time.time()
    saved_count = 0
    skipped_count = 0
    
    for idx in range(total):
        sample_rel = dataset.valid_samples[idx]
        file_key = sample_rel.replace("/", "_").strip("_")
        out_path = cache_root / f"{file_key}.npz"
        
        if out_path.exists():
            skipped_count += 1
            continue
            
        try:
            s = dataset[idx]
            # Convert normalized float image back to uint8 [0..255]
            img_tensor = s["image"] # (3, 512, 512)
            mean = dataset.mean.view(3, 1, 1)
            std = dataset.std.view(3, 1, 1)
            img_unnorm = (img_tensor * std + mean) * 255.0
            img_np = img_unnorm.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
            
            struct_np = s["struct_mask"].numpy().astype(np.uint8)
            room_np = s["room_mask"].numpy().astype(np.uint8)
            mask4_np = s["mask"].numpy().astype(np.uint8)
            
            np.savez_compressed(
                out_path,
                image=img_np,
                struct_mask=struct_np,
                room_mask=room_np,
                mask=mask4_np
            )
            saved_count += 1
        except Exception as e:
            print(f"Error caching {sample_rel}: {e}")
            
        if (idx + 1) % 200 == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            eta = (total - (idx + 1)) / max(rate, 0.001)
            print(f"[{split.upper()}] {idx+1}/{total} ({rate:.1f} plans/sec | ETA: {eta/60:.1f}m) | Saved: {saved_count}, Skipped: {skipped_count}")
            
    total_time = time.time() - start_time
    print(f"[CACHE BUILDER] Finished '{split}' in {total_time/60:.2f}m. Total cached: {saved_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    if args.split == "all":
        cache_split("val", args.max_samples)
        cache_split("test", args.max_samples)
        cache_split("train", args.max_samples)
    else:
        cache_split(args.split, args.max_samples)
