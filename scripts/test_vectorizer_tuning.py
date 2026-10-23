"""
Benchmark Vectorizer Room Area Thresholds on Dev Plans (N=20)
Tests min_area_habitable_m2 in [1.5, 2.5, 3.5, 4.5] with Challenger and Champion
to measure the reduction of room over-splitting.
"""

import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline
from src.floorplan.vectorizer import FloorplanVectorizer

def test_tuning():
    manifest_path = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = FloorplanPipeline.get_instance()
    
    # Load Challenger to test over-splitting mitigation
    chal_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth"
    import torch
    ckpt = torch.load(chal_p, map_location=pipeline.device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    pipeline.multitask_model.load_state_dict(sd)
    pipeline.multitask_model.eval()

    thresholds = [1.5, 2.5, 3.5, 4.5]
    results = {th: {"total_rooms": 0, "plans": {}} for th in thresholds}

    for item in manifest:
        pid = item["id"]
        fname = item["filename"]
        img_p = manifest_path.parent / fname
        with open(img_p, "rb") as f:
            raw_bytes = f.read()

        for th in thresholds:
            # Set vectorizer threshold
            pipeline.vectorizer.min_area_habitable_m2 = th
            res = pipeline.process_image(raw_bytes, tta_mode="none", front_only=True)
            rooms = res.get("layout", {}).get("rooms", [])
            results[th]["total_rooms"] += len(rooms)
            results[th]["plans"][pid] = len(rooms)

    print("=" * 80)
    print("VECTORIZER MIN_AREA_HABITABLE_M2 TUNING ON CHALLENGER (DEV SET N=20)")
    print("=" * 80)
    print(f"{'Threshold (m2)':<20} | {'Total Rooms':<15} | {'Mean Rooms/Plan':<18} | {'dev_12 (was 21)':<15}")
    print("-" * 80)
    for th in thresholds:
        tot = results[th]["total_rooms"]
        mean_r = tot / len(manifest)
        d12 = results[th]["plans"]["dev_12"]
        print(f"{th:<20} | {tot:<15} | {mean_r:<18.2f} | {d12:<15}")
    print("=" * 80)

if __name__ == "__main__":
    test_tuning()
