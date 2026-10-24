"""
Room Threshold Calibration on CubiCasa5K Validation Split (N=400)
Calibrates min_area_habitable_m2 against ground-truth room counts.
Evaluates MAE, Signed Error, and Shell Validity across thresholds in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5].
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def count_gt_rooms(gt_room_mask):
    total = 0
    for c in range(1, 12):
        c_bin = (gt_room_mask == c).astype(np.uint8)
        num_c, _ = cv2.connectedComponents(c_bin)
        total += max(0, num_c - 1)
    return total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CALIBRATE] Running on device: {device}")

    ds = CubiCasaDataset(
        root_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cubicasa5k/cubicasa5k"),
        split="val",
        img_size=512,
        augment=False,
        cache_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cached_512")
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    model_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    ckpt = torch.load(model_p, map_location=device, weights_only=False)
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Pre-compute model predictions and GT room counts for all 400 plans
    print("[CALIBRATE] Pre-computing predictions for 400 validation floorplans...")
    val_records = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            out = model(imgs)
            pred_s = torch.argmax(out["struct"], dim=1).cpu().numpy().astype(np.uint8)
            pred_r = torch.argmax(out["room"], dim=1).cpu().numpy().astype(np.uint8)
            gt_r = batch["room_mask"].numpy().astype(np.uint8)

            for i in range(len(imgs)):
                gt_count = count_gt_rooms(gt_r[i])
                s = pred_s[i]
                r = pred_r[i]
                m4 = np.zeros_like(s)
                m4[(s == 1) | (s == 2)] = 1
                m4[s == 3] = 2
                m4[s == 4] = 3
                val_records.append({
                    "m4": m4,
                    "pred_s": s,
                    "pred_r": r,
                    "gt_rooms": gt_count
                })

    print(f"[CALIBRATE] Stored {len(val_records)} validation samples. Sweeping thresholds...")

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results = {}

    for th in thresholds:
        vec = FloorplanVectorizer(min_area_habitable_m2=th)
        abs_errors = []
        signed_errors = []
        valid_shells = 0

        for rec in val_records:
            layout = vec.vectorize(
                mask=rec["m4"],
                struct_mask=rec["pred_s"],
                room_mask=rec["pred_r"]
            )
            n_pred = len(layout.get("rooms", []))
            n_gt = rec["gt_rooms"]
            diff = n_pred - n_gt
            abs_errors.append(abs(diff))
            signed_errors.append(diff)
            if layout.get("metadata", {}).get("is_building_shell_valid", False):
                valid_shells += 1

        mae = float(np.mean(abs_errors))
        bias = float(np.mean(signed_errors))
        shell_rate = float(valid_shells / len(val_records) * 100.0)

        results[th] = {
            "mae": round(mae, 3),
            "bias": round(bias, 3),
            "shell_rate": round(shell_rate, 2)
        }
        print(f"  Threshold {th:4.1f} m2 -> Room MAE: {mae:5.2f} | Signed Bias: {bias:+5.2f} rooms | Shell Valid: {shell_rate:5.1f}%")

    out_file = PROJECT_ROOT / "outputs" / "room_threshold_calibration_val.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    best_th = min(results.keys(), key=lambda t: results[t]["mae"])
    print(f"\n[CALIBRATE] Optimal validation threshold on CubiCasa5K: {best_th} m2 (MAE = {results[best_th]['mae']})")

if __name__ == "__main__":
    main()
