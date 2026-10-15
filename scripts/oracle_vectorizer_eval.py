"""
Oracle Evaluation Suite for Floorplan Vectorizer.
Feeds GROUND-TRUTH masks into FloorplanVectorizer to establish the theoretical
performance ceiling and diagnose systematic metric artifacts (e.g. perimeter error).
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.vectorizer import FloorplanVectorizer
from shapely.geometry import Polygon

def run_oracle_evaluation(split="val", max_samples=400):
    dataset = CubiCasaDataset(split=split, img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    vectorizer = FloorplanVectorizer()

    print(f"\n[ORACLE EVALUATION] Running on {len(dataset)} plans from '{split}' split with GT masks...")

    perimeter_rel_errors_raw = []
    perimeter_rel_errors_outer = []
    room_count_signed_errors = []
    room_count_abs_errors = []
    valid_shell_count = 0
    total_plans = 0

    debug_samples = []

    for idx, batch in enumerate(loader):
        if idx >= max_samples:
            break

        gt_struct = batch["struct_mask"][0].numpy()  # 0: bg, 1: int_wall, 2: ext_wall, 3: door, 4: win
        gt_room = batch["room_mask"][0].numpy()      # 0: bg, 1..11: room classes
        gt_scale = float(batch["gt_scale"][0])

        # Prepare 4-class mask for vectorizer
        gt_m4 = np.zeros_like(gt_struct)
        gt_m4[(gt_struct == 1) | (gt_struct == 2)] = 1
        gt_m4[gt_struct == 3] = 2
        gt_m4[gt_struct == 4] = 3

        # Run vectorizer on PERFECT ground truth
        layout = vectorizer.vectorize(
            mask=gt_m4,
            struct_mask=gt_struct,
            room_mask=gt_room
        )

        meta = layout.get("metadata", {})
        pred_perim_m = meta.get("building_perimeter_m", 0.0)
        is_shell_valid = meta.get("is_building_shell_valid", False)
        if is_shell_valid:
            valid_shell_count += 1

        # 1. PERIMETER DIAGNOSIS:
        # A) Raw metric used before: cv2.findContours on (gt_struct == 2)
        gt_ext_mask = (gt_struct == 2).astype(np.uint8)
        cnts, _ = cv2.findContours(gt_ext_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # B) Outer boundary only: fill all internal holes of building envelope
        building_envelope = ((gt_struct == 1) | (gt_struct == 2) | (gt_room > 0)).astype(np.uint8)
        env_cnts, _ = cv2.findContours(building_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            raw_gt_perim_m = sum(cv2.arcLength(c, True) for c in cnts) * gt_scale
            if raw_gt_perim_m > 1.0:
                err_raw = abs(pred_perim_m - raw_gt_perim_m) / raw_gt_perim_m
                perimeter_rel_errors_raw.append(min(err_raw, 2.0))

        if env_cnts:
            # The true outer building footprint perimeter
            largest_env = max(env_cnts, key=cv2.contourArea)
            outer_gt_perim_m = cv2.arcLength(largest_env, True) * gt_scale
            if outer_gt_perim_m > 1.0:
                err_outer = abs(pred_perim_m - outer_gt_perim_m) / outer_gt_perim_m
                perimeter_rel_errors_outer.append(min(err_outer, 2.0))

        # 2. ROOM COUNT DIAGNOSIS on GT:
        # Ground truth room instances (contours >= 100px)
        gt_num_rooms = 0
        for c in range(1, 12):
            c_mask = (gt_room == c).astype(np.uint8)
            rcnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_num_rooms += sum(1 for cnt in rcnts if cv2.contourArea(cnt) >= 100)

        pred_num_rooms = len(layout.get("rooms", []))
        s_err = pred_num_rooms - gt_num_rooms
        room_count_signed_errors.append(s_err)
        room_count_abs_errors.append(abs(s_err))

        total_plans += 1
        if total_plans <= 5:
            debug_samples.append({
                "plan": idx,
                "pred_perim_m": round(pred_perim_m, 2),
                "raw_gt_perim_m": round(raw_gt_perim_m, 2) if cnts else 0,
                "outer_gt_perim_m": round(outer_gt_perim_m, 2) if env_cnts else 0,
                "gt_rooms": gt_num_rooms,
                "pred_rooms": pred_num_rooms
            })

    print("\n" + "=" * 70)
    print(f"        ORACLE RESULTS (PERFECT GT MASKS, N={total_plans})")
    print("=" * 70)
    print(f"Watertight Building Shell Rate on GT: {(valid_shell_count / total_plans) * 100:.2f}%")
    print(f"\n--- PERIMETER METRIC DIAGNOSIS ---")
    p_raw = np.array(perimeter_rel_errors_raw) * 100
    p_outer = np.array(perimeter_rel_errors_outer) * 100
    print(f"1. Evaluator's Raw Metric (cv2.findContours on wall_ext ribbon):")
    print(f"   Mean Error:   {p_raw.mean():.2f}% | Median Error: {np.median(p_raw):.2f}%")
    print(f"2. True Building Envelope Outer Perimeter (External Footprint):")
    print(f"   Mean Error:   {p_outer.mean():.2f}% | Median Error: {np.median(p_outer):.2f}%")

    print(f"\n--- ROOM COUNT CEILING ON GT ---")
    s_arr = np.array(room_count_signed_errors)
    a_arr = np.array(room_count_abs_errors)
    print(f"Room Count MAE on GT:  {a_arr.mean():.2f}")
    print(f"Room Count Signed Err: {s_arr.mean():+.2f}")
    print(f"Over-segmented plans:  {(s_arr > 0).mean() * 100:.1f}%")
    print(f"Exact match plans:     {(s_arr == 0).mean() * 100:.1f}%")
    print(f"Under-segmented plans: {(s_arr < 0).mean() * 100:.1f}%")

    print("\nSample Inspection:")
    for s in debug_samples:
        print(f"  Plan {s['plan']}: PredPerim={s['pred_perim_m']}m | RawGTPerim={s['raw_gt_perim_m']}m | OuterGTPerim={s['outer_gt_perim_m']}m | GTRooms={s['gt_rooms']} | PredRooms={s['pred_rooms']}")

if __name__ == "__main__":
    run_oracle_evaluation()
