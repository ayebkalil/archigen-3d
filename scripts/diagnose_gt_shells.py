"""
Diagnose the 38 plans where is_building_shell_valid == False on perfect Ground-Truth masks.
Inspects the failure reasons:
1. Is it a self-touching / non-valid Shapely polygon?
2. Is it disconnected building components (e.g. detached garage/shed)?
3. Is it an internal courtyard cutout?
4. Is it a strict rule in vectorizer (e.g. wall count, room count)?
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.vectorizer import FloorplanVectorizer

def diagnose_failing_shells():
    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    vectorizer = FloorplanVectorizer()

    failed_plans = []

    for idx, batch in enumerate(loader):
        gt_struct = batch["struct_mask"][0].numpy()
        gt_room = batch["room_mask"][0].numpy()
        sample_id = batch["sample_id"][0]

        gt_m4 = np.zeros_like(gt_struct)
        gt_m4[(gt_struct == 1) | (gt_struct == 2)] = 1
        gt_m4[gt_struct == 3] = 2
        gt_m4[gt_struct == 4] = 3

        layout = vectorizer.vectorize(mask=gt_m4, struct_mask=gt_struct, room_mask=gt_room)
        meta = layout.get("metadata", {})
        
        if not meta.get("is_building_shell_valid", False):
            # Inspect the check conditions in vectorizer.py
            walls = layout.get("walls", [])
            rooms = layout.get("rooms", [])
            ext_walls = [w for w in walls if w.get("is_exterior")]
            
            # Check building envelope contour from vectorizer
            wall_clean = (gt_m4 == 1).astype(np.uint8)
            cnts, _ = cv2.findContours(wall_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            reason = "Unknown"
            if len(rooms) == 0:
                reason = "No rooms detected"
            elif len(walls) < 4:
                reason = f"Too few walls ({len(walls)})"
            elif len(ext_walls) < 3:
                reason = f"Too few exterior walls ({len(ext_walls)})"
            elif len(cnts) == 0:
                reason = "No wall contours"
            else:
                largest_c = max(cnts, key=cv2.contourArea)
                pts = largest_c.reshape(-1, 2)
                if len(pts) < 3:
                    reason = f"Contour points < 3 ({len(pts)})"
                else:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        reason = f"Self-intersecting envelope: {poly.is_valid}"
                    else:
                        area = cv2.contourArea(largest_c)
                        if area < 1000:
                            reason = f"Envelope area too small ({area})"
                        else:
                            reason = f"Envelope valid but shell flag failed (status: {meta.get('envelope_status')})"

            failed_plans.append({
                "plan_idx": idx,
                "sample_id": sample_id,
                "reason": reason,
                "envelope_status": meta.get("envelope_status"),
                "num_walls": len(walls),
                "num_ext_walls": len(ext_walls),
                "num_rooms": len(rooms)
            })

            if len(failed_plans) <= 10:
                print(f"[FAIL #{len(failed_plans)}] Plan {idx} ({sample_id}): {reason} | Walls: {len(walls)} ({len(ext_walls)} ext) | Rooms: {len(rooms)} | Status: {meta.get('envelope_status')}")

    print("\n" + "="*70)
    print(f"Total failing shells on perfect GT: {len(failed_plans)} / {len(dataset)} ({len(failed_plans)/len(dataset)*100:.1f}%)")
    from collections import Counter
    reasons_count = Counter([f["reason"] for f in failed_plans])
    print("\nBreakdown of Failure Reasons:")
    for r, cnt in reasons_count.most_common():
        print(f"  - {r}: {cnt} plans")

if __name__ == "__main__":
    diagnose_failing_shells()
