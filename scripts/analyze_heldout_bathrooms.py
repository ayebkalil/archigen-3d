"""
scripts/analyze_heldout_bathrooms.py
------------------------------------
Detailed investigation of failed bathrooms across the 100 held-out validation plans (plans 100 to 200):
- Exact share of failed bathrooms < 2.0 m2 and < 3.0 m2 (UNet++ over-segmentation fragments)
- Detailed breakdown by failure reason (TOO_SMALL, NO_USABLE_WALL, DOOR_BLOCKED, etc.)
- Distribution of failed room areas
"""

import os
import sys
import glob
import json
import io
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline

def analyze_heldout_bathrooms():
    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    heldout_files = all_val[100:200]
    print(f"Loaded {len(heldout_files)} heldout validation plans (plans 100 to 200).")

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    all_bathrooms = []
    failed_bathrooms = []
    valid_bathrooms = []

    for fidx, fpath in enumerate(heldout_files):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        pred_layout = proc["layout"]
        scale_m = pred_layout["metadata"].get("scale_meters_per_pixel", 0.03)
        raw_walls = pred_layout.get("walls", [])
        doors = pred_layout.get("doors", [])
        windows = pred_layout.get("windows", [])
        pred_wall_mask = proc.get("wall_mask")

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if "bath" not in norm_type and "wc" not in norm_type:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            area_m2 = poly.area

            # Filter matching benchmark threshold
            if area_m2 < 1.0:
                continue

            room_uid = f"heldout_{fname}_{r_idx}_bath"

            res = engine.design_room(
                room_polygon_meters=pts,
                room_type="bathroom",
                room_id=room_uid,
                wall_mask=pred_wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                seed=42
            )

            record = {
                "plan": fname,
                "room_idx": r_idx,
                "area_m2": round(area_m2, 3),
                "is_valid": res.is_valid,
                "failure_reason": res.failure_reason,
                "items_count": len(res.items)
            }
            all_bathrooms.append(record)

            if res.is_valid:
                valid_bathrooms.append(record)
            else:
                failed_bathrooms.append(record)

    total_baths = len(all_bathrooms)
    n_valid = len(valid_bathrooms)
    n_failed = len(failed_bathrooms)

    print("\n" + "=" * 70)
    print("FAILED BATHROOMS ANALYSIS (HELDOUT PLANS 100-200)")
    print("=" * 70)
    print(f"Total evaluated bathrooms: {total_baths}")
    print(f"Valid bathrooms: {n_valid} ({n_valid/total_baths*100:.2f}%)")
    print(f"Failed bathrooms: {n_failed} ({n_failed/total_baths*100:.2f}%)")

    # Area distribution of failed bathrooms
    failed_areas = [b["area_m2"] for b in failed_bathrooms]
    failed_under_2 = [b for b in failed_bathrooms if b["area_m2"] < 2.0]
    failed_under_3 = [b for b in failed_bathrooms if b["area_m2"] < 3.0]
    failed_over_3 = [b for b in failed_bathrooms if b["area_m2"] >= 3.0]

    print(f"\n--- FAILED BATHROOM AREA DISTRIBUTION ---")
    print(f"Failed < 2.0 m²: {len(failed_under_2)} / {n_failed} ({len(failed_under_2)/n_failed*100:.1f}% of failed, {len(failed_under_2)/total_baths*100:.1f}% of total)")
    print(f"Failed < 3.0 m²: {len(failed_under_3)} / {n_failed} ({len(failed_under_3)/n_failed*100:.1f}% of failed, {len(failed_under_3)/total_baths*100:.1f}% of total)")
    print(f"Failed >= 3.0 m²: {len(failed_over_3)} / {n_failed} ({len(failed_over_3)/n_failed*100:.1f}% of failed, {len(failed_over_3)/total_baths*100:.1f}% of total)")

    if failed_areas:
        print(f"Failed area min:    {min(failed_areas):.2f} m²")
        print(f"Failed area 25%:    {np.percentile(failed_areas, 25):.2f} m²")
        print(f"Failed area median: {np.median(failed_areas):.2f} m²")
        print(f"Failed area 75%:    {np.percentile(failed_areas, 75):.2f} m²")
        print(f"Failed area max:    {max(failed_areas):.2f} m²")

    # Breakdown by failure reason
    reason_counts = {}
    for b in failed_bathrooms:
        r = b["failure_reason"] or "UNKNOWN"
        reason_counts[r] = reason_counts.get(r, 0) + 1

    print(f"\n--- BREAKDOWN BY FAILURE REASON ---")
    for r, cnt in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r:25s}: {cnt:3d} ({cnt/n_failed*100:.1f}% of failed)")

    out_data = {
        "total_bathrooms": total_baths,
        "valid_bathrooms": n_valid,
        "failed_bathrooms": n_failed,
        "valid_pct": round(n_valid / total_baths * 100, 2),
        "failed_pct": round(n_failed / total_baths * 100, 2),
        "failed_under_2m2_count": len(failed_under_2),
        "failed_under_2m2_pct_of_failed": round(len(failed_under_2) / n_failed * 100, 2),
        "failed_under_3m2_count": len(failed_under_3),
        "failed_under_3m2_pct_of_failed": round(len(failed_under_3) / n_failed * 100, 2),
        "failed_over_3m2_count": len(failed_over_3),
        "failed_over_3m2_pct_of_failed": round(len(failed_over_3) / n_failed * 100, 2),
        "area_stats_m2": {
            "min": round(float(min(failed_areas)), 2) if failed_areas else 0.0,
            "p25": round(float(np.percentile(failed_areas, 25)), 2) if failed_areas else 0.0,
            "median": round(float(np.median(failed_areas)), 2) if failed_areas else 0.0,
            "p75": round(float(np.percentile(failed_areas, 75)), 2) if failed_areas else 0.0,
            "max": round(float(max(failed_areas)), 2) if failed_areas else 0.0
        },
        "failure_reasons": reason_counts,
        "failed_samples": failed_bathrooms
    }

    out_dir = Path("outputs/design_stage3")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "heldout_bathrooms_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved analysis to outputs/design_stage3/heldout_bathrooms_analysis.json")

if __name__ == "__main__":
    analyze_heldout_bathrooms()
