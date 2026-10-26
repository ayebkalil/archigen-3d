"""
scripts/test_snapping_tolerances.py
------------------------------------
Tests snapping tolerance at 0.15m, 0.20m, and 0.25m on predicted masks from the validation set.
Excludes the plans used for the screenshots: colorful_11261.npz and colorful_1148.npz.
Evaluates:
  - Overall collision-free and reachable validity rate
  - Count of NO_USABLE_WALL vs TOO_SMALL vs other failures
  - Any new pairwise collisions / overlaps (count and max overlap area in m2)
  - Any door clearance violations
  - Out-of-bounds containment violations
  - Per room type breakdown (Bedroom, Living Room, Kitchen, Bathroom)
"""

import os
import sys
import glob
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.vectorizer import FloorplanVectorizer
from src.design.engine import RoomDesignEngine
from src.design.edge_classifier import EdgeClassifier
from src.design.validator import LayoutValidator
from backend.app.floorplan_engine import FloorplanPipeline

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "design_stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_PLANS = {"colorful_11261.npz", "colorful_1148.npz"}

def run_tolerance_experiment():
    print("=" * 80)
    print("SNAPPING TOLERANCE EXPERIMENT (0.15m vs 0.20m vs 0.25m) ON PREDICTED MASKS")
    print(f"Excluding screenshot plans: {EXCLUDED_PLANS}")
    print("=" * 80)

    val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    eval_files = [f for f in val_files if os.path.basename(f) not in EXCLUDED_PLANS][:80]
    print(f"Evaluating across {len(eval_files)} validation plans.")

    pipeline = FloorplanPipeline.get_instance()
    tolerances = [0.15, 0.20, 0.25]
    results = {}

    item_min_widths = {
        "bedroom": 1.60,
        "living_room": 2.20,
        "kitchen": 1.20,
        "bathroom": 0.80
    }

    # Pre-compute predictions for the plans to ensure identical inputs across tolerances
    print("Pre-computing model predictions for 80 plans...")
    cached_predictions = []
    for fpath in eval_files:
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        cached_predictions.append({
            "fname": fname,
            "layout": proc["layout"]
        })

    for tol in tolerances:
        print(f"\n--- Testing Snapping Tolerance: {tol:.2f} m ---")
        engine = RoomDesignEngine()
        # Override tolerance in classifier
        engine.classifier = EdgeClassifier(wall_tolerance_m=tol, door_snap_tolerance_m=0.40)

        stats = {
            "tolerance_m": tol,
            "total_plans": len(eval_files),
            "total_rooms": 0,
            "eligible_rooms": 0,
            "valid_rooms": 0,
            "overlap_count": 0,
            "max_overlap_area_m2": 0.0,
            "door_violations": 0,
            "out_of_bounds_count": 0,
            "max_wall_gap_cm": 0.0,
            "failure_counts": {
                "NO_USABLE_WALL": 0,
                "TOO_SMALL": 0,
                "DOOR_BLOCKED": 0,
                "COLLISION": 0,
                "UNREACHABLE": 0,
                "OTHER": 0
            },
            "by_type": {}
        }

        for item in cached_predictions:
            fname = item["fname"]
            pred_layout = item["layout"]
            scale_m = pred_layout["metadata"].get("scale_meters_per_pixel", 0.03)
            doors = pred_layout.get("doors", [])
            windows = pred_layout.get("windows", [])
            walls = pred_layout.get("walls", [])

            for r_idx, r in enumerate(pred_layout.get("rooms", [])):
                stats["total_rooms"] += 1
                raw_type = r.get("room_type", "other")
                poly_pts = r.get("polygon_meters", [])
                if len(poly_pts) < 3:
                    continue

                # Proper normalized room type mapping (handling spaces)
                norm_type = raw_type.lower().replace(" ", "_")
                if norm_type not in item_min_widths:
                    continue

                stats["eligible_rooms"] += 1
                if norm_type not in stats["by_type"]:
                    stats["by_type"][norm_type] = {
                        "total": 0,
                        "valid": 0,
                        "failures": {},
                        "overlap_count": 0,
                        "door_violations": 0
                    }
                t_entry = stats["by_type"][norm_type]
                t_entry["total"] += 1

                room_uid = f"tol_{tol}_{fname}_{r_idx}_{norm_type}"
                res = engine.design_room(
                    room_polygon_meters=poly_pts,
                    room_type=norm_type,
                    room_id=room_uid,
                    wall_segments=walls,
                    doors=doors,
                    windows=windows,
                    preset="contemporary_minimal",
                    seed=42
                )

                if res.validation:
                    v = res.validation
                    if v.overlap_count > 0:
                        stats["overlap_count"] += v.overlap_count
                        t_entry["overlap_count"] += v.overlap_count
                        stats["max_overlap_area_m2"] = max(stats["max_overlap_area_m2"], v.max_overlap_area_m2)
                    if v.door_clearance_violation:
                        stats["door_violations"] += 1
                        t_entry["door_violations"] += 1
                    if v.out_of_bounds_count > 0:
                        stats["out_of_bounds_count"] += v.out_of_bounds_count
                    stats["max_wall_gap_cm"] = max(stats["max_wall_gap_cm"], v.max_wall_gap_cm)

                if res.is_valid:
                    stats["valid_rooms"] += 1
                    t_entry["valid"] += 1
                else:
                    reason = res.failure_reason or "OTHER"
                    if reason in stats["failure_counts"]:
                        stats["failure_counts"][reason] += 1
                    else:
                        stats["failure_counts"]["OTHER"] += 1
                    t_entry["failures"][reason] = t_entry["failures"].get(reason, 0) + 1

        stats["validity_pct"] = round((stats["valid_rooms"] / stats["eligible_rooms"]) * 100.0, 2) if stats["eligible_rooms"] > 0 else 0.0
        for rtype, d in stats["by_type"].items():
            d["validity_pct"] = round((d["valid"] / d["total"]) * 100.0, 2) if d["total"] > 0 else 0.0

        print(f"Tolerance {tol:.2f} m: Valid {stats['valid_rooms']}/{stats['eligible_rooms']} ({stats['validity_pct']}%)")
        print(f"  Failures: NO_USABLE_WALL={stats['failure_counts']['NO_USABLE_WALL']}, TOO_SMALL={stats['failure_counts']['TOO_SMALL']}, OTHER={stats['failure_counts']['OTHER']}")
        print(f"  Collisions / Overlaps: {stats['overlap_count']} (max area: {stats['max_overlap_area_m2']} m2)")
        print(f"  Door Violations: {stats['door_violations']}, Out of bounds: {stats['out_of_bounds_count']}")
        results[f"tol_{tol:.2f}m"] = stats

    out_file = OUTPUT_DIR / "snapping_tolerance_experiment.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved tolerance experiment results to: {out_file}")
    return results

if __name__ == "__main__":
    run_tolerance_experiment()
