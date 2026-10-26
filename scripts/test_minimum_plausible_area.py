"""
scripts/test_minimum_plausible_area.py
--------------------------------------
Evaluates the effect of minimum plausible room areas on held-out plans 100 to 200:
Thresholds:
- bathroom: 2.0 m²
- bedroom: 5.0 m²
- kitchen: 3.0 m²
- living_room: 6.0 m²

Measures:
1. Total rooms per type
2. Share excluded as "not a room (fragment)" (< min plausible area)
3. Placement validity & completeness on the remaining plausible rooms
"""

import os
import sys
import glob
import json
import time
import io
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline

MIN_PLAUSIBLE_ROOM_AREAS = {
    "bathroom": 2.0,
    "bedroom": 5.0,
    "kitchen": 3.0,
    "living_room": 6.0
}

def run_test():
    print("=" * 80)
    print("TEST MINIMUM PLAUSIBLE ROOM AREA PER TYPE ON HELDOUT PLANS 100-200")
    print("Thresholds:", MIN_PLAUSIBLE_ROOM_AREAS)
    print("=" * 80)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    heldout_files = all_val[100:200]

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    required_categories = {
        "bedroom": {"bed", "nightstand", "wardrobe"},
        "living_room": {"sofa", "coffee_table", "tv_unit"},
        "kitchen": {"kitchen_sink", "kitchen_counter", "refrigerator"},
        "bathroom": {"toilet", "bathroom_sink"}
    }

    stats = {
        rtype: {
            "total_all": 0,
            "excluded_fragments": 0,
            "plausible_rooms": 0,
            "plausible_valid": 0,
            "plausible_all_required": 0,
            "plausible_failures": {
                "NO_USABLE_WALL": 0,
                "TOO_SMALL": 0,
                "DOOR_BLOCKED": 0,
                "VALIDATION_FAILED": 0,
                "OTHER": 0
            }
        }
        for rtype in ["bedroom", "living_room", "kitchen", "bathroom"]
    }

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
            if norm_type not in stats:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 0.5:
                continue

            area = poly.area
            min_a = MIN_PLAUSIBLE_ROOM_AREAS[norm_type]

            st = stats[norm_type]
            st["total_all"] += 1

            if area < min_a:
                st["excluded_fragments"] += 1
                continue

            st["plausible_rooms"] += 1

            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=f"heldout_{fname}_{r_idx}",
                wall_mask=pred_wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )

            if res.is_valid:
                st["plausible_valid"] += 1
                placed_cats = set(it.category for it in res.items)
                if required_categories[norm_type].issubset(placed_cats):
                    st["plausible_all_required"] += 1
            else:
                reason = res.failure_reason or "OTHER"
                if reason in st["plausible_failures"]:
                    st["plausible_failures"][reason] += 1
                else:
                    st["plausible_failures"]["OTHER"] += 1

    print("\n" + "=" * 100)
    print(f"{'ROOM TYPE':12s} | {'TOTAL':5s} | {'EXCLUDED (FRAGMENTS)':22s} | {'PLAUSIBLE ROOMS':15s} | {'VALID (COLLISION-FREE)':22s} | {'ALL REQUIRED':15s}")
    print("-" * 100)

    tot_all = sum(s["total_all"] for s in stats.values())
    tot_excl = sum(s["excluded_fragments"] for s in stats.values())
    tot_plaus = sum(s["plausible_rooms"] for s in stats.values())
    tot_val = sum(s["plausible_valid"] for s in stats.values())
    tot_req = sum(s["plausible_all_required"] for s in stats.values())

    for rtype, s in stats.items():
        excl_pct = (s["excluded_fragments"] / s["total_all"] * 100.0) if s["total_all"] > 0 else 0.0
        val_pct = (s["plausible_valid"] / s["plausible_rooms"] * 100.0) if s["plausible_rooms"] > 0 else 0.0
        req_pct = (s["plausible_all_required"] / s["plausible_rooms"] * 100.0) if s["plausible_rooms"] > 0 else 0.0
        
        print(f"{rtype:12s} | {s['total_all']:5d} | {s['excluded_fragments']:3d} / {s['total_all']:3d} ({excl_pct:5.1f}%)       | {s['plausible_rooms']:4d}            | {s['plausible_valid']:3d} / {s['plausible_rooms']:3d} ({val_pct:5.1f}%)         | {s['plausible_all_required']:3d} ({req_pct:5.1f}%)")

    tot_excl_pct = (tot_excl / tot_all * 100.0) if tot_all > 0 else 0.0
    tot_val_pct = (tot_val / tot_plaus * 100.0) if tot_plaus > 0 else 0.0
    tot_req_pct = (tot_req / tot_plaus * 100.0) if tot_plaus > 0 else 0.0

    print("-" * 100)
    print(f"{'OVERALL':12s} | {tot_all:5d} | {tot_excl:3d} / {tot_all:3d} ({tot_excl_pct:5.1f}%)       | {tot_plaus:4d}            | {tot_val:3d} / {tot_plaus:3d} ({tot_val_pct:5.1f}%)         | {tot_req:3d} ({tot_req_pct:5.1f}%)")

    print("\nFAILURES ON REMAINING PLAUSIBLE ROOMS:")
    for rtype, s in stats.items():
        print(f"  {rtype}: {s['plausible_failures']}")

    # Save output JSON
    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "minimum_plausible_area_evaluation.json", "w", encoding="utf-8") as f:
        json.dump({
            "thresholds_m2": MIN_PLAUSIBLE_ROOM_AREAS,
            "results": stats,
            "overall": {
                "total_all": tot_all,
                "excluded_fragments": tot_excl,
                "excluded_pct": round(tot_excl_pct, 2),
                "plausible_rooms": tot_plaus,
                "plausible_valid": tot_val,
                "plausible_valid_pct": round(tot_val_pct, 2),
                "plausible_all_required": tot_req,
                "plausible_all_required_pct": round(tot_req_pct, 2)
            }
        }, f, indent=2)

if __name__ == "__main__":
    run_test()
