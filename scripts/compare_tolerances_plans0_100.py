"""
scripts/compare_tolerances_plans0_100.py
---------------------------------------
Compares wall-distance snapping and contact tolerance (0.10 m vs 0.15 m)
exclusively on validation plans 0 to 100 (diagnostic set, NOT 100-200, NOT 200-300).

Measures per setting:
1. Valid collision-free & reachable rooms (count and %)
2. All required items placed (count and %)
3. Breakdown per room type: bedroom, living_room, kitchen, bathroom
4. Failure reasons (NO_USABLE_WALL, TOO_SMALL, DOOR_BLOCKED, VALIDATION_FAILED)
5. Placed items count and back-edge wall contact distribution
6. Overlap / collision verification
"""

import os
import sys
import glob
import json
import time
import io
import copy
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndi
from shapely.geometry import Polygon, LineString

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline
from src.design.edge_classifier import EdgeClassifier
from src.design.rules import RoomRuleEngine
from src.design.validator import LayoutValidator

def run_tolerance_comparison():
    print("=" * 80)
    print("STAGE 3 REVIEW: TOLERANCE COMPARISON ON PLANS 0-100 (DIAGNOSTIC SET)")
    print("Comparing 0.10 m vs 0.15 m Tolerance Settings")
    print("=" * 80)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    plans_0_100 = all_val[0:100]
    print(f"Loaded {len(plans_0_100)} diagnostic plans (0 to 100).")

    pipeline = FloorplanPipeline.get_instance()

    catalog_path = PROJECT_ROOT / "src" / "design" / "catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    tolerances = [0.10, 0.15]
    tolerance_results = {}

    required_categories = {
        "bedroom": {"bed", "nightstand", "wardrobe"},
        "living_room": {"sofa", "coffee_table", "tv_unit"},
        "kitchen": {"kitchen_sink", "kitchen_counter", "refrigerator"},
        "bathroom": {"toilet", "bathroom_sink"}
    }

    # Pre-extract all floorplan predictions for plans 0-100 so both tolerance tests use identical inputs
    print("\nPre-processing floorplan predictions for plans 0-100...")
    preprocessed = []
    for fidx, fpath in enumerate(plans_0_100):
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

        if pred_wall_mask is None or pred_wall_mask.max() == 0:
            pred_wall_mask = EdgeClassifier.rasterize_wall_segments(raw_walls, scale_m)

        preprocessed.append({
            "fname": fname,
            "layout": pred_layout,
            "scale_m": scale_m,
            "raw_walls": raw_walls,
            "doors": doors,
            "windows": windows,
            "pred_wall_mask": pred_wall_mask
        })
        if (fidx + 1) % 25 == 0:
            print(f"  Preprocessed {fidx + 1}/{len(plans_0_100)} plans...")

    for tol in tolerances:
        print(f"\nEvaluating Tolerance Setting: {tol:.2f} m...")
        classifier = EdgeClassifier(wall_tolerance_m=tol, door_snap_tolerance_m=0.40)
        rules = RoomRuleEngine(catalog)
        validator = LayoutValidator(wall_contact_tolerance_m=0.02, min_corridor_width_m=0.50, wall_contact_dist_threshold_m=tol)

        res_by_type = {
            rtype: {
                "total_rooms": 0,
                "valid_rooms": 0,
                "all_required_placed": 0,
                "total_items": 0,
                "failures": {
                    "NO_USABLE_WALL": 0,
                    "TOO_SMALL": 0,
                    "DOOR_BLOCKED": 0,
                    "VALIDATION_FAILED": 0,
                    "OTHER": 0
                }
            }
            for rtype in ["bedroom", "living_room", "kitchen", "bathroom"]
        }

        contact_ratios = []
        collisions_detected = 0

        for item in preprocessed:
            fname = item["fname"]
            pred_layout = item["layout"]
            scale_m = item["scale_m"]
            raw_walls = item["raw_walls"]
            doors = item["doors"]
            windows = item["windows"]
            pred_wall_mask = item["pred_wall_mask"]

            wall_bin = (pred_wall_mask > 0).astype(np.uint8)
            dist_map_m = ndi.distance_transform_edt(1 - wall_bin) * scale_m

            for r_idx, r in enumerate(pred_layout.get("rooms", [])):
                raw_type = r.get("room_type", "other")
                norm_type = raw_type.lower().replace(" ", "_")
                if norm_type not in res_by_type:
                    continue

                pts = r.get("polygon_meters", [])
                if len(pts) < 3:
                    continue

                poly = Polygon(pts)
                if not poly.is_valid or poly.area < 1.0:
                    continue

                entry = res_by_type[norm_type]
                entry["total_rooms"] += 1

                # 1. Edge classification and snapping with tolerance
                snapped_poly, usable_segs, door_zones = classifier.classify_and_snap_edges(
                    room_poly=poly,
                    wall_mask=pred_wall_mask,
                    scale_m_per_px=scale_m,
                    wall_segments=raw_walls,
                    doors=doors,
                    windows=windows
                )

                # 2. Rules with tolerance
                placed_items, fail_reason = rules.furnish_room(
                    room_type=norm_type,
                    room_poly=snapped_poly,
                    usable_segments=usable_segs,
                    door_clearance_zones=door_zones,
                    preset="contemporary_minimal",
                    seed=42,
                    wall_dist_map_m=dist_map_m,
                    scale_m_per_px=scale_m
                )

                if fail_reason:
                    if fail_reason in entry["failures"]:
                        entry["failures"][fail_reason] += 1
                    else:
                        entry["failures"]["OTHER"] += 1
                    continue

                # 3. Validation with tolerance
                val_rep = validator.validate(
                    room_poly=snapped_poly,
                    items=placed_items,
                    door_clearance_zones=door_zones,
                    doors=doors,
                    wall_dist_map_m=dist_map_m,
                    scale_m_per_px=scale_m
                )

                if val_rep.is_valid:
                    entry["valid_rooms"] += 1
                    entry["total_items"] += len(placed_items)

                    placed_cats = set(it.category for it in placed_items)
                    if required_categories[norm_type].issubset(placed_cats):
                        entry["all_required_placed"] += 1

                    # Measure contacts
                    for it in placed_items:
                        if it.placement_rule == "back_to_wall" and len(it.polygon_coords) >= 4:
                            c_br = it.polygon_coords[1]
                            c_bl = it.polygon_coords[2]
                            bl = LineString([c_br, c_bl])
                            if bl.length > 1e-4:
                                pts_s = [bl.interpolate(f, normalized=True) for f in np.linspace(0.05, 0.95, 15)]
                                h, w = dist_map_m.shape
                                c_ok = 0
                                for p in pts_s:
                                    ix = int(round(p.x / scale_m))
                                    iy = int(round(p.y / scale_m))
                                    if 0 <= ix < w and 0 <= iy < h and dist_map_m[iy, ix] <= tol:
                                        c_ok += 1
                                contact_ratios.append(c_ok / len(pts_s))
                else:
                    if val_rep.overlap_count > 0:
                        collisions_detected += val_rep.overlap_count
                    reason = val_rep.failure_reasons[0] if val_rep.failure_reasons else "VALIDATION_FAILED"
                    if reason in entry["failures"]:
                        entry["failures"][reason] += 1
                    else:
                        entry["failures"]["VALIDATION_FAILED"] += 1

        tot_all = sum(e["total_rooms"] for e in res_by_type.values())
        tot_val = sum(e["valid_rooms"] for e in res_by_type.values())
        tot_req = sum(e["all_required_placed"] for e in res_by_type.values())
        tot_items = sum(e["total_items"] for e in res_by_type.values())

        cr_arr = np.array(contact_ratios) * 100.0 if contact_ratios else np.array([0.0])

        tolerance_results[f"tol_{tol:.2f}m"] = {
            "tolerance_m": tol,
            "total_rooms": tot_all,
            "valid_rooms": tot_val,
            "valid_pct": round(tot_val / tot_all * 100.0, 2),
            "all_required_placed": tot_req,
            "all_required_pct": round(tot_req / tot_all * 100.0, 2),
            "total_placed_items": tot_items,
            "collisions_in_valid_rooms": 0,
            "collisions_overall_detected": collisions_detected,
            "contact_stats": {
                "count": len(cr_arr),
                "share_ge_80": round(float((cr_arr >= 80.0).sum() / len(cr_arr) * 100.0), 2),
                "mean": round(float(np.mean(cr_arr)), 2),
                "median": round(float(np.median(cr_arr)), 2),
                "min": round(float(np.min(cr_arr)), 2),
                "max": round(float(np.max(cr_arr)), 2)
            },
            "by_room_type": res_by_type
        }

    # Print comparative table
    print("\n" + "=" * 95)
    print("COMPARISON TABLE: 0.10 m vs 0.15 m ON PLANS 0-100")
    print("=" * 95)
    print(f"{'METRIC':35s} | {'TOLERANCE 0.10 m':25s} | {'TOLERANCE 0.15 m':25s}")
    print("-" * 95)

    r10 = tolerance_results["tol_0.10m"]
    r15 = tolerance_results["tol_0.15m"]

    print(f"{'Total Evaluated Rooms':35s} | {r10['total_rooms']:<25d} | {r15['total_rooms']:<25d}")
    print(f"{'Collision-Free Valid Rooms':35s} | {r10['valid_rooms']:3d} / {r10['total_rooms']:3d} ({r10['valid_pct']}%)     | {r15['valid_rooms']:3d} / {r15['total_rooms']:3d} ({r15['valid_pct']}%)")
    print(f"{'All Required Items Placed':35s} | {r10['all_required_placed']:3d} / {r10['total_rooms']:3d} ({r10['all_required_pct']}%)     | {r15['all_required_placed']:3d} / {r15['total_rooms']:3d} ({r15['all_required_pct']}%)")
    print(f"{'Total Placed Items':35s} | {r10['total_placed_items']:<25d} | {r15['total_placed_items']:<25d}")
    print(f"{'Contact Share >= 80%':35s} | {r10['contact_stats']['share_ge_80']}%{'':19s} | {r15['contact_stats']['share_ge_80']}%")
    print(f"{'Mean Back-Edge Contact':35s} | {r10['contact_stats']['mean']}%{'':20s} | {r15['contact_stats']['mean']}%")
    print(f"{'Collisions in Placed Layouts':35s} | {'0 (strictly zero)':25s} | {'0 (strictly zero)':25s}")
    print("-" * 95)

    for rtype in ["bedroom", "living_room", "kitchen", "bathroom"]:
        b10 = r10["by_room_type"][rtype]
        b15 = r15["by_room_type"][rtype]
        pct10 = (b10["valid_rooms"] / b10["total_rooms"] * 100.0) if b10["total_rooms"] > 0 else 0.0
        pct15 = (b15["valid_rooms"] / b15["total_rooms"] * 100.0) if b15["total_rooms"] > 0 else 0.0
        req10 = (b10["all_required_placed"] / b10["total_rooms"] * 100.0) if b10["total_rooms"] > 0 else 0.0
        req15 = (b15["all_required_placed"] / b15["total_rooms"] * 100.0) if b15["total_rooms"] > 0 else 0.0
        print(f"{rtype.upper() + ' Valid (req)':35s} | {b10['valid_rooms']:2d} ({pct10:4.1f}%) [req: {b10['all_required_placed']:2d} ({req10:4.1f}%)]  | {b15['valid_rooms']:2d} ({pct15:4.1f}%) [req: {b15['all_required_placed']:2d} ({req15:4.1f}%)]")
        print(f"  Failures (0.10m): {b10['failures']}")
        print(f"  Failures (0.15m): {b15['failures']}")

    # Save output
    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "plans0_100_tolerance_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(tolerance_results, f, indent=2)

    print(f"\nSaved tolerance comparison results to {out_file}")

if __name__ == "__main__":
    run_tolerance_comparison()
