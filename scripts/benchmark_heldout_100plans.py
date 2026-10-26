"""
scripts/benchmark_heldout_100plans.py
-------------------------------------
Authoritative held-out benchmark across 100 validation plans (plans 100 to 200)
with strict floating-item prevention rule active:
- Method B (direct wall-mask raster classification)
- Back-to-wall items require >= 80% of back edge within 0.10 m of predicted wall mask
- Floating items (rug, coffee table, centered dining table) exempt
- Zero tuning on plans 100-200
- Plans 200-300 strictly reserved and untouched for Stage 4
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
import scipy.ndimage as ndi
from shapely.geometry import Polygon, LineString

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline

def run_heldout_benchmark():
    print("=" * 80)
    print("BENCHMARK ON 100 HELDOUT VALIDATION PLANS (PLANS 100 to 200)")
    print("WITH STRICT 80% BACK-EDGE WALL CONTACT RULE ACTIVE (PREDICTED WALL MASK)")
    print("=" * 80)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    heldout_files = all_val[100:200]
    print(f"Total heldout plans loaded: {len(heldout_files)}")
    print(f"First 3 heldout plans: {[os.path.basename(f) for f in heldout_files[:3]]}")
    print(f"Last 3 heldout plans: {[os.path.basename(f) for f in heldout_files[-3:]]}")

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    item_min_widths = {
        "bedroom": 1.60,
        "living_room": 2.20,
        "kitchen": 1.20,
        "bathroom": 0.80
    }

    # Required items definition per room type
    required_categories = {
        "bedroom": {"bed", "nightstand", "wardrobe"},
        "living_room": {"sofa", "coffee_table", "tv_unit"},
        "kitchen": {"kitchen_sink", "kitchen_counter", "refrigerator"},
        "bathroom": {"toilet", "bathroom_sink"}
    }

    results = {
        cls_name: {
            "total_rooms": 0,
            "valid_rooms": 0,
            "all_required_placed": 0,
            "items_per_room": [],
            "runtimes_ms": [],
            "failures": {
                "NO_USABLE_WALL": 0,
                "TOO_SMALL": 0,
                "DOOR_BLOCKED": 0,
                "VALIDATION_FAILED": 0,
                "OTHER": 0
            }
        }
        for cls_name in ["bedroom", "living_room", "kitchen", "bathroom"]
    }

    all_back_to_wall_contacts = []
    all_placed_items_records = []

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

        # Distance map on predicted wall mask
        if pred_wall_mask is not None and pred_wall_mask.max() > 0:
            pred_dist_m = ndi.distance_transform_edt(1 - (pred_wall_mask > 0).astype(np.uint8)) * scale_m
        else:
            # Reconstruct from walls
            from src.design.edge_classifier import EdgeClassifier
            wall_bin = EdgeClassifier.rasterize_wall_segments(raw_walls, scale_m)
            pred_dist_m = ndi.distance_transform_edt(1 - wall_bin) * scale_m
            pred_wall_mask = wall_bin

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in item_min_widths:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 1.0:
                continue

            entry = results[norm_type]
            entry["total_rooms"] += 1
            room_uid = f"heldout_{fname}_{r_idx}_{norm_type}"

            t0 = time.perf_counter()
            # Method B: Direct Wall Mask Classification with predicted wall mask
            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=room_uid,
                wall_mask=pred_wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            entry["runtimes_ms"].append(dt_ms)

            if res.is_valid:
                entry["valid_rooms"] += 1
                entry["items_per_room"].append(len(res.items))

                placed_cats = set(it.category for it in res.items)
                req_cats = required_categories[norm_type]
                if req_cats.issubset(placed_cats):
                    entry["all_required_placed"] += 1

                # Measure back-edge contact for all placed back-to-wall items
                for it in res.items:
                    if it.placement_rule == "back_to_wall" and len(it.polygon_coords) >= 4:
                        c_br = it.polygon_coords[1]
                        c_bl = it.polygon_coords[2]
                        back_line = LineString([c_br, c_bl])
                        if back_line.length > 1e-4:
                            pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                            h, w = pred_dist_m.shape
                            contact_count = 0
                            dists = []
                            for pt in pts_sample:
                                ix = int(round(pt.x / scale_m))
                                iy = int(round(pt.y / scale_m))
                                if 0 <= ix < w and 0 <= iy < h:
                                    d = pred_dist_m[iy, ix]
                                else:
                                    d = 99.0
                                dists.append(d)
                                if d <= 0.10:
                                    contact_count += 1
                            ratio = contact_count / len(pts_sample)
                            all_back_to_wall_contacts.append(ratio)
                            all_placed_items_records.append({
                                "plan": fname,
                                "room_type": norm_type,
                                "item_id": it.item_id,
                                "name": it.name,
                                "contact_ratio": round(ratio, 3),
                                "median_dist_m": round(float(np.median(dists)), 3)
                            })
            else:
                entry["items_per_room"].append(0)
                reason = res.failure_reason or "OTHER"
                if reason in entry["failures"]:
                    entry["failures"][reason] += 1
                else:
                    entry["failures"]["OTHER"] += 1

    # Format summary table
    summary_report = {}
    print("\n=== HELDOUT 100-PLAN BENCHMARK RESULTS (80% CONTACT RULE ACTIVE) ===")
    print(f"{'ROOM TYPE':14s} | {'TOTAL':5s} | {'VALID':12s} | {'ALL REQUIRED':14s} | {'ITEMS/ROOM':12s} | {'RUNTIME (ms)':12s}")
    print("-" * 85)

    tot_all = sum(v["total_rooms"] for v in results.values())
    val_all = sum(v["valid_rooms"] for v in results.values())
    all_req_all = sum(v["all_required_placed"] for v in results.values())
    all_runtimes = [rt for v in results.values() for rt in v["runtimes_ms"]]
    all_items = [it for v in results.values() for it in v["items_per_room"] if it > 0]

    for rtype, data in results.items():
        tot = data["total_rooms"]
        val = data["valid_rooms"]
        val_pct = (val / tot * 100) if tot > 0 else 0.0
        req = data["all_required_placed"]
        req_pct = (req / tot * 100) if tot > 0 else 0.0
        req_of_valid = (req / val * 100) if val > 0 else 0.0
        
        valid_items = [it for it in data["items_per_room"] if it > 0]
        mean_items = float(np.mean(valid_items)) if valid_items else 0.0
        mean_rt = float(np.mean(data["runtimes_ms"])) if data["runtimes_ms"] else 0.0

        summary_report[rtype] = {
            "total_rooms": tot,
            "valid_rooms": val,
            "valid_pct": round(val_pct, 2),
            "all_required_placed": req,
            "all_required_pct_of_total": round(req_pct, 2),
            "all_required_pct_of_valid": round(req_of_valid, 2),
            "mean_items_per_valid_room": round(mean_items, 2),
            "mean_runtime_ms": round(mean_rt, 2),
            "failures": data["failures"]
        }

        print(f"{rtype:14s} | {tot:5d} | {val:3d} ({val_pct:5.1f}%) | {req:3d} ({req_pct:5.1f}%)  | {mean_items:5.2f} items  | {mean_rt:6.2f} ms")

    overall_val_pct = (val_all / tot_all * 100) if tot_all > 0 else 0.0
    overall_req_pct = (all_req_all / tot_all * 100) if tot_all > 0 else 0.0
    overall_mean_items = float(np.mean(all_items)) if all_items else 0.0
    overall_mean_rt = float(np.mean(all_runtimes)) if all_runtimes else 0.0

    print("-" * 85)
    print(f"{'OVERALL':14s} | {tot_all:5d} | {val_all:3d} ({overall_val_pct:5.1f}%) | {all_req_all:3d} ({overall_req_pct:5.1f}%)  | {overall_mean_items:5.2f} items  | {overall_mean_rt:6.2f} ms")

    # Back-edge contact statistics
    n_contact_items = len(all_back_to_wall_contacts)
    share_ge_80 = (sum(1 for r in all_back_to_wall_contacts if r >= 0.80) / n_contact_items * 100) if n_contact_items > 0 else 0.0
    mean_contact = float(np.mean(all_back_to_wall_contacts)) * 100 if n_contact_items > 0 else 0.0
    median_contact = float(np.median(all_back_to_wall_contacts)) * 100 if n_contact_items > 0 else 0.0
    p25_contact = float(np.percentile(all_back_to_wall_contacts, 25)) * 100 if n_contact_items > 0 else 0.0
    p75_contact = float(np.percentile(all_back_to_wall_contacts, 75)) * 100 if n_contact_items > 0 else 0.0
    min_contact = float(min(all_back_to_wall_contacts)) * 100 if n_contact_items > 0 else 0.0
    max_contact = float(max(all_back_to_wall_contacts)) * 100 if n_contact_items > 0 else 0.0

    print("\n=== BACK-EDGE WALL CONTACT DISTRIBUTION (PREDICTED WALL MASK) ===")
    print(f"Total placed back-to-wall items: {n_contact_items}")
    print(f"Share with >= 80% wall contact:  {share_ge_80:.2f}% ({sum(1 for r in all_back_to_wall_contacts if r >= 0.80)} / {n_contact_items})")
    print(f"Mean wall contact:               {mean_contact:.2f}%")
    print(f"Median wall contact:             {median_contact:.2f}%")
    print(f"Distribution: min={min_contact:.1f}%, 25%={p25_contact:.1f}%, med={median_contact:.1f}%, 75%={p75_contact:.1f}%, max={max_contact:.1f}%")

    contact_stats = {
        "total_placed_back_to_wall_items": n_contact_items,
        "items_ge_80_contact_count": sum(1 for r in all_back_to_wall_contacts if r >= 0.80),
        "share_ge_80_pct": round(share_ge_80, 2),
        "mean_contact_pct": round(mean_contact, 2),
        "median_contact_pct": round(median_contact, 2),
        "min_contact_pct": round(min_contact, 2),
        "p25_contact_pct": round(p25_contact, 2),
        "p75_contact_pct": round(p75_contact, 2),
        "max_contact_pct": round(max_contact, 2)
    }

    summary_report["OVERALL"] = {
        "total_rooms": tot_all,
        "valid_rooms": val_all,
        "valid_pct": round(overall_val_pct, 2),
        "all_required_placed": all_req_all,
        "all_required_pct": round(overall_req_pct, 2),
        "mean_items_per_valid_room": round(overall_mean_items, 2),
        "mean_runtime_ms": round(overall_mean_rt, 2),
        "wall_contact_distribution": contact_stats
    }

    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "heldout_100plans_contact80_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved heldout 100-plan benchmark with 80% contact rule to: {out_file}")

    # Also update heldout_100plans_benchmark.json as standard record
    with open(out_dir / "heldout_100plans_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

if __name__ == "__main__":
    run_heldout_benchmark()
