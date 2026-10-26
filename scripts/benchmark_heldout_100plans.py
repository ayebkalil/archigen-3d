import os
import sys
import glob
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import io
import scipy.ndimage as ndi
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline

def run_heldout_benchmark():
    print("=" * 80)
    print("BENCHMARK ON 100 HELDOUT VALIDATION PLANS (PLANS 100 to 200)")
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

    for fidx, fpath in enumerate(heldout_files):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        
        gt_mask = d["mask"]  # GT wall raster (1 is wall)
        gt_wall_bin = (gt_mask == 1).astype(np.uint8)

        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        pred_layout = proc["layout"]
        scale_m = pred_layout["metadata"].get("scale_meters_per_pixel", 0.03)

        raw_walls = pred_layout.get("walls", [])
        doors = pred_layout.get("doors", [])
        windows = pred_layout.get("windows", [])

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
            # Method B: Direct Wall Mask Classification (preferring wall raster)
            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=room_uid,
                wall_mask=gt_wall_bin,
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
            else:
                entry["items_per_room"].append(0)
                reason = res.failure_reason or "OTHER"
                if reason in entry["failures"]:
                    entry["failures"][reason] += 1
                else:
                    entry["failures"]["OTHER"] += 1

    # Format summary table
    summary_report = {}
    print("\n=== HELDOUT 100-PLAN BENCHMARK RESULTS ===")
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

    summary_report["OVERALL"] = {
        "total_rooms": tot_all,
        "valid_rooms": val_all,
        "valid_pct": round(overall_val_pct, 2),
        "all_required_placed": all_req_all,
        "all_required_pct": round(overall_req_pct, 2),
        "mean_items_per_valid_room": round(overall_mean_items, 2),
        "mean_runtime_ms": round(overall_mean_rt, 2)
    }

    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "heldout_100plans_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved heldout 100-plan benchmark to: {out_file}")

if __name__ == "__main__":
    run_heldout_benchmark()
