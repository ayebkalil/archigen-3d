"""
scripts/benchmark_heldout_plans200_300.py
----------------------------------------
Authoritative final benchmark run ONCE on held-out CubiCasa validation plans 200 to 300.
Strict protocol:
- Zero hyperparameter tuning on plans 200-300
- Chosen configuration:
  * Method B (direct predicted wall-mask raster distance transform)
  * Strict 80% back-edge contact rule (<= 0.10 m)
  * Minimum plausible room area thresholds (bathroom: 2.0m², bedroom: 5.0m², kitchen: 3.0m², living: 6.0m²)
  * Sub-threshold fragments marked as "not a room (fragment)"
  * Fixed seed = 42
- Saves output to outputs/design_stage3/heldout_plans200_300_benchmark.json
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
from shapely.geometry import Polygon, LineString, Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline
from src.design.types import MIN_PLAUSIBLE_ROOM_AREAS

def run_benchmark():
    print("=" * 85)
    print("AUTHORITATIVE FINAL BENCHMARK: HELD-OUT VALIDATION PLANS 200 TO 300")
    print("Strictly run once with zero hyperparameter tuning.")
    print("=" * 85)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    plans_200_300 = all_val[200:300]
    print(f"Loaded {len(plans_200_300)} untouched validation plans.")
    print(f"First plan: {os.path.basename(plans_200_300[0])}")
    print(f"Last plan:  {os.path.basename(plans_200_300[-1])}")

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    required_categories = {
        "bedroom": {"bed", "nightstand", "wardrobe"},
        "living_room": {"sofa", "coffee_table", "tv_unit"},
        "kitchen": {"kitchen_sink", "kitchen_counter", "refrigerator"},
        "bathroom": {"toilet", "bathroom_sink"}
    }

    results = {
        rtype: {
            "total_detected_rooms": 0,
            "excluded_fragments": 0,
            "plausible_rooms": 0,
            "valid_rooms": 0,
            "all_required_placed": 0,
            "items_per_valid_room": [],
            "runtimes_ms": [],
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

    pred_contacts = []
    gt_contacts = []
    total_placed_items = 0
    items_blocking_gt_win = 0
    items_blocking_pred_win = 0

    t0_all = time.perf_counter()

    for fidx, fpath in enumerate(plans_200_300):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        
        gt_struct = d["struct_mask"]
        gt_wall_mask = ((gt_struct == 1) | (gt_struct == 2)).astype(np.uint8)
        gt_win_mask = (gt_struct == 4).astype(np.uint8)

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
            from src.design.edge_classifier import EdgeClassifier
            pred_wall_mask = EdgeClassifier.rasterize_wall_segments(raw_walls, scale_m)

        pred_dist_m = ndi.distance_transform_edt(1 - (pred_wall_mask > 0).astype(np.uint8)) * scale_m
        gt_dist_m = ndi.distance_transform_edt(1 - (gt_wall_mask > 0).astype(np.uint8)) * scale_m

        pred_win_polys = []
        for win in windows:
            if "polygon_meters" in win and len(win["polygon_meters"]) >= 3:
                wp = Polygon(win["polygon_meters"])
                if wp.is_valid:
                    pred_win_polys.append(wp)
            elif "position_meters" in win:
                px, py = win["position_meters"]
                ww = win.get("width_m", 1.0)
                pred_win_polys.append(Point(px, py).buffer(ww / 2.0))

        h, w = pred_wall_mask.shape

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in results:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 0.5:
                continue

            entry = results[norm_type]
            entry["total_detected_rooms"] += 1

            min_a = MIN_PLAUSIBLE_ROOM_AREAS[norm_type]
            if poly.area < min_a:
                entry["excluded_fragments"] += 1
                continue

            entry["plausible_rooms"] += 1

            t_room = time.perf_counter()
            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=f"heldout_{fname}_{r_idx}_{norm_type}",
                wall_mask=pred_wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )
            dt_room_ms = (time.perf_counter() - t_room) * 1000.0
            entry["runtimes_ms"].append(dt_room_ms)

            if res.is_valid:
                entry["valid_rooms"] += 1
                entry["items_per_valid_room"].append(len(res.items))

                placed_cats = set(it.category for it in res.items)
                if required_categories[norm_type].issubset(placed_cats):
                    entry["all_required_placed"] += 1

                for it in res.items:
                    total_placed_items += 1
                    it_poly = Polygon(it.polygon_coords) if len(it.polygon_coords) >= 3 else None

                    # Check window blocking
                    blocks_gt = False
                    if it_poly is not None and it_poly.is_valid:
                        minx, miny, maxx, maxy = it_poly.bounds
                        ix0 = max(0, int(np.floor(minx / scale_m)))
                        ix1 = min(w, int(np.ceil(maxx / scale_m)) + 1)
                        iy0 = max(0, int(np.floor(miny / scale_m)))
                        iy1 = min(h, int(np.ceil(maxy / scale_m)) + 1)
                        sub_win = gt_win_mask[iy0:iy1, ix0:ix1]
                        if sub_win.max() > 0:
                            win_ys, win_xs = np.where(sub_win > 0)
                            for wy, wx in zip(win_ys, win_xs):
                                pt_m = Point((ix0 + wx + 0.5) * scale_m, (iy0 + wy + 0.5) * scale_m)
                                if it_poly.contains(pt_m) or it_poly.distance(pt_m) < 1e-3:
                                    blocks_gt = True
                                    break
                    if blocks_gt:
                        items_blocking_gt_win += 1

                    blocks_pred = False
                    if it_poly is not None and it_poly.is_valid:
                        for wp in pred_win_polys:
                            if it_poly.intersects(wp):
                                blocks_pred = True
                                break
                    if blocks_pred:
                        items_blocking_pred_win += 1

                    # Check back-edge contact
                    if it.placement_rule == "back_to_wall" and len(it.polygon_coords) >= 4:
                        c_br = it.polygon_coords[1]
                        c_bl = it.polygon_coords[2]
                        bl = LineString([c_br, c_bl])
                        if bl.length > 1e-4:
                            pts_sample = [bl.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                            c_p = sum(1 for p in pts_sample if 0 <= int(p.x/scale_m) < w and 0 <= int(p.y/scale_m) < h and pred_dist_m[int(p.y/scale_m), int(p.x/scale_m)] <= 0.10)
                            c_g = sum(1 for p in pts_sample if 0 <= int(p.x/scale_m) < w and 0 <= int(p.y/scale_m) < h and gt_dist_m[int(p.y/scale_m), int(p.x/scale_m)] <= 0.10)
                            pred_contacts.append(c_p / len(pts_sample))
                            gt_contacts.append(c_g / len(pts_sample))
            else:
                reason = res.failure_reason or "OTHER"
                if reason in entry["failures"]:
                    entry["failures"][reason] += 1
                else:
                    entry["failures"]["OTHER"] += 1

        if (fidx + 1) % 25 == 0:
            print(f"Processed {fidx + 1}/100 plans...")

    dt_total = time.perf_counter() - t0_all
    print(f"\nExecution finished in {dt_total:.2f}s.")

    # Summarize results
    summary_report = {}
    print("\n" + "=" * 105)
    print("FINAL HELD-OUT BENCHMARK (PLANS 200-300) RESULTS TABLE")
    print("=" * 105)
    print(f"{'ROOM TYPE':12s} | {'DETECTED':8s} | {'FRAGMENTS':14s} | {'PLAUSIBLE':10s} | {'COLLISION-FREE':18s} | {'ALL REQUIRED':16s} | {'ITEMS/ROOM':11s} | {'RUNTIME':9s}")
    print("-" * 105)

    tot_det = sum(v["total_detected_rooms"] for v in results.values())
    tot_frag = sum(v["excluded_fragments"] for v in results.values())
    tot_plaus = sum(v["plausible_rooms"] for v in results.values())
    tot_val = sum(v["valid_rooms"] for v in results.values())
    tot_req = sum(v["all_required_placed"] for v in results.values())
    all_items = [it for v in results.values() for it in v["items_per_valid_room"]]
    all_rt = [rt for v in results.values() for rt in v["runtimes_ms"]]

    for rtype, data in results.items():
        det = data["total_detected_rooms"]
        frag = data["excluded_fragments"]
        frag_pct = (frag / det * 100.0) if det > 0 else 0.0
        plaus = data["plausible_rooms"]
        val = data["valid_rooms"]
        val_pct = (val / plaus * 100.0) if plaus > 0 else 0.0
        req = data["all_required_placed"]
        req_pct = (req / plaus * 100.0) if plaus > 0 else 0.0
        mean_items = float(np.mean(data["items_per_valid_room"])) if data["items_per_valid_room"] else 0.0
        mean_rt = float(np.mean(data["runtimes_ms"])) if data["runtimes_ms"] else 0.0

        summary_report[rtype] = {
            "total_detected_rooms": det,
            "excluded_fragments": frag,
            "excluded_fragments_pct": round(frag_pct, 2),
            "plausible_rooms": plaus,
            "valid_rooms": val,
            "valid_pct_of_plausible": round(val_pct, 2),
            "all_required_placed": req,
            "all_required_pct_of_plausible": round(req_pct, 2),
            "mean_items_per_valid_room": round(mean_items, 2),
            "mean_runtime_ms": round(mean_rt, 2),
            "failures": data["failures"]
        }

        print(f"{rtype:12s} | {det:8d} | {frag:3d} ({frag_pct:4.1f}%)   | {plaus:10d} | {val:3d} / {plaus:3d} ({val_pct:5.1f}%) | {req:3d} ({req_pct:5.1f}%)      | {mean_items:5.2f} items | {mean_rt:5.1f} ms")

    overall_frag_pct = (tot_frag / tot_det * 100.0) if tot_det > 0 else 0.0
    overall_val_pct = (tot_val / tot_plaus * 100.0) if tot_plaus > 0 else 0.0
    overall_req_pct = (tot_req / tot_plaus * 100.0) if tot_plaus > 0 else 0.0
    overall_mean_items = float(np.mean(all_items)) if all_items else 0.0
    overall_mean_rt = float(np.mean(all_rt)) if all_rt else 0.0

    print("-" * 105)
    print(f"{'OVERALL':12s} | {tot_det:8d} | {tot_frag:3d} ({overall_frag_pct:4.1f}%)   | {tot_plaus:10d} | {tot_val:3d} / {tot_plaus:3d} ({overall_val_pct:5.1f}%) | {tot_req:3d} ({overall_req_pct:5.1f}%)      | {overall_mean_items:5.2f} items | {overall_mean_rt:5.1f} ms")

    # Contacts summary
    pr_arr = np.array(pred_contacts) * 100.0 if pred_contacts else np.array([0.0])
    gt_arr = np.array(gt_contacts) * 100.0 if gt_contacts else np.array([0.0])

    print("\n" + "=" * 85)
    print("BACK-EDGE CONTACT & WINDOW BLOCKING (PLANS 200-300)")
    print("=" * 85)
    print(f"Total placed items:               {total_placed_items}")
    print(f"Placed back-to-wall items:        {len(pred_contacts)}")
    print(f"Pred Wall Mask Contact >= 80%:    {(pr_arr >= 80.0).sum()} / {len(pr_arr)} ({((pr_arr >= 80.0).sum()/len(pr_arr)*100.0):.2f}%)")
    print(f"Pred Wall Mask Mean Contact:      {np.mean(pr_arr):.2f}% (median: {np.median(pr_arr):.2f}%)")
    print(f"GT Wall Mask Contact >= 80%:      {(gt_arr >= 80.0).sum()} / {len(gt_arr)} ({((gt_arr >= 80.0).sum()/len(gt_arr)*100.0):.2f}%)")
    print(f"GT Wall Mask Mean Contact:        {np.mean(gt_arr):.2f}% (median: {np.median(gt_arr):.2f}%)")
    print(f"Items blocking GT Window Mask:    {items_blocking_gt_win} / {total_placed_items} ({(items_blocking_gt_win/total_placed_items*100.0):.2f}%)")
    print(f"Items blocking Predicted Window:  {items_blocking_pred_win} / {total_placed_items} ({(items_blocking_pred_win/total_placed_items*100.0):.2f}%)")

    # Save to JSON
    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "heldout_plans200_300_benchmark.json"

    final_payload = {
        "dataset": "CubiCasa5k validation plans 200-300 (strictly held-out, run once)",
        "plans_count": len(plans_200_300),
        "protocol": {
            "method": "Method B (direct wall-mask raster distance transform)",
            "wall_contact_rule": ">= 80% of back-edge sample points within 0.10 m of predicted wall mask",
            "minimum_plausible_room_areas_m2": MIN_PLAUSIBLE_ROOM_AREAS,
            "seed": 42
        },
        "overall": {
            "total_detected_rooms": tot_det,
            "excluded_fragments": tot_frag,
            "excluded_fragments_pct": round(overall_frag_pct, 2),
            "plausible_rooms": tot_plaus,
            "valid_rooms": tot_val,
            "valid_pct_of_plausible": round(overall_val_pct, 2),
            "all_required_placed": tot_req,
            "all_required_pct_of_plausible": round(overall_req_pct, 2),
            "total_placed_items": total_placed_items,
            "mean_items_per_valid_room": round(overall_mean_items, 2),
            "mean_runtime_ms": round(overall_mean_rt, 2)
        },
        "by_room_type": summary_report,
        "contact_and_windows": {
            "back_to_wall_items_count": len(pred_contacts),
            "pred_contact_share_ge_80": round(float((pr_arr >= 80.0).sum() / len(pr_arr) * 100.0), 2),
            "pred_contact_mean": round(float(np.mean(pr_arr)), 2),
            "pred_contact_median": round(float(np.median(pr_arr)), 2),
            "gt_contact_share_ge_80": round(float((gt_arr >= 80.0).sum() / len(gt_arr) * 100.0), 2),
            "gt_contact_mean": round(float(np.mean(gt_arr)), 2),
            "gt_contact_median": round(float(np.median(gt_arr)), 2),
            "items_blocking_gt_window_count": items_blocking_gt_win,
            "items_blocking_gt_window_pct": round(float(items_blocking_gt_win / total_placed_items * 100.0), 2),
            "items_blocking_pred_window_count": items_blocking_pred_win,
            "items_blocking_pred_window_pct": round(float(items_blocking_pred_win / total_placed_items * 100.0), 2)
        }
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nSaved final held-out benchmark results to {out_file}")

if __name__ == "__main__":
    run_benchmark()
