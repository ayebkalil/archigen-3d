"""
scripts/measure_contact_and_windows_heldout.py
---------------------------------------------
Evaluates held-out plans 100 to 200 to:
1. Measure back-edge contact against both:
   - Predicted wall mask
   - Ground-truth wall mask (struct_mask == 1 | struct_mask == 2)
   and compute full percentile distributions.
2. Measure the share of placed items whose footprint blocks a window
   against ground-truth window mask (struct_mask == 4) and predicted windows.
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

def run_evaluation():
    print("=" * 80)
    print("STAGE 3 FLOATING-RULE REVIEW: CONTACT DISTRIBUTIONS & WINDOW BLOCKING")
    print("Evaluated across held-out plans 100 to 200")
    print("=" * 80)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    heldout_files = all_val[100:200]
    print(f"Loaded {len(heldout_files)} held-out validation plans.")

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    pred_contacts = []
    gt_contacts = []
    
    total_placed_items = 0
    items_blocking_gt_window = 0
    items_blocking_pred_window = 0
    
    items_detail = []

    t0_all = time.perf_counter()

    for idx, fpath in enumerate(heldout_files):
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

        # Build window polygons in meter space for predicted windows
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

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in ["bedroom", "living_room", "kitchen", "bathroom"]:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 1.0:
                continue

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

            if not res.is_valid:
                continue

            h, w = pred_wall_mask.shape

            for it in res.items:
                total_placed_items += 1
                it_poly = Polygon(it.polygon_coords) if len(it.polygon_coords) >= 3 else None

                # 1. Window blocking check against Ground-Truth Window Mask
                blocks_gt_win = False
                if it_poly is not None and it_poly.is_valid:
                    # Raster check: sample points inside item or bounding box
                    minx, miny, maxx, maxy = it_poly.bounds
                    ix0 = max(0, int(np.floor(minx / scale_m)))
                    ix1 = min(w, int(np.ceil(maxx / scale_m)) + 1)
                    iy0 = max(0, int(np.floor(miny / scale_m)))
                    iy1 = min(h, int(np.ceil(maxy / scale_m)) + 1)
                    
                    sub_win = gt_win_mask[iy0:iy1, ix0:ix1]
                    if sub_win.max() > 0:
                        # Check precise point-in-polygon for window pixels
                        win_ys, win_xs = np.where(sub_win > 0)
                        for wy, wx in zip(win_ys, win_xs):
                            pt_m = Point((ix0 + wx + 0.5) * scale_m, (iy0 + wy + 0.5) * scale_m)
                            if it_poly.contains(pt_m) or it_poly.distance(pt_m) < 1e-3:
                                blocks_gt_win = True
                                break

                # Window blocking check against predicted window objects
                blocks_pred_win = False
                if it_poly is not None and it_poly.is_valid:
                    for wp in pred_win_polys:
                        if it_poly.intersects(wp):
                            blocks_pred_win = True
                            break

                if blocks_gt_win:
                    items_blocking_gt_window += 1
                if blocks_pred_win:
                    items_blocking_pred_window += 1

                # 2. Back-edge contact measurement (back_to_wall items)
                if it.placement_rule == "back_to_wall" and len(it.polygon_coords) >= 4:
                    c_br = it.polygon_coords[1]
                    c_bl = it.polygon_coords[2]
                    back_line = LineString([c_br, c_bl])
                    if back_line.length > 1e-4:
                        pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                        
                        pred_contact_pts = 0
                        gt_contact_pts = 0
                        
                        pred_dists = []
                        gt_dists = []

                        for pt in pts_sample:
                            ix = int(round(pt.x / scale_m))
                            iy = int(round(pt.y / scale_m))
                            
                            # Pred dist
                            if 0 <= ix < w and 0 <= iy < h:
                                dp = pred_dist_m[iy, ix]
                            else:
                                dp = 99.0
                            pred_dists.append(dp)
                            if dp <= 0.10:
                                pred_contact_pts += 1

                            # GT dist
                            if 0 <= ix < w and 0 <= iy < h:
                                dg = gt_dist_m[iy, ix]
                            else:
                                dg = 99.0
                            gt_dists.append(dg)
                            if dg <= 0.10:
                                gt_contact_pts += 1

                        p_ratio = pred_contact_pts / len(pts_sample)
                        g_ratio = gt_contact_pts / len(pts_sample)

                        pred_contacts.append(p_ratio)
                        gt_contacts.append(g_ratio)

                        items_detail.append({
                            "plan": fname,
                            "room_type": norm_type,
                            "item": it.name,
                            "pred_contact": round(p_ratio, 3),
                            "gt_contact": round(g_ratio, 3),
                            "pred_med_dist": round(float(np.median(pred_dists)), 3),
                            "gt_med_dist": round(float(np.median(gt_dists)), 3),
                            "blocks_gt_win": blocks_gt_win,
                            "blocks_pred_win": blocks_pred_win
                        })

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(heldout_files)} plans...")

    dt_total = time.perf_counter() - t0_all
    print(f"\nCompleted in {dt_total:.2f}s.")

    # Calculate distributions
    def calc_stats(arr):
        arr = np.array(arr) * 100.0
        return {
            "count": len(arr),
            "share_ge_80": round(float((arr >= 80.0).sum() / len(arr) * 100.0), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "p10": round(float(np.percentile(arr, 10)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "median": round(float(np.median(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    pred_stats = calc_stats(pred_contacts)
    gt_stats = calc_stats(gt_contacts)

    gt_win_block_pct = (items_blocking_gt_window / total_placed_items * 100.0) if total_placed_items > 0 else 0.0
    pred_win_block_pct = (items_blocking_pred_window / total_placed_items * 100.0) if total_placed_items > 0 else 0.0

    print("\n" + "=" * 80)
    print("RESULTS: BACK-EDGE CONTACT DISTRIBUTIONS")
    print("=" * 80)
    print(f"Total placed items evaluated: {total_placed_items}")
    print(f"Total back-to-wall items:     {len(pred_contacts)}")
    print("\n1. PREDICTED WALL MASK CONTACT (Model's Distance Map):")
    for k, v in pred_stats.items():
        print(f"   {k:15s}: {v}")

    print("\n2. GROUND-TRUTH WALL MASK CONTACT (True Structural Walls):")
    for k, v in gt_stats.items():
        print(f"   {k:15s}: {v}")

    print("\n" + "=" * 80)
    print("RESULTS: WINDOW BLOCKING INCIDENCE")
    print("=" * 80)
    print(f"Items blocking GT Window Mask:   {items_blocking_gt_window} / {total_placed_items} ({gt_win_block_pct:.2f}%)")
    print(f"Items blocking Predicted Window: {items_blocking_pred_window} / {total_placed_items} ({pred_win_block_pct:.2f}%)")

    # Output JSON
    output_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "heldout_contact_and_window_blocking.json"
    
    out_data = {
        "dataset": "CubiCasa5k validation plans 100-200 (held-out)",
        "plans_count": len(heldout_files),
        "total_placed_items": total_placed_items,
        "back_to_wall_items_count": len(pred_contacts),
        "predicted_wall_mask_contact_distribution": pred_stats,
        "ground_truth_wall_mask_contact_distribution": gt_stats,
        "window_blocking": {
            "gt_window_blocking_count": items_blocking_gt_window,
            "gt_window_blocking_pct": round(gt_win_block_pct, 2),
            "pred_window_blocking_count": items_blocking_pred_window,
            "pred_window_blocking_pct": round(pred_win_block_pct, 2)
        },
        "sample_items": items_detail[:50]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved detailed analysis to {out_path}")

if __name__ == "__main__":
    run_evaluation()
