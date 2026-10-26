import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import scipy.ndimage as ndi
from shapely.geometry import Polygon, LineString

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from src.design.edge_classifier import EdgeClassifier
from src.design.rules import compute_footprint_polygon
from backend.app.floorplan_engine import FloorplanPipeline

def run_wall_mask_comparison():
    print("=" * 80)
    print("VALIDATING PARTITION FALLBACK & WALL MASK CLASSIFICATION")
    print("=" * 80)

    val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))[:100]
    pipeline = FloorplanPipeline.get_instance()

    item_min_widths = {
        "bedroom": 1.60,
        "living_room": 2.20,
        "kitchen": 1.20,
        "bathroom": 0.80
    }

    # Tracking for Method A (simplified walls list + partition fallback)
    stats_method_a = {
        "total_rooms": 0,
        "valid_rooms": 0,
        "by_type": {"bedroom": {"total": 0, "valid": 0}, "living_room": {"total": 0, "valid": 0}, "kitchen": {"total": 0, "valid": 0}, "bathroom": {"total": 0, "valid": 0}},
        "placed_items": []
    }

    # Tracking for Method B (classifying edges directly from wall_mask raster)
    stats_method_b = {
        "total_rooms": 0,
        "valid_rooms": 0,
        "by_type": {"bedroom": {"total": 0, "valid": 0}, "living_room": {"total": 0, "valid": 0}, "kitchen": {"total": 0, "valid": 0}, "bathroom": {"total": 0, "valid": 0}},
        "placed_items": []
    }

    # Tracking back-edge wall contact metrics
    items_contact_a = []
    items_contact_b = []
    non_wall_rooms_a = []
    non_wall_rooms_b = []

    engine = RoomDesignEngine()

    for fidx, fpath in enumerate(val_files):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        
        gt_mask = d["mask"]  # GT 4-class mask (1 is wall)
        gt_wall_bin = (gt_mask == 1).astype(np.uint8)
        
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        pred_layout = proc["layout"]
        scale_m = pred_layout["metadata"].get("scale_meters_per_pixel", 0.03)

        # Get the predicted wall mask from UNet++ struct_pred
        # struct_pred: 1 is wall_interior, 2 is wall_exterior
        # We can extract predicted wall mask from pipeline or construct it
        raw_walls = pred_layout.get("walls", [])
        doors = pred_layout.get("doors", [])
        windows = pred_layout.get("windows", [])

        # Reconstruct high-resolution predicted wall binary raster from mask_4class
        # Or from the pipeline run
        pred_wall_bin = np.zeros((512, 512), dtype=np.uint8)
        for w in raw_walls:
            pts_px = w.get("polygon_px", [])
            if len(pts_px) >= 3:
                import cv2
                cv2.fillPoly(pred_wall_bin, [np.array(pts_px, dtype=np.int32)], 1)

        # Precompute distance transforms
        gt_dist_m = ndi.distance_transform_edt(1 - gt_wall_bin) * scale_m
        pred_dist_m = ndi.distance_transform_edt(1 - pred_wall_bin) * scale_m

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in item_min_widths:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            stats_method_a["total_rooms"] += 1
            stats_method_a["by_type"][norm_type]["total"] += 1

            stats_method_b["total_rooms"] += 1
            stats_method_b["by_type"][norm_type]["total"] += 1

            room_uid = f"{fname}_{r_idx}_{norm_type}"

            # ---------------------------------------------------------
            # METHOD A: using wall_segments only (with partition fallback)
            # ---------------------------------------------------------
            res_a = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=f"A_{room_uid}",
                wall_mask=None,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                seed=42
            )

            if res_a.is_valid:
                stats_method_a["valid_rooms"] += 1
                stats_method_a["by_type"][norm_type]["valid"] += 1

                for item in res_a.items:
                    if item.placement_rule != "back_to_wall":
                        continue
                    cx, cy = item.position_m
                    w, d, h = item.dimensions_m
                    poly_corners = compute_footprint_polygon(cx, cy, w, d, item.rotation_deg)
                    # Back edge is between c_br (index 1) and c_bl (index 2)
                    c_br = poly_corners[1]
                    c_bl = poly_corners[2]
                    back_line = LineString([c_br, c_bl])

                    # Sample 15 points along back edge
                    pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                    
                    gt_dists = []
                    pred_dists = []
                    for pt in pts_sample:
                        ix = int(round(pt.x / scale_m))
                        iy = int(round(pt.y / scale_m))
                        if 0 <= ix < 512 and 0 <= iy < 512:
                            gt_dists.append(gt_dist_m[iy, ix])
                            pred_dists.append(pred_dist_m[iy, ix])
                        else:
                            gt_dists.append(99.0)
                            pred_dists.append(99.0)

                    share_gt = sum(1 for d in gt_dists if d <= 0.10) / len(gt_dists)
                    share_pred = sum(1 for d in pred_dists if d <= 0.10) / len(pred_dists)

                    items_contact_a.append({
                        "room_uid": room_uid,
                        "room_type": norm_type,
                        "item_name": item.name,
                        "share_within_10cm_gt": share_gt,
                        "share_within_10cm_pred": share_pred,
                        "median_dist_gt": float(np.median(gt_dists)),
                        "median_dist_pred": float(np.median(pred_dists))
                    })

                    if share_gt < 0.50:
                        non_wall_rooms_a.append({
                            "plan": fname,
                            "room_id": f"room_{r_idx}",
                            "room_type": norm_type,
                            "item": item.name,
                            "share_gt": share_gt,
                            "median_dist_gt_m": round(float(np.median(gt_dists)), 3)
                        })

            # ---------------------------------------------------------
            # METHOD B: passing wall_mask directly (preferring raster)
            # ---------------------------------------------------------
            res_b = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=f"B_{room_uid}",
                wall_mask=gt_wall_bin, # Pass GT or predicted wall mask
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                seed=42
            )

            if res_b.is_valid:
                stats_method_b["valid_rooms"] += 1
                stats_method_b["by_type"][norm_type]["valid"] += 1

                for item in res_b.items:
                    if item.placement_rule != "back_to_wall":
                        continue
                    cx, cy = item.position_m
                    w, d, h = item.dimensions_m
                    poly_corners = compute_footprint_polygon(cx, cy, w, d, item.rotation_deg)
                    c_br = poly_corners[1]
                    c_bl = poly_corners[2]
                    back_line = LineString([c_br, c_bl])

                    pts_sample = [back_line.interpolate(frac, normalized=True) for frac in np.linspace(0.05, 0.95, 15)]
                    gt_dists = []
                    for pt in pts_sample:
                        ix = int(round(pt.x / scale_m))
                        iy = int(round(pt.y / scale_m))
                        if 0 <= ix < 512 and 0 <= iy < 512:
                            gt_dists.append(gt_dist_m[iy, ix])
                        else:
                            gt_dists.append(99.0)

                    share_gt = sum(1 for d in gt_dists if d <= 0.10) / len(gt_dists)
                    items_contact_b.append({
                        "room_uid": room_uid,
                        "room_type": norm_type,
                        "item_name": item.name,
                        "share_within_10cm_gt": share_gt
                    })

    # Summary calculations
    gt_shares_a = [x["share_within_10cm_gt"] for x in items_contact_a]
    pred_shares_a = [x["share_within_10cm_pred"] for x in items_contact_a]

    print("\n=== METHOD A: PARTITION FALLBACK CONTACT VALIDATION ===")
    print(f"Total back_to_wall items evaluated: {len(gt_shares_a)}")
    print("Against Ground-Truth Wall Mask:")
    print(f"  Mean share within 0.10m: {np.mean(gt_shares_a)*100:.2f}%")
    print(f"  Median share within 0.10m: {np.median(gt_shares_a)*100:.2f}%")
    print(f"  25th percentile: {np.percentile(gt_shares_a, 25)*100:.2f}%")
    print(f"  75th percentile: {np.percentile(gt_shares_a, 75)*100:.2f}%")
    print(f"  Items with 100% back edge within 0.10m: {sum(1 for s in gt_shares_a if s >= 0.99)} / {len(gt_shares_a)} ({sum(1 for s in gt_shares_a if s >= 0.99)/len(gt_shares_a)*100:.2f}%)")
    print(f"  Items with >= 80% back edge within 0.10m: {sum(1 for s in gt_shares_a if s >= 0.80)} / {len(gt_shares_a)} ({sum(1 for s in gt_shares_a if s >= 0.80)/len(gt_shares_a)*100:.2f}%)")
    print(f"  Items with < 50% back edge within 0.10m (non-wall edges): {len(non_wall_rooms_a)} / {len(gt_shares_a)} ({len(non_wall_rooms_a)/len(gt_shares_a)*100:.2f}%)")

    print("\nAgainst Predicted Wall Mask:")
    print(f"  Mean share within 0.10m: {np.mean(pred_shares_a)*100:.2f}%")
    print(f"  Median share within 0.10m: {np.median(pred_shares_a)*100:.2f}%")

    print("\nRooms where items sit against non-wall edges (share_gt < 50%):")
    for r in non_wall_rooms_a[:10]:
        print(f"  Plan: {r['plan']} | Room: {r['room_id']} ({r['room_type']}) | Item: {r['item']} | Share on Wall: {r['share_gt']*100:.1f}% | Median Dist: {r['median_dist_gt_m']}m")

    print("\n=== VALIDITY COMPARISON: METHOD A vs METHOD B ===")
    print(f"Method A (Simplified walls list + Fallback): {stats_method_a['valid_rooms']} / {stats_method_a['total_rooms']} ({stats_method_a['valid_rooms']/stats_method_a['total_rooms']*100:.2f}%)")
    for k, v in stats_method_a["by_type"].items():
        print(f"  {k:12s}: {v['valid']} / {v['total']} ({v['valid']/v['total']*100:.2f}%)")

    print(f"\nMethod B (Direct Wall Mask Classification): {stats_method_b['valid_rooms']} / {stats_method_b['total_rooms']} ({stats_method_b['valid_rooms']/stats_method_b['total_rooms']*100:.2f}%)")
    for k, v in stats_method_b["by_type"].items():
        print(f"  {k:12s}: {v['valid']} / {v['total']} ({v['valid']/v['total']*100:.2f}%)")

    out_data = {
        "method_a": {
            "total_items": len(gt_shares_a),
            "mean_share_gt_10cm": float(np.mean(gt_shares_a)),
            "median_share_gt_10cm": float(np.median(gt_shares_a)),
            "p25_share_gt_10cm": float(np.percentile(gt_shares_a, 25)),
            "p75_share_gt_10cm": float(np.percentile(gt_shares_a, 75)),
            "items_ge_80pct_gt": sum(1 for s in gt_shares_a if s >= 0.80),
            "items_lt_50pct_gt": len(non_wall_rooms_a),
            "non_wall_rooms_sample": non_wall_rooms_a[:15],
            "validity": stats_method_a
        },
        "method_b": {
            "validity": stats_method_b
        }
    }

    out_file = PROJECT_ROOT / "outputs" / "design_stage3" / "wall_contact_and_mask_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved analysis to: {out_file}")

if __name__ == "__main__":
    run_wall_mask_comparison()
