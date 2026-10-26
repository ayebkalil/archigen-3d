import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from src.design.edge_classifier import EdgeClassifier
from backend.app.floorplan_engine import FloorplanPipeline

def run_bathroom_and_extra_rooms_analysis():
    val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))[:100]
    print(f"Loaded {len(val_files)} plans for benchmark & diagnostics.")
    
    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()
    
    # 1. Minimum wall length required for bathroom items:
    with open("src/design/catalog.json") as f:
        catalog = json.load(f)["items"]
        
    toilet_dim = catalog["toilet"]["dimensions_m"]
    vanity_dim = catalog["bathroom_vanity"]["dimensions_m"]
    bathtub_dim = catalog["bathtub"]["dimensions_m"]
    
    print("\n--- BATHROOM ITEM DIMENSIONS & WALL REQUIREMENTS ---")
    print(f"Toilet: W={toilet_dim[0]}m, D={toilet_dim[1]}m, H={toilet_dim[2]}m")
    print(f"  Min wall run length required (width + 2*0.15m clearance): {toilet_dim[0] + 0.30:.3f} m")
    print(f"Vanity: W={vanity_dim[0]}m, D={vanity_dim[1]}m, H={vanity_dim[2]}m")
    print(f"  Min wall run length required (width + 0.10m margin): {vanity_dim[0] + 0.10:.3f} m")
    print(f"Bathtub: W={bathtub_dim[0]}m, D={bathtub_dim[1]}m, H={bathtub_dim[2]}m")
    print(f"  Min wall run length required (width + 0.10m margin): {bathtub_dim[0] + 0.10:.3f} m")

    item_min_widths = {
        "bedroom": 1.60,
        "living_room": 2.20,
        "kitchen": 1.20,
        "bathroom": 0.80
    }

    # Tracking for Task 1: 100 plans predicted benchmark with living room
    pred_stats = {
        "total_rooms": 0,
        "eligible_rooms": 0,
        "valid_rooms": 0,
        "by_type": {
            "bedroom": {"total": 0, "valid": 0, "NO_USABLE_WALL": 0, "TOO_SMALL": 0, "VALIDATION_FAILED": 0, "DOOR_BLOCKED": 0, "OTHER": 0},
            "living_room": {"total": 0, "valid": 0, "NO_USABLE_WALL": 0, "TOO_SMALL": 0, "VALIDATION_FAILED": 0, "DOOR_BLOCKED": 0, "OTHER": 0},
            "kitchen": {"total": 0, "valid": 0, "NO_USABLE_WALL": 0, "TOO_SMALL": 0, "VALIDATION_FAILED": 0, "DOOR_BLOCKED": 0, "OTHER": 0},
            "bathroom": {"total": 0, "valid": 0, "NO_USABLE_WALL": 0, "TOO_SMALL": 0, "VALIDATION_FAILED": 0, "DOOR_BLOCKED": 0, "OTHER": 0}
        }
    }

    # Tracking for Task 3: Extra predicted rooms < 4 m2
    gt_total = 0
    gt_under_4 = 0
    gt_areas = []
    
    pred_total = 0
    pred_under_4 = 0
    pred_areas = []

    # Tracking for Task 2: Failed bathrooms wall length distribution
    failed_bathrooms_wall_lengths = []
    failed_bathrooms_longest_seg = []
    failed_bathrooms_areas = []
    failed_bathrooms_reasons = {"TOO_SMALL": 0, "NO_USABLE_WALL": 0, "OTHER": 0}

    for fidx, fpath in enumerate(val_files):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        
        # Count GT rooms
        gt_mask = d["mask"]
        from src.floorplan.vectorizer import FloorplanVectorizer
        vect = FloorplanVectorizer()
        gt_layout = vect.vectorize(gt_mask)
        for gtr in gt_layout.get("rooms", []):
            gt_total += 1
            pts = gtr.get("polygon_meters", [])
            if len(pts) >= 3:
                a = Polygon(pts).area
                gt_areas.append(a)
                if a < 4.0:
                    gt_under_4 += 1

        # Model prediction
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        pred_layout = proc["layout"]
        
        walls = pred_layout.get("walls", [])
        doors = pred_layout.get("doors", [])
        windows = pred_layout.get("windows", [])

        for r_idx, r in enumerate(pred_layout.get("rooms", [])):
            pred_total += 1
            raw_type = r.get("room_type", "other")
            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            area = poly.area
            pred_areas.append(area)
            if area < 4.0:
                pred_under_4 += 1

            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in item_min_widths:
                continue

            pred_stats["eligible_rooms"] += 1
            t_entry = pred_stats["by_type"][norm_type]
            t_entry["total"] += 1

            room_uid = f"pred_{fname}_{r_idx}_{norm_type}"
            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=room_uid,
                wall_segments=walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )

            if res.is_valid:
                pred_stats["valid_rooms"] += 1
                t_entry["valid"] += 1
            else:
                reason = res.failure_reason or "OTHER"
                if reason in t_entry:
                    t_entry[reason] += 1
                else:
                    t_entry["OTHER"] += 1

                # If bathroom failed, record wall lengths
                if norm_type == "bathroom":
                    failed_bathrooms_reasons[reason] = failed_bathrooms_reasons.get(reason, 0) + 1
                    failed_bathrooms_areas.append(area)
                    
                    # Extract usable wall segments for this room
                    _, usable_segs, _ = engine.classifier.classify_and_snap_edges(
                        room_poly=poly,
                        wall_mask=None,
                        scale_m_per_px=pred_layout["metadata"].get("scale_meters_per_pixel", 0.03),
                        wall_segments=walls,
                        doors=doors,
                        windows=windows
                    )
                    lens = [s.length_m for s in usable_segs]
                    failed_bathrooms_wall_lengths.extend(lens)
                    failed_bathrooms_longest_seg.append(max(lens) if lens else 0.0)

    # Save benchmark results
    out_dir = PROJECT_ROOT / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_out = out_dir / "predicted_benchmark_100plans_all4_rooms.json"
    with open(bench_out, "w", encoding="utf-8") as f:
        json.dump(pred_stats, f, indent=2)
    print(f"\nSaved 100-plan benchmark to: {bench_out}")

    print("\n=== TASK 1: PREDICTED-MASK BENCHMARK RESULTS (ALL 4 ROOM TYPES) ===")
    for rtype, data in pred_stats["by_type"].items():
        tot = data["total"]
        val = data["valid"]
        pct = (val / tot * 100) if tot > 0 else 0.0
        print(f"{rtype:12s}: Total {tot:3d} | Valid {val:3d} ({pct:5.2f}%) | NO_USABLE_WALL: {data['NO_USABLE_WALL']:2d} | TOO_SMALL: {data['TOO_SMALL']:2d} | OTHER: {data['VALIDATION_FAILED'] + data['DOOR_BLOCKED'] + data['OTHER']:2d}")
    
    tot_all = sum(d["total"] for d in pred_stats["by_type"].values())
    val_all = sum(d["valid"] for d in pred_stats["by_type"].values())
    print(f"{'OVERALL':12s}: Total {tot_all:3d} | Valid {val_all:3d} ({val_all/tot_all*100:5.2f}%)")

    # Extra predicted rooms analysis
    print("\n=== TASK 3: EXTRA PREDICTED ROOMS UNDER 4 m² ===")
    extra_rooms = pred_total - gt_total
    extra_under_4 = pred_under_4 - gt_under_4
    print(f"Total GT rooms: {gt_total}")
    print(f"Total Predicted rooms: {pred_total}")
    print(f"Extra predicted rooms: {extra_rooms}")
    print(f"GT rooms < 4 m²: {gt_under_4} / {gt_total} ({gt_under_4/gt_total*100:.2f}%)")
    print(f"Predicted rooms < 4 m²: {pred_under_4} / {pred_total} ({pred_under_4/pred_total*100:.2f}%)")
    print(f"Extra rooms < 4 m²: {extra_under_4} / {extra_rooms} ({extra_under_4/extra_rooms*100:.2f}%)")

    # Bathroom distribution
    print("\n=== TASK 2: BATHROOM USABLE WALL RUN LENGTH DISTRIBUTION (FAILED ROOMS) ===")
    print(f"Failed bathrooms total: {len(failed_bathrooms_areas)}")
    print(f"Reasons: {failed_bathrooms_reasons}")
    if failed_bathrooms_wall_lengths:
        arr = np.array(failed_bathrooms_wall_lengths)
        l_arr = np.array(failed_bathrooms_longest_seg)
        print(f"All usable wall segments in failed bathrooms (N={len(arr)}):")
        print(f"  Min: {np.min(arr):.2f} m")
        print(f"  25th percentile: {np.percentile(arr, 25):.2f} m")
        print(f"  Median: {np.median(arr):.2f} m")
        print(f"  75th percentile: {np.percentile(arr, 75):.2f} m")
        print(f"  Max: {np.max(arr):.2f} m")
        print(f"Longest usable wall per failed bathroom (N={len(l_arr)}):")
        print(f"  0 m (no usable segment): {np.sum(l_arr == 0)}")
        print(f"  < 0.77m (cannot fit toilet at 0.77m): {np.sum((l_arr > 0) & (l_arr < 0.77))}")
        print(f"  0.77m - 1.20m (can fit toilet alone): {np.sum((l_arr >= 0.77) & (l_arr < 1.20))}")
        print(f"  >= 1.20m: {np.sum(l_arr >= 1.20)}")

    # Save distribution summary
    dist_data = {
        "bathroom_items": {
            "toilet": {"dim": toilet_dim, "min_wall_m": round(toilet_dim[0] + 0.30, 3)},
            "vanity": {"dim": vanity_dim, "min_wall_m": round(vanity_dim[0] + 0.10, 3)},
            "bathtub": {"dim": bathtub_dim, "min_wall_m": round(bathtub_dim[0] + 0.10, 3)}
        },
        "failed_bathrooms": {
            "total_failed": len(failed_bathrooms_areas),
            "reasons": failed_bathrooms_reasons,
            "wall_length_distribution": {
                "count": len(failed_bathrooms_wall_lengths),
                "min": float(np.min(arr)) if len(failed_bathrooms_wall_lengths) else 0.0,
                "p25": float(np.percentile(arr, 25)) if len(failed_bathrooms_wall_lengths) else 0.0,
                "median": float(np.median(arr)) if len(failed_bathrooms_wall_lengths) else 0.0,
                "p75": float(np.percentile(arr, 75)) if len(failed_bathrooms_wall_lengths) else 0.0,
                "max": float(np.max(arr)) if len(failed_bathrooms_wall_lengths) else 0.0
            },
            "longest_wall_per_room_distribution": {
                "zero_segments": int(np.sum(l_arr == 0)),
                "under_0_77m": int(np.sum((l_arr > 0) & (l_arr < 0.77))),
                "between_0_77m_and_1_20m": int(np.sum((l_arr >= 0.77) & (l_arr < 1.20))),
                "ge_1_20m": int(np.sum(l_arr >= 1.20))
            }
        },
        "extra_rooms_under_4m2": {
            "gt_total": gt_total,
            "gt_under_4": gt_under_4,
            "pred_total": pred_total,
            "pred_under_4": pred_under_4,
            "extra_rooms": extra_rooms,
            "extra_under_4": extra_under_4,
            "extra_under_4_pct": round(extra_under_4 / extra_rooms * 100, 2)
        }
    }
    with open(out_dir / "bathroom_diagnostics_and_extra_rooms.json", "w", encoding="utf-8") as f:
        json.dump(dist_data, f, indent=2)
    print(f"Saved bathroom diagnostics & extra rooms data to: {out_dir / 'bathroom_diagnostics_and_extra_rooms.json'}")

if __name__ == "__main__":
    run_bathroom_and_extra_rooms_analysis()
