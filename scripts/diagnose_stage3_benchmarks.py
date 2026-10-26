import os
import sys
import json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.design.engine import RoomDesignPlacementEngine
from src.design.rules import RoomRuleEngine
from src.design.types import UsableWallSegment

def main():
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "data" / "cubicasa_cache"
    val_files = sorted(list(cache_dir.glob("colorful_*.npz")))[:100]
    
    print(f"Total validation plans found: {len(val_files)}")
    
    catalog_path = repo_root / "src" / "design" / "catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    engine = RoomDesignPlacementEngine(catalog_path=str(catalog_path))
    
    # -------------------------------------------------------------
    # TASK 1: Re-run predicted-mask benchmark with living-room fix
    # -------------------------------------------------------------
    # Room classes to benchmark
    target_classes = ["bedroom", "kitchen", "bathroom", "living_room"]
    
    benchmark_results = {
        cls_name: {
            "total_rooms": 0,
            "valid_rooms": 0,
            "no_usable_wall": 0,
            "too_small": 0,
            "validation_failed": 0,
            "door_blocked": 0,
            "other_failures": 0,
            "collisions_count": 0,
            "total_overlap_area_m2": 0.0,
            "door_clearance_violations": 0
        }
        for cls_name in target_classes
    }
    
    # Track extra predicted rooms vs GT
    gt_rooms_count = 0
    gt_room_areas = []
    pred_rooms_count = 0
    pred_room_areas = []
    
    # Track bathroom details
    bathroom_stats = {
        "total": 0,
        "valid": 0,
        "too_small": 0,
        "no_usable_wall": 0,
        "failed_areas": [],
        "failed_wall_lengths": [],
        "failed_segment_counts": []
    }

    for npz_path in val_files:
        data = np.load(npz_path, allow_pickle=True)
        pred_rooms = data.get("pred_rooms", [])
        pred_walls = data.get("pred_walls", [])
        pred_doors = data.get("pred_doors", [])
        pred_windows = data.get("pred_windows", [])
        
        gt_rooms = data.get("rooms", [])
        for gtr in gt_rooms:
            gt_rooms_count += 1
            pts = gtr.get("points", [])
            if len(pts) >= 3:
                gt_room_areas.append(Polygon(pts).area)
        
        for r in pred_rooms:
            pred_rooms_count += 1
            pts = r.get("points", [])
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            area = poly.area
            pred_room_areas.append(area)
            
            raw_type = r.get("class_name", r.get("type", "other")).lower().strip()
            # Normalize room type properly
            norm_type = None
            if "bed" in raw_type: norm_type = "bedroom"
            elif "liv" in raw_type: norm_type = "living_room"
            elif "kitch" in raw_type: norm_type = "kitchen"
            elif "bath" in raw_type or "wc" in raw_type: norm_type = "bathroom"
            
            if norm_type is None:
                continue
                
            stats = benchmark_results[norm_type]
            stats["total_rooms"] += 1
            
            # Run placement engine
            res = engine.place_room(
                room_polygon=pts,
                room_type=norm_type,
                preset="contemporary_minimal",
                wall_segments=pred_walls,
                door_segments=pred_doors,
                window_segments=pred_windows,
                seed=42
            )
            
            if norm_type == "bathroom":
                bathroom_stats["total"] += 1
                if res.is_valid:
                    bathroom_stats["valid"] += 1
                else:
                    if res.failure_reason == "TOO_SMALL":
                        bathroom_stats["too_small"] += 1
                    elif res.failure_reason == "NO_USABLE_WALL":
                        bathroom_stats["no_usable_wall"] += 1
                    bathroom_stats["failed_areas"].append(area)
                    bathroom_stats["failed_segment_counts"].append(res.metadata.usable_wall_segments_count)
            
            if res.is_valid:
                stats["valid_rooms"] += 1
                if res.validation.overlap_count > 0:
                    stats["collisions_count"] += res.validation.overlap_count
                if res.validation.door_clearance_violation:
                    stats["door_clearance_violations"] += 1
            else:
                reason = res.failure_reason
                if reason == "NO_USABLE_WALL":
                    stats["no_usable_wall"] += 1
                elif reason == "TOO_SMALL":
                    stats["too_small"] += 1
                elif reason == "VALIDATION_FAILED":
                    stats["validation_failed"] += 1
                elif reason == "DOOR_BLOCKED":
                    stats["door_blocked"] += 1
                else:
                    stats["other_failures"] += 1

    # Save benchmark json
    out_dir = repo_root / "outputs" / "design_stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predicted_benchmark_all4_rooms.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Saved all 4 room types benchmark to: {out_path}")
    
    print("\n=== BENCHMARK ALL 4 ROOM TYPES (PREDICTED MASKS) ===")
    for k, v in benchmark_results.items():
        tot = v["total_rooms"]
        val = v["valid_rooms"]
        pct = (val / tot * 100) if tot > 0 else 0
        print(f"{k.upper():12s}: Total {tot:3d} | Valid {val:3d} ({pct:5.2f}%) | NO_USABLE_WALL: {v['no_usable_wall']:3d} | TOO_SMALL: {v['too_small']:3d} | OTHER: {v['validation_failed'] + v['door_blocked'] + v['other_failures']:2d}")

    # TASK 3: Extra predicted rooms under 4m2
    print(f"\n=== EXTRA PREDICTED ROOMS ANALYSIS ===")
    print(f"Total GT rooms across 100 plans: {gt_rooms_count}")
    print(f"Total Predicted rooms across 100 plans: {pred_rooms_count}")
    extra_rooms_count = pred_rooms_count - gt_rooms_count
    print(f"Extra predicted rooms: {extra_rooms_count}")
    
    under_4_gt = sum(1 for a in gt_room_areas if a < 4.0)
    under_4_pred = sum(1 for a in pred_room_areas if a < 4.0)
    print(f"GT rooms < 4 m²: {under_4_gt} / {gt_rooms_count} ({under_4_gt/gt_rooms_count*100:.2f}%)")
    print(f"Predicted rooms < 4 m²: {under_4_pred} / {pred_rooms_count} ({under_4_pred/pred_rooms_count*100:.2f}%)")
    print(f"Extra rooms < 4 m² (predicted diff): {under_4_pred - under_4_gt} of the {extra_rooms_count} extra rooms")

    # TASK 2: Bathrooms analysis
    print(f"\n=== BATHROOM FAILURE DIAGNOSTICS ===")
    print(f"Total bathrooms: {bathroom_stats['total']}")
    print(f"Valid: {bathroom_stats['valid']} ({bathroom_stats['valid']/bathroom_stats['total']*100:.2f}%)")
    print(f"TOO_SMALL: {bathroom_stats['too_small']} ({bathroom_stats['too_small']/bathroom_stats['total']*100:.2f}%)")
    print(f"NO_USABLE_WALL: {bathroom_stats['no_usable_wall']} ({bathroom_stats['no_usable_wall']/bathroom_stats['total']*100:.2f}%)")
    
    # Save diagnostic summary
    diag_summary = {
        "benchmark": benchmark_results,
        "extra_rooms": {
            "gt_count": gt_rooms_count,
            "pred_count": pred_rooms_count,
            "extra_count": extra_rooms_count,
            "gt_under_4m2": under_4_gt,
            "pred_under_4m2": under_4_pred,
            "extra_under_4m2": under_4_pred - under_4_gt
        },
        "bathroom_stats": {
            "total": bathroom_stats["total"],
            "valid": bathroom_stats["valid"],
            "too_small": bathroom_stats["too_small"],
            "no_usable_wall": bathroom_stats["no_usable_wall"],
            "failed_areas_summary": {
                "min": float(np.min(bathroom_stats["failed_areas"])) if bathroom_stats["failed_areas"] else 0,
                "median": float(np.median(bathroom_stats["failed_areas"])) if bathroom_stats["failed_areas"] else 0,
                "max": float(np.max(bathroom_stats["failed_areas"])) if bathroom_stats["failed_areas"] else 0,
            }
        }
    }
    with open(out_dir / "stage3_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)

if __name__ == "__main__":
    main()
