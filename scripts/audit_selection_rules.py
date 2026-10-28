import json
import glob
import io
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.types import MIN_PLAUSIBLE_ROOM_AREAS
from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline

def run_audit():
    print("=== AUDIT OF SELECTION RULES AND THRESHOLDS ===")
    print(f"Stage 3 MIN_PLAUSIBLE_ROOM_AREAS in src/design/types.py: {MIN_PLAUSIBLE_ROOM_AREAS}")
    
    # 1. Read existing selected_rooms.json
    with open("outputs/blind_study/selected_rooms.json") as f:
        selected_data = json.load(f)
    
    print("\n--- Currently Selected Rooms in outputs/blind_study/selected_rooms.json ---")
    stage3_thresh = {"bedroom": 5.0, "living_room": 6.0, "kitchen": 3.0, "bathroom": 2.0}
    stage4_stated_thresh = {"bedroom": 6.0, "living_room": 10.0, "kitchen": 4.0, "bathroom": 2.5}
    
    for rtype, rlist in selected_data["selected_rooms"].items():
        print(f"\nType: {rtype} ({len(rlist)} selected)")
        areas = [r["area_m2"] for r in rlist]
        thresh_recorded = [r["min_threshold_m2"] for r in rlist]
        print(f"  Recorded min_threshold_m2 in JSON: {set(thresh_recorded)}")
        print(f"  Min area: {min(areas):.2f} m2, Max area: {max(areas):.2f} m2")
        print(f"  All areas: {sorted(areas)}")
        
        # Check if any selected room is between stage3_thresh and stage4_stated_thresh
        between = [a for a in areas if stage3_thresh[rtype] <= a < stage4_stated_thresh[rtype]]
        print(f"  Selected rooms with area in [{stage3_thresh[rtype]}, {stage4_stated_thresh[rtype]}): {between}")

    # 2. Rescan the 11 plans (300 to 310) to see ALL detected rooms and if any were dropped because the engine failed on it
    print("\n--- Rescanning Plans 300 to 310 to inspect ALL detected rooms and engine status ---")
    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    plans_300_310 = all_val[300:311]
    
    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()
    
    all_detected_rooms = []
    
    for p_idx, p_path in enumerate(plans_300_310):
        val_idx = 300 + p_idx
        fname = os.path.basename(p_path)
        d = np.load(p_path)
        img_np = d["image"] if "image" in d else np.stack([d["mask"] * 50] * 3, axis=-1)
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format="PNG")
        proc = pipeline.process_image(buf.getvalue(), front_only=True)
        layout = proc["layout"]
        scale_m = layout["metadata"].get("scale_meters_per_pixel", 0.03)
        raw_walls = layout.get("walls", [])
        doors = layout.get("doors", [])
        windows = layout.get("windows", [])
        pred_wall_mask = proc.get("wall_mask")
        if pred_wall_mask is None or pred_wall_mask.max() == 0:
            from src.design.edge_classifier import EdgeClassifier
            pred_wall_mask = EdgeClassifier.rasterize_wall_segments(raw_walls, scale_m)
            
        for r_idx, r in enumerate(layout.get("rooms", [])):
            norm_type = r.get("room_type", "other").lower().replace(" ", "_")
            if norm_type not in stage3_thresh:
                continue
            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            area = round(poly.area, 2)
            
            # Test engine placement regardless of area
            res = engine.design_room(
                room_polygon_meters=pts,
                room_type=norm_type,
                room_id=f"val_{val_idx}_{fname}_{r_idx}_{norm_type}",
                wall_mask=pred_wall_mask,
                scale_m_per_px=scale_m,
                wall_segments=raw_walls,
                doors=doors,
                windows=windows,
                preset="contemporary_minimal",
                seed=42
            )
            
            all_detected_rooms.append({
                "val_plan_index": val_idx,
                "plan_filename": fname,
                "room_index": r_idx,
                "room_type": norm_type,
                "area_m2": area,
                "engine_is_valid": res.is_valid,
                "engine_item_count": len(res.items),
                "engine_status": res.status if hasattr(res, 'status') else None,
                "engine_error": getattr(res, 'error_message', None)
            })

    print(f"\nTotal rooms detected in plans 300-310 for the 4 target types: {len(all_detected_rooms)}")
    
    # Check for rooms between the two sets of thresholds
    print("\n--- Analysis of thresholds on ALL detected rooms (plans 300-310) ---")
    for rtype in ["bedroom", "living_room", "kitchen", "bathroom"]:
        rooms_of_type = [r for r in all_detected_rooms if r["room_type"] == rtype]
        t3 = stage3_thresh[rtype]
        t4 = stage4_stated_thresh[rtype]
        below_t3 = [r for r in rooms_of_type if r["area_m2"] < t3]
        between = [r for r in rooms_of_type if t3 <= r["area_m2"] < t4]
        above_t4 = [r for r in rooms_of_type if r["area_m2"] >= t4]
        
        print(f"\nType: {rtype} (Total detected: {len(rooms_of_type)})")
        print(f"  Below Stage 3 threshold (< {t3} m2): {len(below_t3)} rooms")
        for r in below_t3:
            print(f"    Plan {r['val_plan_index']} ({r['plan_filename']}) room {r['room_index']}: {r['area_m2']} m2 (engine_valid={r['engine_is_valid']})")
        print(f"  Between thresholds [{t3}, {t4}) m2: {len(between)} rooms")
        for r in between:
            print(f"    Plan {r['val_plan_index']} ({r['plan_filename']}) room {r['room_index']}: {r['area_m2']} m2 (engine_valid={r['engine_is_valid']}, items={r['engine_item_count']})")
        print(f"  Above higher threshold (>= {t4} m2): {len(above_t4)} rooms")
        
    print("\n--- Engine Failure Check ---")
    engine_failures = [r for r in all_detected_rooms if not r["engine_is_valid"]]
    print(f"Rooms where engine was NOT valid: {len(engine_failures)}")
    for r in engine_failures:
        print(f"  Plan {r['val_plan_index']} ({r['plan_filename']}) room {r['room_index']} ({r['room_type']}): area={r['area_m2']} m2, items={r['engine_item_count']}, status={r['engine_status']}")

    # Also check if any room >= stage3_thresh had engine failure or <2 items:
    plausible_failures = [r for r in all_detected_rooms if r["area_m2"] >= stage3_thresh[r["room_type"]] and (not r["engine_is_valid"] or r["engine_item_count"] < 2)]
    print(f"\nPlausible rooms (>= Stage 3 threshold) where engine failed or produced < 2 items: {len(plausible_failures)}")
    for r in plausible_failures:
        print(f"  Plan {r['val_plan_index']} ({r['plan_filename']}) room {r['room_index']} ({r['room_type']}): area={r['area_m2']} m2, valid={r['engine_is_valid']}, items={r['engine_item_count']}")

if __name__ == "__main__":
    run_audit()
