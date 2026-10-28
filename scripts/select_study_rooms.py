import os
import sys
import glob
import io
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.design.engine import RoomDesignEngine
from backend.app.floorplan_engine import FloorplanPipeline
from src.design.types import MIN_PLAUSIBLE_ROOM_AREAS

def main():
    print("=" * 80)
    print("STEP 1: SELECTING ~30 PLAUSIBLE ROOMS FROM VALIDATION PLANS 300-400")
    print("=" * 80)

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    plans_300_400 = all_val[300:400]
    print(f"Total available plans in untouched range (indices 300-399): {len(plans_300_400)}")

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    target_types = ["bedroom", "living_room", "kitchen", "bathroom"]
    target_per_type = 8  # 8 * 4 = 32 rooms total (~30)

    selected_rooms = {t: [] for t in target_types}
    excluded_fragments = {t: 0 for t in target_types}
    total_detected_rooms = {t: 0 for t in target_types}
    plans_scanned_count = 0

    import scipy.ndimage as ndi

    for p_idx, p_path in enumerate(plans_300_400):
        val_idx = 300 + p_idx
        fname = os.path.basename(p_path)

        # Stop early if we have enough rooms of all types
        if all(len(selected_rooms[t]) >= target_per_type for t in target_types):
            break

        plans_scanned_count += 1
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
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            if norm_type not in target_types:
                continue

            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue

            poly = Polygon(pts)
            if not poly.is_valid or poly.area < 0.5:
                continue

            total_detected_rooms[norm_type] += 1
            min_a = MIN_PLAUSIBLE_ROOM_AREAS[norm_type]

            if poly.area < min_a:
                excluded_fragments[norm_type] += 1
                continue

            # Check if we already have enough of this type
            if len(selected_rooms[norm_type]) >= target_per_type:
                continue

            # Run engine placement
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

            if res.is_valid and len(res.items) >= 2:
                selected_rooms[norm_type].append({
                    "val_plan_index": val_idx,
                    "plan_filename": fname,
                    "room_index": r_idx,
                    "room_type": norm_type,
                    "area_m2": round(poly.area, 2),
                    "min_threshold_m2": min_a,
                    "polygon_meters": pts,
                    "doors": doors,
                    "windows": windows,
                    "scale_m_per_px": scale_m,
                    "engine_result": res.to_dict(),
                    "qualification_reason": f"Plausible {norm_type} (area {poly.area:.2f} m2 >= {min_a} m2 threshold, closed valid polygon, zero collisions, {len(res.items)} items placed)"
                })
                print(f"Selected {norm_type.upper():12s} from plan {val_idx} ({fname}): {poly.area:.2f} m2, {len(res.items)} items. Total {norm_type}: {len(selected_rooms[norm_type])}/{target_per_type}")

    total_selected = sum(len(v) for v in selected_rooms.values())
    total_detected = sum(total_detected_rooms.values())
    total_excluded = sum(excluded_fragments.values())

    print("\n" + "=" * 80)
    print("SELECTION SUMMARY:")
    print(f"Plans scanned (indices 300+): {plans_scanned_count}")
    print(f"Total detected rooms of 4 types: {total_detected}")
    print(f"Total excluded fragments: {total_excluded} ({total_excluded/max(total_detected,1)*100:.1f}%)")
    for t in target_types:
        det = total_detected_rooms[t]
        exc = excluded_fragments[t]
        sel = len(selected_rooms[t])
        print(f"  {t:12s}: {sel} selected | {exc}/{det} fragments excluded ({exc/max(det,1)*100:.1f}%) | threshold: {MIN_PLAUSIBLE_ROOM_AREAS[t]} m2")
    print(f"TOTAL SELECTED ROOMS: {total_selected}")
    print("=" * 80)

    out_file = Path("outputs/blind_study/selected_rooms.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "plans_scanned_count": plans_scanned_count,
            "val_index_range": [300, 300 + plans_scanned_count - 1],
            "total_detected_rooms": total_detected_rooms,
            "excluded_fragments": excluded_fragments,
            "selected_rooms": selected_rooms
        }, f, indent=2)
    print(f"Saved selected rooms to {out_file}")

if __name__ == "__main__":
    main()
