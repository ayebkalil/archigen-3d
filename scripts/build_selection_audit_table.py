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

def build_full_audit_table():
    # Load selected_rooms.json
    with open("outputs/blind_study/selected_rooms.json") as f:
        selected_data = json.load(f)

    # Set of selected room identifiers: (val_plan_index, room_index)
    selected_keys = set()
    selected_by_type = {t: 0 for t in ["bedroom", "living_room", "kitchen", "bathroom"]}
    for rtype, rlist in selected_data["selected_rooms"].items():
        for r in rlist:
            selected_keys.add((r["val_plan_index"], r["room_index"]))

    print(f"Total selected in JSON: {len(selected_keys)}")

    all_val = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))
    plans_300_310 = all_val[300:311]

    pipeline = FloorplanPipeline.get_instance()
    engine = RoomDesignEngine()

    stage3_thresh = MIN_PLAUSIBLE_ROOM_AREAS
    furnishable_types = set(stage3_thresh.keys())

    all_detected = []
    
    # We will simulate the sequential selection process as done by select_study_rooms.py
    # target_per_type = 8
    running_counts = {t: 0 for t in furnishable_types}

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
            raw_type = r.get("room_type", "other")
            norm_type = raw_type.lower().replace(" ", "_")
            pts = r.get("polygon_meters", [])

            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            area = round(poly.area, 2)

            room_entry = {
                "val_plan_index": val_idx,
                "plan_filename": fname,
                "room_index": r_idx,
                "raw_type": raw_type,
                "norm_type": norm_type,
                "area_m2": area,
                "status": None,
                "details": ""
            }

            if norm_type not in furnishable_types:
                room_entry["status"] = "not furnishable type"
                room_entry["details"] = f"Type '{raw_type}' not in {sorted(furnishable_types)}"
                all_detected.append(room_entry)
                continue

            min_a = stage3_thresh[norm_type]
            if area < min_a:
                room_entry["status"] = "fragment"
                room_entry["details"] = f"Area {area:.2f} m2 < {min_a:.1f} m2 threshold"
                all_detected.append(room_entry)
                continue

            # Run engine placement to verify
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

            is_in_selected_json = (val_idx, r_idx) in selected_keys

            if not res.is_valid:
                room_entry["status"] = "engine failed"
                room_entry["details"] = f"Engine invalid: {res.failure_reason}"
            elif len(res.items) < 2:
                room_entry["status"] = "engine failed"
                room_entry["details"] = f"Engine placed only {len(res.items)} items (< 2)"
            elif running_counts[norm_type] >= 8 and not is_in_selected_json:
                room_entry["status"] = "cap reached"
                room_entry["details"] = f"Cap of 8 reached earlier for {norm_type}"
            else:
                if is_in_selected_json:
                    room_entry["status"] = "selected"
                    room_entry["details"] = f"Selected #{running_counts[norm_type]+1} for {norm_type} ({len(res.items)} items)"
                    running_counts[norm_type] += 1
                else:
                    room_entry["status"] = "cap reached"
                    room_entry["details"] = f"Valid but category cap already filled"

            all_detected.append(room_entry)

    # Summarize
    status_counts = {}
    for r in all_detected:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    print("\n" + "=" * 80)
    print("STATUS SUMMARY OF ALL DETECTED ROOMS (PLANS 300-310):")
    print(f"Total detected rooms: {len(all_detected)}")
    for st, count in sorted(status_counts.items()):
        print(f"  {st:22s}: {count:2d}")
    print("=" * 80)

    # Save to json for reference
    with open("outputs/blind_study/data/selection_audit_full_table.json", "w") as f:
        json.dump({
            "total_detected": len(all_detected),
            "status_counts": status_counts,
            "rooms": all_detected
        }, f, indent=2)

    # Print the table
    print(f"{'Plan Index':<10} {'Filename':<35} {'Room':<6} {'Type':<14} {'Area (m2)':<10} {'Status':<22} {'Details'}")
    print("-" * 125)
    for r in all_detected:
        print(f"{r['val_plan_index']:<10} {r['plan_filename']:<35} {r['room_index']:<6} {r['norm_type']:<14} {r['area_m2']:<10.2f} {r['status']:<22} {r['details']}")

if __name__ == "__main__":
    build_full_audit_table()
