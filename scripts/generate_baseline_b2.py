import json
import math
import random
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate

from src.design.types import PlacedItem
from src.design.validator import LayoutValidator

def create_oriented_box(center_x, center_y, width, depth, rotation_deg):
    hw = width / 2.0
    hd = depth / 2.0
    poly = Polygon([
        (-hw, -hd),
        (hw, -hd),
        (hw, hd),
        (-hw, hd)
    ])
    poly_rot = rotate(poly, -rotation_deg, origin=(0, 0))
    poly_trans = translate(poly_rot, xoff=center_x, yoff=center_y)
    return poly_trans

def generate_b2_layout_for_room(room_polygon_pts, engine_items, seed=42, max_room_restarts=100, max_tries_per_item=1000, sort_largest_first=True):
    rng = random.Random(seed)
    room_poly = Polygon(room_polygon_pts)
    min_x, min_y, max_x, max_y = room_poly.bounds

    items_with_idx = list(enumerate(engine_items))
    if sort_largest_first:
        items_with_idx.sort(key=lambda x: x[1]["dimensions_m"][0] * x[1]["dimensions_m"][1], reverse=True)

    for restart in range(max_room_restarts):
        placed_boxes = []
        placed_items = []
        success = True

        for orig_idx, item in items_with_idx:
            w = item["dimensions_m"][0]
            d = item["dimensions_m"][1]
            h = item["dimensions_m"][2]

            item_placed = False
            for t in range(max_tries_per_item):
                cx = rng.uniform(min_x + 0.05, max_x - 0.05)
                cy = rng.uniform(min_y + 0.05, max_y - 0.05)
                rot = rng.uniform(0, 360.0)

                box_poly = create_oriented_box(cx, cy, w, d, rot)

                if not room_poly.contains(box_poly):
                    continue

                overlap = False
                for pb in placed_boxes:
                    if box_poly.intersects(pb):
                        inter = box_poly.intersection(pb)
                        if inter.area > 1e-4:
                            overlap = True
                            break
                if overlap:
                    continue

                placed_boxes.append(box_poly)
                coords = list(box_poly.exterior.coords)[:-1]
                placed_items.append((orig_idx, PlacedItem(
                    item_id=f"b2_{item['item_id']}",
                    catalog_id=item["catalog_id"],
                    name=item["name"],
                    category=item["category"],
                    position_m=(round(cx, 3), round(cy, 3)),
                    rotation_deg=round(rot, 1),
                    dimensions_m=(w, d, h),
                    glb=item["glb"],
                    polygon_coords=coords,
                    placement_rule="b2_random_non_overlapping",
                    wall_distance_cm=99.9,
                    wall_contact_ratio=0.0
                )))
                item_placed = True
                break

            if not item_placed:
                success = False
                break

        if success:
            placed_items.sort(key=lambda x: x[0])
            return [it[1] for it in placed_items], restart + 1

    raise RuntimeError(f"Failed to place all items non-overlapping after {max_room_restarts} restarts!")

def main():
    print("=" * 80)
    print("GENERATING BASELINE B2 (NON-OVERLAPPING RANDOM REJECTION SAMPLING)")
    print("=" * 80)

    with open("outputs/blind_study/selected_rooms.json") as f:
        selected_data = json.load(f)

    flat_rooms = []
    for rtype, rlist in selected_data["selected_rooms"].items():
        for r in rlist:
            flat_rooms.append(r)

    validator = LayoutValidator()

    os.makedirs("outputs/blind_study/keys", exist_ok=True)
    os.makedirs("outputs/blind_study/data", exist_ok=True)
    os.makedirs("outputs/blind_study/ratings/ratings_b1", exist_ok=True)
    os.makedirs("outputs/blind_study/ratings/ratings_b2", exist_ok=True)

    b2_room_entries = []
    validation_summary = {
        "total_evaluated": len(flat_rooms),
        "passed": 0,
        "failed": 0,
        "overlap_failures": 0,
        "door_clearance_failures": 0,
        "wall_contact_failures": 0,
        "bfs_failures": 0,
        "per_type": {}
    }

    # B2 key generator (seed 2027)
    lr_rng_b2 = random.Random(2027)
    keys_b2 = []

    for i, r in enumerate(flat_rooms):
        room_label = f"room_{i+1:02d}"
        rtype = r["room_type"]
        if rtype not in validation_summary["per_type"]:
            validation_summary["per_type"][rtype] = {
                "total": 0, "passed": 0, "failed": 0, "overlap": 0, "door": 0, "wall": 0
            }
        validation_summary["per_type"][rtype]["total"] += 1

        b2_seed = 4200 + i
        items_b2, restarts = generate_b2_layout_for_room(
            r["polygon_meters"], r["engine_result"]["items"], seed=b2_seed, max_room_restarts=100, max_tries_per_item=1000
        )

        room_poly = Polygon(r["polygon_meters"])
        doors = r.get("doors", [])
        door_clearance_zones = []
        for d in doors:
            if "clearance_polygon_meters" in d:
                dp = Polygon(d["clearance_polygon_meters"])
                if dp.is_valid:
                    door_clearance_zones.append(dp)
            elif "position_meters" in d:
                px, py = d["position_meters"]
                door_clearance_zones.append(Point(px, py).buffer(0.85))

        rep = validator.validate(
            room_poly=room_poly,
            items=items_b2,
            door_clearance_zones=door_clearance_zones
        )

        if rep.is_valid:
            validation_summary["passed"] += 1
            validation_summary["per_type"][rtype]["passed"] += 1
        else:
            validation_summary["failed"] += 1
            validation_summary["per_type"][rtype]["failed"] += 1

        if rep.overlap_count > 0:
            validation_summary["overlap_failures"] += 1
            validation_summary["per_type"][rtype]["overlap"] += 1
        if rep.door_clearance_violation:
            validation_summary["door_clearance_failures"] += 1
            validation_summary["per_type"][rtype]["door"] += 1
        if any("WALL CONTACT" in reason for reason in rep.failure_reasons):
            validation_summary["wall_contact_failures"] += 1
            validation_summary["per_type"][rtype]["wall"] += 1
        if not rep.bfs_reachable:
            validation_summary["bfs_failures"] += 1

        # Key randomization for B2: 50% chance engine on Left (A) or Right (B)
        engine_is_a = lr_rng_b2.choice([True, False])
        keys_b2.append({
            "room_id": room_label,
            "room_type": rtype,
            "val_plan_index": r["val_plan_index"],
            "plan_filename": r["plan_filename"],
            "area_m2": r["area_m2"],
            "option_a": "engine" if engine_is_a else "baseline_b2",
            "option_b": "baseline_b2" if engine_is_a else "engine",
            "engine_is": "A" if engine_is_a else "B"
        })

        b2_items_dict = [
            {
                "item_id": it.item_id,
                "catalog_id": it.catalog_id,
                "name": it.name,
                "category": it.category,
                "position_m": [it.position_m[0], it.position_m[1]],
                "rotation_deg": it.rotation_deg,
                "dimensions_m": list(it.dimensions_m),
                "glb": it.glb,
                "placement_rule": it.placement_rule,
                "footprint_polygon": it.polygon_coords
            }
            for it in items_b2
        ]

        b2_room_entries.append({
            "room_id": room_label,
            "room_type": rtype,
            "plan_filename": r["plan_filename"],
            "val_plan_index": r["val_plan_index"],
            "area_m2": r["area_m2"],
            "polygon_meters": r["polygon_meters"],
            "engine_items": r["engine_result"]["items"],
            "b2_seed": b2_seed,
            "b2_restarts": restarts,
            "b2_items": b2_items_dict,
            "b2_validation": {
                "is_valid": rep.is_valid,
                "overlap_count": rep.overlap_count,
                "door_clearance_violation": rep.door_clearance_violation,
                "failure_reasons": rep.failure_reasons
            }
        })

    # Save B2 layout data
    b2_out_file = Path("outputs/blind_study/data/b2_random_layouts.json")
    with open(b2_out_file, "w") as f:
        json.dump({
            "description": "Baseline B2: Random non-overlapping placement constrained to room polygon via rejection sampling",
            "parameters": {
                "seed_formula": "4200 + room_index",
                "max_tries_per_item": 1000,
                "max_room_restarts": 100,
                "sort_largest_first": True
            },
            "validation_summary": validation_summary,
            "rooms": b2_room_entries
        }, f, indent=2)

    # Save B2 secret key
    key_b2_file = Path("outputs/blind_study/keys/study_key_b2.json")
    with open(key_b2_file, "w") as f:
        json.dump({
            "study_id": "engine_vs_b2",
            "description": "Secret evaluation key for Engine vs Baseline B2 (non-overlapping random). NEVER distribute to raters.",
            "seed": 2027,
            "rooms": keys_b2
        }, f, indent=2)

    print(f"Saved B2 layouts to {b2_out_file}")
    print(f"Saved B2 secret key to {key_b2_file}")

    # Also move / ensure B1 secret key is saved in keys/
    # Read existing study_key.json
    old_key_file = Path("outputs/blind_study/study_key.json")
    if old_key_file.exists():
        with open(old_key_file) as f:
            old_key_data = json.load(f)
        key_b1_file = Path("outputs/blind_study/keys/study_key_b1.json")
        with open(key_b1_file, "w") as f:
            json.dump({
                "study_id": "engine_vs_b1",
                "description": "Secret evaluation key for Engine vs Baseline B1 (naive random). NEVER distribute to raters.",
                "seed": old_key_data.get("seed", 2026),
                "rooms": [
                    {
                        "room_id": k["room_id"],
                        "room_type": k["room_type"],
                        "val_plan_index": flat_rooms[idx]["val_plan_index"],
                        "plan_filename": flat_rooms[idx]["plan_filename"],
                        "area_m2": k["area_m2"],
                        "option_a": "engine" if k["is_engine_left"] else "baseline_b1",
                        "option_b": "baseline_b1" if k["is_engine_left"] else "engine",
                        "engine_is": "A" if k["is_engine_left"] else "B"
                    }
                    for idx, k in enumerate(old_key_data["keys"])
                ]
            }, f, indent=2)
        print(f"Migrated B1 secret key to {key_b1_file}")

    print("\n" + "=" * 80)
    print("BASELINE B2 VALIDATION SUMMARY:")
    print(f"Passed: {validation_summary['passed']}/{validation_summary['total_evaluated']} ({validation_summary['passed']/validation_summary['total_evaluated']*100:.1f}%)")
    print(f"Failed: {validation_summary['failed']}/{validation_summary['total_evaluated']} ({validation_summary['failed']/validation_summary['total_evaluated']*100:.1f}%)")
    print(f"  Overlap failures: {validation_summary['overlap_failures']}")
    print(f"  Door clearance violations: {validation_summary['door_clearance_failures']}")
    for t, stat in validation_summary["per_type"].items():
        print(f"  {t:12s}: Passed {stat['passed']}/{stat['total']} | Door fails: {stat['door']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
