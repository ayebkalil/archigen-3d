import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate

from src.design.types import PlacedItem
from src.design.validator import LayoutValidator

def create_oriented_box(center_x, center_y, width, depth, rotation_deg):
    """
    Creates an oriented rectangle polygon in plan coordinates.
    Rotation is clockwise from North (+Y).
    """
    hw = width / 2.0
    hd = depth / 2.0
    # Base box centered at origin: x in [-hw, hw], y in [-hd, hd]
    poly = Polygon([
        (-hw, -hd),
        (hw, -hd),
        (hw, hd),
        (-hw, hd)
    ])
    # Shapely rotate rotates counter-clockwise by default, so -rotation_deg for clockwise
    poly_rot = rotate(poly, -rotation_deg, origin=(0, 0))
    poly_trans = translate(poly_rot, xoff=center_x, yoff=center_y)
    return poly_trans

def generate_random_layout_for_room(room_polygon_pts, engine_items, seed=42):
    """
    Generates a naive random baseline layout:
    - Same items (same dimensions, catalog_id, glb, category)
    - Random positions and rotations inside room polygon
    - Strictly constrained only to stay inside the room polygon: room_poly.contains(item_box)
    - No wall contact, door zone, or collision rules.
    """
    rng = random.Random(seed)
    room_poly = Polygon(room_polygon_pts)
    min_x, min_y, max_x, max_y = room_poly.bounds

    random_items = []

    for idx, item in enumerate(engine_items):
        w = item["dimensions_m"][0]
        d = item["dimensions_m"][1]
        h = item["dimensions_m"][2]

        placed = False
        best_box = None
        best_pos = None
        best_rot = None

        # Rejection sampling: try up to 2000 times to find a position fully inside room_poly
        for attempt in range(2000):
            rx = rng.uniform(min_x + 0.1, max_x - 0.1)
            ry = rng.uniform(min_y + 0.1, max_y - 0.1)
            # Continuous rotation [0, 360)
            rot = rng.uniform(0.0, 360.0)

            box_poly = create_oriented_box(rx, ry, w, d, rot)

            # Check constraint: constrained only to stay inside the room polygon
            if room_poly.contains(box_poly):
                best_box = box_poly
                best_pos = (rx, ry)
                best_rot = rot
                placed = True
                break

        if not placed:
            # Fallback if room is very narrow: center of room
            cx, cy = room_poly.centroid.x, room_poly.centroid.y
            best_pos = (cx, cy)
            best_rot = 0.0
            best_box = create_oriented_box(cx, cy, w, d, 0.0)

        coords = list(best_box.exterior.coords)[:-1]

        random_items.append(PlacedItem(
            item_id=f"rand_{item['item_id']}",
            catalog_id=item["catalog_id"],
            name=item["name"],
            category=item["category"],
            position_m=(round(best_pos[0], 3), round(best_pos[1], 3)),
            rotation_deg=round(best_rot, 1),
            dimensions_m=(w, d, h),
            glb=item["glb"],
            polygon_coords=coords,
            placement_rule="random_baseline",
            wall_distance_cm=99.9,
            wall_contact_ratio=0.0
        ))

    return random_items

def main():
    print("=" * 80)
    print("STEP 2: GENERATING RANDOM BASELINE LAYOUTS & VALIDATING")
    print("=" * 80)

    selected_file = Path("outputs/blind_study/selected_rooms.json")
    with open(selected_file) as f:
        data = json.load(f)

    validator = LayoutValidator()

    # Flatten selected rooms
    flat_rooms = []
    for rtype, rlist in data["selected_rooms"].items():
        for r in rlist:
            flat_rooms.append(r)

    print(f"Total rooms to process: {len(flat_rooms)}")

    baseline_results = []
    validation_summary = {
        "total_evaluated": len(flat_rooms),
        "passed_validation": 0,
        "failed_validation": 0,
        "overlap_failures": 0,
        "door_clearance_failures": 0,
        "wall_contact_failures": 0,
        "bfs_failures": 0,
        "per_type": {}
    }

    study_manifest_items = []
    study_keys = []

    # RNG for Left/Right presentation randomization (seed 2026)
    lr_rng = random.Random(2026)

    for i, r in enumerate(flat_rooms):
        rtype = r["room_type"]
        if rtype not in validation_summary["per_type"]:
            validation_summary["per_type"][rtype] = {
                "total": 0,
                "passed": 0,
                "overlap_failures": 0,
                "door_clearance_failures": 0,
                "wall_contact_failures": 0
            }
        validation_summary["per_type"][rtype]["total"] += 1

        engine_items = r["engine_result"]["items"]
        poly_pts = r["polygon_meters"]
        room_poly = Polygon(poly_pts)

        # Build door swing zones
        door_clearance_zones = []
        for d in r.get("doors", []):
            if "clearance_polygon_meters" in d:
                dp = Polygon(d["clearance_polygon_meters"])
                if dp.is_valid:
                    door_clearance_zones.append(dp)
            elif "position_meters" in d:
                px, py = d["position_meters"]
                door_clearance_zones.append(Point(px, py).buffer(0.85))

        # 1. Generate random layout with fixed seed = 42 + i
        rand_seed = 42 + i
        rand_items = generate_random_layout_for_room(poly_pts, engine_items, seed=rand_seed)

        # 2. Validate random layout with our validator
        val_rep = validator.validate(
            room_poly=room_poly,
            items=rand_items,
            door_clearance_zones=door_clearance_zones
        )

        if val_rep.is_valid:
            validation_summary["passed_validation"] += 1
            validation_summary["per_type"][rtype]["passed"] += 1
        else:
            validation_summary["failed_validation"] += 1
            if val_rep.overlap_count > 0:
                validation_summary["overlap_failures"] += 1
                validation_summary["per_type"][rtype]["overlap_failures"] += 1
            if val_rep.door_clearance_violation:
                validation_summary["door_clearance_failures"] += 1
                validation_summary["per_type"][rtype]["door_clearance_failures"] += 1
            if any("WALL CONTACT" in reason for reason in val_rep.failure_reasons):
                validation_summary["wall_contact_failures"] += 1
                validation_summary["per_type"][rtype]["wall_contact_failures"] += 1

        rand_items_dict = [
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
            for it in rand_items
        ]

        room_entry = {
            "room_id": f"room_{i+1:02d}",
            "room_type": rtype,
            "plan_filename": r["plan_filename"],
            "val_plan_index": r["val_plan_index"],
            "area_m2": r["area_m2"],
            "polygon_meters": poly_pts,
            "engine_items": engine_items,
            "random_seed": rand_seed,
            "random_items": rand_items_dict,
            "random_validation": {
                "is_valid": val_rep.is_valid,
                "overlap_count": val_rep.overlap_count,
                "max_overlap_area_m2": round(val_rep.max_overlap_area_m2, 4),
                "door_clearance_violation": val_rep.door_clearance_violation,
                "out_of_bounds_count": val_rep.out_of_bounds_count,
                "failure_reasons": val_rep.failure_reasons
            }
        }
        baseline_results.append(room_entry)

        # 3. Randomize Left / Right presentation
        # If is_engine_left is True, Left = Engine, Right = Random
        # If False, Left = Random, Right = Engine
        is_engine_left = (lr_rng.random() < 0.5)
        room_label = f"room_{i+1:02d}"

        if is_engine_left:
            img_left = f"{room_label}_img_engine.png"
            img_right = f"{room_label}_img_random.png"
            correct_winner = "Left"
        else:
            img_left = f"{room_label}_img_random.png"
            img_right = f"{room_label}_img_engine.png"
            correct_winner = "Right"

        study_manifest_items.append({
            "room_id": room_label,
            "room_type": rtype,
            "area_m2": r["area_m2"],
            "image_left": f"renders/{img_left}",
            "image_right": f"renders/{img_right}"
        })

        study_keys.append({
            "room_id": room_label,
            "room_type": rtype,
            "area_m2": r["area_m2"],
            "is_engine_left": is_engine_left,
            "left_condition": "Engine" if is_engine_left else "Random",
            "right_condition": "Random" if is_engine_left else "Engine",
            "engine_choice": correct_winner,
            "image_left": img_left,
            "image_right": img_right,
            "random_seed": rand_seed
        })

    print("\n" + "=" * 80)
    print("RANDOM BASELINE VALIDATION SUMMARY (WHAT THE TEST MEASURES):")
    print(f"Total evaluated layouts: {validation_summary['total_evaluated']}")
    print(f"Passed full validation: {validation_summary['passed_validation']} / {validation_summary['total_evaluated']} ({validation_summary['passed_validation']/validation_summary['total_evaluated']*100:.1f}%)")
    print(f"Failed validation:       {validation_summary['failed_validation']} / {validation_summary['total_evaluated']} ({validation_summary['failed_validation']/validation_summary['total_evaluated']*100:.1f}%)")
    print(f"  Overlap / Collision violations: {validation_summary['overlap_failures']}")
    print(f"  Door clearance violations:      {validation_summary['door_clearance_failures']}")
    print(f"  Wall contact rule violations:   {validation_summary['wall_contact_failures']}")
    print("-" * 50)
    for t, s in validation_summary["per_type"].items():
        print(f"  {t:12s}: {s['passed']}/{s['total']} passed ({s['passed']/s['total']*100:.1f}%) | {s['overlap_failures']} overlaps | {s['door_clearance_failures']} door blocks")
    print("=" * 80)

    # Save baseline layouts and validation report
    out_dir = Path("outputs/blind_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "random_baseline_layouts.json", "w") as f:
        json.dump({
            "validation_summary": validation_summary,
            "rooms": baseline_results
        }, f, indent=2)

    # Save Secret Key file (raters never see this)
    with open(out_dir / "study_key.json", "w") as f:
        json.dump({
            "description": "Authoritative secret key for blind room design evaluation. Maps anonymous Left/Right choices to Engine vs Random.",
            "total_rooms": len(study_keys),
            "random_assignment_seed": 2026,
            "keys": study_keys
        }, f, indent=2)
    print(f"Saved secret key to {out_dir / 'study_key.json'}")

    # Save Public Study Manifest (for the static HTML page)
    with open(out_dir / "study_manifest.json", "w") as f:
        json.dump({
            "version": "1.0",
            "title": "ArchiGen 3D — Room Layout Blind Evaluation",
            "total_rooms": len(study_manifest_items),
            "rooms": study_manifest_items
        }, f, indent=2)
    print(f"Saved public study manifest to {out_dir / 'study_manifest.json'}")

if __name__ == "__main__":
    main()
