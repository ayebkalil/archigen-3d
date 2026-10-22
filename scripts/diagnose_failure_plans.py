"""
diagnose_failure_plans.py
Detailed inspection of failure modes on real_05, real_07, and real_08.
Extracts:
1. Shell status and validity.
2. Walls, doors, rooms detected.
3. Why the building envelope is flagged as INVALID_GEOMETRY.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import explain_validity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

def diagnose():
    pipeline = FloorplanPipeline()
    manifest_p = PROJECT_ROOT / "data/real_external_plans/real_external_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rid in ["real_05", "real_07", "real_08"]:
        plan_info = next(p for p in manifest if p["id"] == rid)
        img_path = PROJECT_ROOT / "data/real_external_plans" / plan_info["filename"]
        print(f"\n=======================================================")
        print(f"PLAN {rid}: {plan_info['title']}")
        print(f"=======================================================")

        # Run pipeline with 4-fold TTA
        img_bytes = img_path.read_bytes()
        res = pipeline.process_image(img_bytes, tta_mode="4_fold", front_only=True)
        layout = res["layout"]
        meta = layout["metadata"]
        print(f"Shell Status: {meta.get('shell_status')}")
        print(f"Is Shell Valid: {meta.get('is_building_shell_valid')}")
        print(f"Walls: {meta.get('num_walls')} (Exterior: {meta.get('num_exterior_walls')})")
        print(f"Rooms: {meta.get('num_rooms')}")
        print(f"Doors count: {len(layout.get('doors', []))}")
        print(f"Perimeter: {meta.get('building_perimeter_m')} m, Surface: {meta.get('total_surface_m2')} m2")

        # Let's inspect rooms and their polygons
        for r in layout.get("rooms", []):
            pts = r.get("polygon_meters", [])
            if len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    print(f"  --> INVALID ROOM {r['id']} ({r['room_type']}): {explain_validity(poly)}")
                else:
                    print(f"  Room {r['id']} ({r['room_type']}): Area={r['surface_m2']} m2, Vertices={len(pts)}, Valid=True")

if __name__ == "__main__":
    diagnose()
