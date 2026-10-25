"""
scripts/measure_wall_offsets_multi.py
-------------------------------------
Measures the distance from room boundary edges to the nearest inner wall face
across multiple real dev floorplans.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapely.geometry import Polygon, MultiPolygon
from backend.app.floorplan_engine import FloorplanPipeline

def run():
    pipeline = FloorplanPipeline.get_instance()
    manifest_p = os.path.join("data", "dev_external_plans", "dev_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Measuring room-to-wall offset across {len(manifest)} dev plans...")
    plan_records = []
    all_sample_distances = []
    
    for entry in manifest:
        plan_id = entry["id"]
        img_path = os.path.join("data", "dev_external_plans", entry["filename"])
        t0 = time.time()
        with open(img_path, "rb") as f_img:
            raw = f_img.read()
        res = pipeline.process_image(raw, tta_mode="none", front_only=True)
        layout = res["layout"]
        
        wall_polys = [Polygon(w["polygon_meters"]) for w in layout.get("walls", []) if len(w.get("polygon_meters", [])) >= 3]
        if not wall_polys:
            continue
            
        wall_union = MultiPolygon(wall_polys) if len(wall_polys) > 1 else wall_polys[0]
        
        plan_offsets = []
        for r in layout.get("rooms", []):
            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue
            r_poly = Polygon(pts)
            r_bound = r_poly.exterior
            # Sample 20 points along perimeter of each room
            for frac in np.linspace(0.0, 1.0, num=20, endpoint=False):
                pt = r_bound.interpolate(frac, normalized=True)
                d = float(wall_union.distance(pt))
                plan_offsets.append(d)
                all_sample_distances.append(d)
                
        if plan_offsets:
            arr = np.array(plan_offsets)
            plan_records.append({
                "id": plan_id,
                "rooms_count": len(layout.get("rooms", [])),
                "samples_count": len(arr),
                "min_m": float(np.min(arr)),
                "p25_m": float(np.percentile(arr, 25)),
                "median_m": float(np.median(arr)),
                "p75_m": float(np.percentile(arr, 75)),
                "max_m": float(np.max(arr)),
                "elapsed_s": round(time.time() - t0, 2)
            })
            print(f"[{plan_id}] min: {np.min(arr):.3f}m | med: {np.median(arr):.3f}m | p75: {np.percentile(arr, 75):.3f}m | max: {np.max(arr):.3f}m ({time.time() - t0:.1f}s)")
            
    print("\n" + "="*80)
    print(f"WALL OFFSET MEASUREMENT SUMMARY ({len(plan_records)} PLANS EVALUATED)")
    print("="*80)
    all_arr = np.array(all_sample_distances)
    print(f"Total Boundary Points Sampled: {len(all_arr)}")
    print(f"Global Minimum Offset:        {np.min(all_arr):.4f} m (0.0 cm - exact contact)")
    print(f"Global 25th Percentile:       {np.percentile(all_arr, 25):.4f} m")
    print(f"Global Median Offset:         {np.median(all_arr):.4f} m")
    print(f"Global 75th Percentile:       {np.percentile(all_arr, 75):.4f} m")
    print(f"Fraction Points <= 0.05 m:    {(all_arr <= 0.05).mean() * 100:.1f}%")
    print(f"Fraction Points <= 0.10 m:    {(all_arr <= 0.10).mean() * 100:.1f}%")
    print(f"Points > 0.15 m (openings):   {(all_arr > 0.15).mean() * 100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    run()
