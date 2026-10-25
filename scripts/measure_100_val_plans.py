"""
scripts/measure_100_val_plans.py
--------------------------------
Evaluates room-to-wall distances and edge classification on 100 CubiCasa validation plans:
1. Measures distance from room boundary points to nearest inner wall faces.
2. Classifies each room polygon edge as:
   - 'wall-backed' if edge distance to nearest wall is <= 0.10 m.
   - 'open' if edge distance > 0.10 m (open concepts, doorways, open boundaries).
3. Computes the share of rooms with at least 2 wall-backed edges, broken down by room type.
"""

import os
import sys
import glob
import time
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon, LineString, Point

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.floorplan.vectorizer import FloorplanVectorizer

def classify_edge(p1: tuple, p2: tuple, wall_union: MultiPolygon, threshold_m: float = 0.10) -> bool:
    """Returns True if the edge is wall-backed (<= threshold_m from wall), else False (open)."""
    edge = LineString([p1, p2])
    # Sample 5 points along edge
    dists = []
    for f in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pt = edge.interpolate(f, normalized=True)
        dists.append(float(wall_union.distance(pt)))
    return float(np.median(dists)) <= threshold_m

def run():
    val_dir = os.path.join("data", "CubiCasa5k", "cached_512", "val")
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))[:100]
    assert len(val_files) >= 100, f"Expected 100 val files, found {len(val_files)}"
    
    print(f"Evaluating {len(val_files)} CubiCasa validation plans...")
    vectorizer = FloorplanVectorizer()
    
    all_sample_distances = []
    room_type_total = defaultdict(int)
    room_type_at_least_2_walls = defaultdict(int)
    
    plans_processed = 0
    t0 = time.time()
    
    for idx, npz_path in enumerate(val_files):
        data = np.load(npz_path)
        struct_mask = data["struct_mask"]
        room_mask = data["room_mask"]
        
        layout = vectorizer.vectorize(struct_mask, room_mask)
        walls = layout.get("walls", [])
        rooms = layout.get("rooms", [])
        
        wall_polys = []
        for w in walls:
            pts = w.get("polygon_meters", [])
            if len(pts) >= 3:
                p = Polygon(pts)
                if p.is_valid and not p.is_empty:
                    wall_polys.append(p)
                    
        if not wall_polys or not rooms:
            continue
            
        wall_union = MultiPolygon(wall_polys) if len(wall_polys) > 1 else wall_polys[0]
        plans_processed += 1
        
        for r in rooms:
            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue
            r_poly = Polygon(pts)
            if not r_poly.is_valid or r_poly.is_empty:
                continue
                
            rtype = r.get("room_type", "other").lower().strip()
            # Normalize room type
            if "bed" in rtype: norm_type = "bedroom"
            elif "liv" in rtype: norm_type = "living"
            elif "din" in rtype: norm_type = "dining"
            elif "off" in rtype or "study" in rtype: norm_type = "office"
            elif "kitch" in rtype: norm_type = "kitchen"
            elif "bath" in rtype: norm_type = "bathroom"
            elif "entry" in rtype or "hall" in rtype or "corr" in rtype: norm_type = "entry/hall"
            elif "stor" in rtype or "clos" in rtype: norm_type = "storage"
            elif "balc" in rtype or "terr" in rtype: norm_type = "balcony"
            else: norm_type = "other"
            
            # 1. Sample perimeter points for global distance distribution
            exterior = r_poly.exterior
            for frac in np.linspace(0.0, 1.0, num=15, endpoint=False):
                pt = exterior.interpolate(frac, normalized=True)
                d = float(wall_union.distance(pt))
                all_sample_distances.append(d)
                
            # 2. Edge classification
            num_pts = len(pts)
            wall_backed_edges_count = 0
            for i in range(num_pts):
                p1 = pts[i]
                p2 = pts[(i + 1) % num_pts]
                if classify_edge(p1, p2, wall_union, threshold_m=0.10):
                    wall_backed_edges_count += 1
                    
            room_type_total[norm_type] += 1
            if wall_backed_edges_count >= 2:
                room_type_at_least_2_walls[norm_type] += 1
                
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/100 plans ({time.time() - t0:.1f}s)...")
            
    print("\n" + "="*85)
    print(f"100 CUBICASA VALIDATION PLANS: ROOM EDGES VS WALLS EMPIRICAL MEASUREMENT")
    print("="*85)
    dists = np.array(all_sample_distances)
    print(f"Total Plans Evaluated:           {plans_processed}")
    print(f"Total Perimeter Points Sampled:   {len(dists)}")
    print(f"Global Minimum Offset:           {np.min(dists):.4f} m (0.0 cm)")
    print(f"Global 25th Percentile:          {np.percentile(dists, 25):.4f} m")
    print(f"Global Median Offset:            {np.median(dists):.4f} m")
    print(f"Global 75th Percentile:          {np.percentile(dists, 75):.4f} m")
    print(f"Share Points <= 0.05 m:          {(dists <= 0.05).mean() * 100:.1f}%")
    print(f"Share Points <= 0.10 m:          {(dists <= 0.10).mean() * 100:.1f}%")
    print(f"Share Points > 0.15 m (open):    {(dists > 0.15).mean() * 100:.1f}%")
    print("="*85)
    
    print("\n" + "="*85)
    print("SHARE OF ROOMS WITH AT LEAST 2 WALL-BACKED EDGES (BY ROOM TYPE)")
    print("="*85)
    print(f"| {'Room Type':<16} | {'Total Rooms':<12} | {'Rooms >= 2 Wall Edges':<22} | {'Share (%)':<10} |")
    print(f"|{'-'*18}|{'-'*14}|{'-'*24}|{'-'*12}|")
    
    target_order = ["bedroom", "living", "dining", "office", "kitchen", "bathroom", "entry/hall", "storage", "balcony", "other"]
    for rt in target_order:
        tot = room_type_total[rt]
        at_least_2 = room_type_at_least_2_walls[rt]
        pct = (at_least_2 / max(tot, 1)) * 100.0
        print(f"| {rt:<16} | {tot:<12} | {at_least_2:<22} | {pct:>8.1f}% |")
    print("="*85 + "\n")

if __name__ == "__main__":
    run()
