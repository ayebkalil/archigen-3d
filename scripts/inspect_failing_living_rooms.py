import os, sys, glob
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.floorplan.vectorizer import FloorplanVectorizer
import scipy.ndimage as ndi
from shapely.geometry import Polygon, LineString

val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))[:100]
v = FloorplanVectorizer()

failing_living_rooms = []

for idx, f in enumerate(val_files):
    fname = os.path.basename(f)
    d = np.load(f)
    mask = d["mask"]
    room_mask = d["room_mask"]
    struct_mask = d["struct_mask"]
    
    gt_wall_bin = (mask == 1).astype(np.uint8)
    layout = v.vectorize(mask, room_mask=room_mask, struct_mask=struct_mask)
    scale_m = layout["metadata"]["scale_meters_per_pixel"]
    
    dist_to_wall_px = ndi.distance_transform_edt(1 - gt_wall_bin)
    dist_to_wall_m = dist_to_wall_px * scale_m
    
    for r_idx, r in enumerate(layout.get("rooms", [])):
        rtype = r.get("room_type", "other").lower()
        if "liv" not in rtype:
            continue
        
        pts = r.get("polygon_meters", [])
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid or poly.is_empty:
            continue
        
        num_pts = len(pts)
        wall_backed_edges = 0
        edge_details = []
        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts]
            edge = LineString([p1, p2])
            length = edge.length
            
            dists = []
            for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
                pt = edge.interpolate(frac, normalized=True)
                ix = int(round(pt.x / scale_m))
                iy = int(round(pt.y / scale_m))
                if 0 <= ix < 512 and 0 <= iy < 512:
                    dists.append(dist_to_wall_m[iy, ix])
            med_d = float(np.median(dists)) if dists else 999.0
            is_wb = (med_d <= 0.15)
            if is_wb:
                wall_backed_edges += 1
            edge_details.append({"len": round(length, 2), "dist": round(med_d, 3), "wall_backed": is_wb})
            
        if wall_backed_edges < 2:
            failing_living_rooms.append({
                "plan_idx": idx,
                "file": fname,
                "room_id": r.get("id", f"room_{r_idx}"),
                "area_m2": round(poly.area, 2),
                "wall_backed_count": wall_backed_edges,
                "total_edges": num_pts,
                "edge_details": edge_details
            })

print(f"Total living rooms with < 2 wall-backed edges: {len(failing_living_rooms)}")
for fl in failing_living_rooms:
    print(f"\nPlan {fl['plan_idx']} ({fl['file']}): {fl['room_id']}, Area: {fl['area_m2']} m2, Wall-backed: {fl['wall_backed_count']}/{fl['total_edges']}")
    for j, ed in enumerate(fl['edge_details']):
        print(f"   Edge {j}: len={ed['len']}m, dist={ed['dist']}m, wall_backed={ed['wall_backed']}")
