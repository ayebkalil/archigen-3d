"""
scripts/diagnose_wall_offsets.py
--------------------------------
Diagnoses room-to-wall distances across CubiCasa validation plans:
1. Compares:
   a) Distance to vectorized wall polygons
   b) Distance to ground-truth wall mask pixels
2. Generates overlay plots for 5 plans showing:
   - GT wall mask / vectorized walls
   - Room polygons
   - Wall-offset values along room boundaries
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import scipy.ndimage as ndi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.floorplan.vectorizer import FloorplanVectorizer

def diagnose():
    val_files = sorted(glob.glob("data/CubiCasa5k/cached_512/val/*.npz"))[:5]
    v = FloorplanVectorizer()
    
    os.makedirs("outputs/wall_offset_diagnostics", exist_ok=True)
    
    for plan_idx, fpath in enumerate(val_files):
        fname = os.path.basename(fpath)
        d = np.load(fpath)
        mask = d["mask"]
        room_mask = d["room_mask"]
        struct_mask = d["struct_mask"]
        
        # 1. Ground truth wall binary mask (mask == 1)
        gt_wall_bin = (mask == 1).astype(np.uint8)
        
        # 2. Vectorize correctly with all masks
        layout = v.vectorize(mask, room_mask=room_mask, struct_mask=struct_mask)
        scale_m = layout["metadata"]["scale_meters_per_pixel"]
        
        # Distance transform to nearest wall pixel (in pixels and meters)
        # 1 - gt_wall_bin is 0 inside walls, 1 outside walls
        dist_to_wall_px = ndi.distance_transform_edt(1 - gt_wall_bin)
        dist_to_wall_m = dist_to_wall_px * scale_m
        
        # Build Shapely MultiPolygon for vectorized walls
        wall_polys = []
        for w in layout.get("walls", []):
            pts = w.get("polygon_meters", [])
            if len(pts) >= 3:
                p = Polygon(pts)
                if p.is_valid and not p.is_empty:
                    wall_polys.append(p)
        wall_union = MultiPolygon(wall_polys) if len(wall_polys) > 1 else (wall_polys[0] if wall_polys else None)
        
        rooms = layout.get("rooms", [])
        
        gt_dists_all = []
        poly_dists_all = []
        
        # Create visual overlay
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left: Ground truth masks & overlay
        img_rgb = d["image"].copy()
        axes[0].imshow(img_rgb)
        axes[0].imshow(gt_wall_bin, cmap="Reds", alpha=0.35)
        
        # Plot room polygons on Left
        for r_idx, r in enumerate(rooms):
            pts = r.get("polygon_meters", [])
            if len(pts) < 3:
                continue
            px_pts = np.array([(p[0] / scale_m, p[1] / scale_m) for p in pts])
            axes[0].plot(px_pts[:, 0], px_pts[:, 1], "b-", linewidth=2)
            axes[0].fill(px_pts[:, 0], px_pts[:, 1], alpha=0.15, label=r["room_type"] if r_idx < 5 else "")
            
            # Sample points along boundary
            r_poly = Polygon(pts)
            for frac in np.linspace(0.0, 1.0, 20, endpoint=False):
                pt = r_poly.exterior.interpolate(frac, normalized=True)
                # Distance to vectorized wall polygon
                if wall_union is not None:
                    d_poly = float(wall_union.distance(pt))
                    poly_dists_all.append(d_poly)
                # Distance to GT wall pixel
                ix = int(round(pt.x / scale_m))
                iy = int(round(pt.y / scale_m))
                if 0 <= ix < 512 and 0 <= iy < 512:
                    d_gt = float(dist_to_wall_m[iy, ix])
                    gt_dists_all.append(d_gt)
                    
        axes[0].set_title(f"Plan {plan_idx}: Image + GT Walls (Red) + Room Polys (Blue)\nScale: {scale_m:.4f} m/px", fontsize=11)
        axes[0].axis("off")
        
        # Right: Vectorized Walls + Room Polygons in Plan Space (Meters)
        axes[1].set_facecolor("#f8fafc")
        if wall_union is not None:
            for wp in wall_polys:
                w_pts = np.array(wp.exterior.coords)
                axes[1].plot(w_pts[:, 0], w_pts[:, 1], "k-", linewidth=2)
                axes[1].fill(w_pts[:, 0], w_pts[:, 1], color="#334155", alpha=0.6)
                
        for r in rooms:
            pts = np.array(r.get("polygon_meters", []))
            if len(pts) < 3:
                continue
            axes[1].plot(pts[:, 0], pts[:, 1], "c-", linewidth=1.5)
            axes[1].fill(pts[:, 0], pts[:, 1], color="#38bdf8", alpha=0.2)
            # Centroid label
            cx, cy = r.get("centroid_meters", [pts[:, 0].mean(), pts[:, 1].mean()])
            axes[1].text(cx, cy, r["room_type"], color="#0f172a", fontsize=9, ha="center", va="center", weight="bold")
            
        axes[1].set_title(f"Plan {plan_idx}: Vectorized Plan Space (Meters)\nWalls (Black) vs Rooms (Cyan)", fontsize=11)
        axes[1].set_aspect("equal")
        axes[1].invert_yaxis()
        
        out_plot = f"outputs/wall_offset_diagnostics/plan_{plan_idx}_overlay.png"
        plt.tight_layout()
        plt.savefig(out_plot, dpi=120)
        plt.close()
        
        gt_arr = np.array(gt_dists_all)
        poly_arr = np.array(poly_dists_all)
        
        print(f"=== PLAN {plan_idx} ({fname}) ===")
        print(f"  Scale: {scale_m:.4f} m/px | Rooms: {len(rooms)} | Walls: {len(layout.get('walls', []))}")
        print(f"  [GT Wall Mask Pixels] Median: {np.median(gt_arr):.4f} m | <=0.10m: {(gt_arr <= 0.10).mean()*100:.1f}% | <=0.20m: {(gt_arr <= 0.20).mean()*100:.1f}% | Min: {gt_arr.min():.4f} m")
        print(f"  [Vectorized Polygons] Median: {np.median(poly_arr):.4f} m | <=0.10m: {(poly_arr <= 0.10).mean()*100:.1f}% | <=0.20m: {(poly_arr <= 0.20).mean()*100:.1f}% | Min: {poly_arr.min():.4f} m")
        print(f"  Saved visual overlay to: {out_plot}\n")

if __name__ == "__main__":
    diagnose()
