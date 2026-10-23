"""
diagnose_heldout_modern.py
Inspects scale estimation, area, perimeter, and door over-detection
on the 8 held-out modern listing floor plans.
"""

import sys
import json
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

pipeline = FloorplanPipeline()
manifest_p = PROJECT_ROOT / "data/real_external_plans/heldout_modern/heldout_modern_manifest.json"
with open(manifest_p, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("=" * 90)
print("DIAGNOSING SCALE & DOOR OVER-DETECTION ON HELD-OUT MODERN PLANS")
print("=" * 90)

diagnostics = []
for p in manifest:
    img_p = PROJECT_ROOT / "data/real_external_plans/heldout_modern" / p["filename"]
    res = pipeline.process_image(img_p.read_bytes(), tta_mode="4_fold", front_only=True)
    meta = res["layout"]["metadata"]
    doors = res["layout"]["doors"]
    rooms = res["layout"]["rooms"]
    walls = res["layout"]["walls"]
    door_widths = [d["width_m"] for d in doors]
    mean_w = sum(door_widths) / len(door_widths) if door_widths else 0

    diag = {
        "id": p["id"],
        "filename": p["filename"],
        "style": p["style"],
        "resolution": p["resolution"],
        "scale_m_per_px": meta["scale_meters_per_pixel"],
        "scale_source": meta["scale_source"],
        "total_surface_m2": meta["total_surface_m2"],
        "perimeter_m": meta["building_perimeter_m"],
        "is_valid": meta["is_building_shell_valid"],
        "shell_status": meta["shell_status"],
        "num_doors": len(doors),
        "mean_door_width_m": round(mean_w, 2),
        "min_door_width_m": round(min(door_widths), 2) if door_widths else 0,
        "max_door_width_m": round(max(door_widths), 2) if door_widths else 0,
        "num_rooms": len(rooms),
        "num_walls": len(walls)
    }
    diagnostics.append(diag)

    print(f"\n[{p['id']}] {p['style']} ({p['filename']}):")
    print(f"  Scale: {meta['scale_meters_per_pixel']:.4f} m/px ({meta['scale_source']})")
    print(f"  Floor Surface: {meta['total_surface_m2']:.1f} m2 | Perimeter: {meta['building_perimeter_m']:.1f} m")
    print(f"  Shell Valid: {meta['is_building_shell_valid']} ({meta['shell_status']})")
    print(f"  Detected Doors: {len(doors)} (mean: {mean_w:.2f}m, min: {diag['min_door_width_m']}m, max: {diag['max_door_width_m']}m)")
    print(f"  Detected Rooms: {len(rooms)} | Walls: {len(walls)}")

out_json = PROJECT_ROOT / "outputs/heldout_modern_scale_diagnostics.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(diagnostics, f, indent=2)

print(f"\n[+] Saved diagnostics to: {out_json}")
