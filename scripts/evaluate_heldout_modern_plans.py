"""
evaluate_heldout_modern_plans.py
Authoritative evaluation of ArchiGen 3D on the brand-new, untouched
held-out modern real estate listing floor plans dataset (N = 8 plans).
Evaluates single-pass vs 4-fold TTA shell validity, wall/room detection,
and renders visual vector layout overlays.
"""

import sys
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

OUT_DIR = PROJECT_ROOT / "outputs/heldout_modern_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def render_plan_overlay(raw_img_bytes: bytes, layout: dict, out_path: Path):
    """Draws predicted walls and rooms over the original floorplan image."""
    img = Image.open(Path(raw_img_bytes) if isinstance(raw_img_bytes, (str, Path)) else Path(out_path)).convert("RGB") if isinstance(raw_img_bytes, (str, Path)) else Image.open(io.BytesIO(raw_img_bytes)).convert("RGB")
    # Actually let's use cv2 or PIL
    pass

def evaluate_modern():
    manifest_p = PROJECT_ROOT / "data/real_external_plans/heldout_modern/heldout_modern_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = FloorplanPipeline()
    print("=" * 90)
    print("EVALUATING ON HELD-OUT MODERN LISTING FLOORPLANS (N = 8)")
    print("=" * 90)

    results = []
    for plan in manifest:
        pid = plan["id"]
        fname = plan["filename"]
        img_p = PROJECT_ROOT / "data/real_external_plans/heldout_modern" / fname
        img_bytes = img_p.read_bytes()

        # 1. Single-Pass Inference
        t0 = time.time()
        res_sp = pipeline.process_image(img_bytes, tta_mode="none", front_only=True)
        sp_time = (time.time() - t0) * 1000.0
        meta_sp = res_sp["layout"]["metadata"]

        # 2. 4-Fold TTA Inference
        t1 = time.time()
        res_tta = pipeline.process_image(img_bytes, tta_mode="4_fold", front_only=True)
        tta_time = (time.time() - t1) * 1000.0
        meta_tta = res_tta["layout"]["metadata"]

        print(f"\n[{pid}] {plan['style']} ({fname}):")
        print(f"  Single-Pass: Valid={meta_sp['is_building_shell_valid']} (Repaired={meta_sp.get('was_shell_repaired', False)}) | Walls={meta_sp['num_walls']} Rooms={meta_sp['num_rooms']} Doors={meta_sp['num_doors']} ({sp_time:.1f}ms)")
        print(f"  4-Fold TTA:  Valid={meta_tta['is_building_shell_valid']} (Repaired={meta_tta.get('was_shell_repaired', False)}) | Walls={meta_tta['num_walls']} Rooms={meta_tta['num_rooms']} Doors={meta_tta['num_doors']} ({tta_time:.1f}ms)")

        # Save visualization overlay for 4-fold
        orig_img = Image.open(img_p).convert("RGB")
        ow, oh = orig_img.size
        # Letterbox overlay coordinates
        overlay = orig_img.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        scale_m = meta_tta["scale_meters_per_pixel"]

        # Draw rooms
        for r in res_tta["layout"].get("rooms", []):
            pts_m = r.get("polygon_meters", [])
            if len(pts_m) >= 3:
                # Convert from meters back to original image space
                # In FloorplanPipeline, canvas is 512x512 letterboxed
                scale_box = min(512 / ow, 512 / oh)
                nw, nh = int(ow * scale_box), int(oh * scale_box)
                pad_x = (512 - nw) // 2
                pad_y = (512 - nh) // 2

                pts_px = []
                for pt in pts_m:
                    canvas_x = pt[0] / scale_m
                    canvas_y = pt[1] / scale_m
                    orig_x = (canvas_x - pad_x) / scale_box
                    orig_y = (canvas_y - pad_y) / scale_box
                    pts_px.append((orig_x, orig_y))
                draw.polygon(pts_px, fill=(0, 200, 100, 70), outline=(0, 150, 50, 200))

        # Draw walls
        for w in res_tta["layout"].get("walls", []):
            pts_m = w.get("polygon_meters", [])
            if len(pts_m) >= 2:
                scale_box = min(512 / ow, 512 / oh)
                nw, nh = int(ow * scale_box), int(oh * scale_box)
                pad_x = (512 - nw) // 2
                pad_y = (512 - nh) // 2

                pts_px = []
                for pt in pts_m:
                    canvas_x = pt[0] / scale_m
                    canvas_y = pt[1] / scale_m
                    orig_x = (canvas_x - pad_x) / scale_box
                    orig_y = (canvas_y - pad_y) / scale_box
                    pts_px.append((orig_x, orig_y))
                if len(pts_px) >= 3:
                    draw.polygon(pts_px, outline=(255, 0, 0, 230), fill=(255, 50, 50, 80))

        overlay_path = OUT_DIR / f"{pid}_vector_overlay.png"
        overlay.save(overlay_path)

        results.append({
            "id": pid,
            "filename": fname,
            "style": plan["style"],
            "resolution": [ow, oh],
            "single_pass": {
                "valid": meta_sp["is_building_shell_valid"],
                "was_repaired": meta_sp.get("was_shell_repaired", False),
                "num_walls": meta_sp["num_walls"],
                "num_rooms": meta_sp["num_rooms"],
                "num_doors": meta_sp["num_doors"],
                "num_windows": meta_sp["num_windows"],
                "perimeter_m": meta_sp.get("building_perimeter_m", 0.0),
                "latency_ms": round(sp_time, 1)
            },
            "four_fold_tta": {
                "valid": meta_tta["is_building_shell_valid"],
                "was_repaired": meta_tta.get("was_shell_repaired", False),
                "num_walls": meta_tta["num_walls"],
                "num_rooms": meta_tta["num_rooms"],
                "num_doors": meta_tta["num_doors"],
                "num_windows": meta_tta["num_windows"],
                "perimeter_m": meta_tta.get("building_perimeter_m", 0.0),
                "latency_ms": round(tta_time, 1)
            },
            "overlay_path": str(overlay_path)
        })

    summary_p = PROJECT_ROOT / "outputs/heldout_modern_eval_summary.json"
    with open(summary_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sp_valid_count = sum(1 for r in results if r["single_pass"]["valid"])
    tta_valid_count = sum(1 for r in results if r["four_fold_tta"]["valid"])

    print("\n" + "=" * 90)
    print("HELD-OUT MODERN LISTING EVALUATION SUMMARY:")
    print("=" * 90)
    print(f"Total Untouched Modern Plans: {len(results)}")
    print(f"Single-Pass Validity:         {sp_valid_count}/{len(results)} ({sp_valid_count/len(results)*100:.1f}%)")
    print(f"4-Fold TTA Validity:          {tta_valid_count}/{len(results)} ({tta_valid_count/len(results)*100:.1f}%)")
    print(f"[+] Overlay images saved to:  {OUT_DIR}")
    print(f"[+] Summary saved to:         {summary_p}")

if __name__ == "__main__":
    evaluate_modern()
