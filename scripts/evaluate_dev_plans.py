"""
Development Set Evaluation Script for Floorplan Domain Adaptation.
Evaluates floorplan pipeline on the 20 independent dev plans (data/dev_external_plans/dev_manifest.json).
Outputs structured metrics, vector overlays, and a summary of usability.
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

def evaluate_dev_dataset(model_weights_path: str = None, run_name: str = "baseline_v1_1"):
    manifest_path = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dev manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_dir = PROJECT_ROOT / "outputs" / "dev_results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FloorplanPipeline.get_instance()
    # If custom weights provided, load them into pipeline
    if model_weights_path:
        import torch
        print(f"[EVAL] Loading custom weights from {model_weights_path}...")
        ckpt = torch.load(model_weights_path, map_location=pipeline.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if hasattr(pipeline, "multitask_model") and pipeline.multitask_model is not None:
            pipeline.multitask_model.load_state_dict(state_dict)
            pipeline.multitask_model.eval()
        else:
            pipeline.model.load_state_dict(state_dict)
            pipeline.model.eval()

    print("=" * 100)
    print(f"Evaluating {len(manifest)} Development Plans (Run: {run_name})")
    print(f"Device: {pipeline.device}")
    print("=" * 100)

    results = []
    valid_shells_count = 0
    total_rooms_found = 0
    total_doors_found = 0

    for idx, item in enumerate(manifest):
        plan_id = item["id"]
        filename = item["filename"]
        img_path = manifest_path.parent / filename

        print(f"[{idx+1:02d}/{len(manifest)}] Evaluating {plan_id}: {filename}...")
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        try:
            res = pipeline.process_image(raw_bytes, tta_mode="none", front_only=True)
            layout = res.get("layout", {})
            meta = layout.get("metadata", {})

            walls = layout.get("walls", [])
            rooms = layout.get("rooms", [])
            doors = layout.get("doors", [])
            windows = layout.get("windows", [])

            is_shell_valid = meta.get("is_building_shell_valid", False)
            shell_status = meta.get("shell_status", "UNKNOWN")
            perim_m = meta.get("building_perimeter_m", 0.0)
            area_m2 = meta.get("total_surface_m2", 0.0)

            if is_shell_valid:
                valid_shells_count += 1
            total_rooms_found += len(rooms)
            total_doors_found += len(doors)

            # Heuristic triage score based on structural coherence:
            # 1. Usable: Valid shell, >= 3 rooms, reasonable area (> 25 m2), non-zero walls
            # 2. Needs fixes: >= 1 room or valid shell, some walls/doors
            # 3. Unusable: 0 rooms or collapsed shell or area < 10 m2
            if is_shell_valid and len(rooms) >= 3 and area_m2 >= 25.0:
                heuristic_triage = "usable"
            elif len(rooms) >= 1 and len(walls) >= 4 and area_m2 >= 15.0:
                heuristic_triage = "needs_fixes"
            else:
                heuristic_triage = "unusable"

            # Render overlay
            orig_img = Image.open(img_path).convert("RGB")
            w_orig, h_orig = orig_img.size
            scale = min(512 / w_orig, 512 / h_orig)
            nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
            pad_x = (512 - nw) // 2
            pad_y = (512 - nh) // 2

            canvas = Image.new("RGB", (512, 512), (255, 255, 255))
            canvas.paste(orig_img.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))
            draw = ImageDraw.Draw(canvas, "RGBA")

            # Draw rooms
            for r in rooms:
                pts = r.get("polygon", [])
                if len(pts) >= 3:
                    pts_px = [(int(p[0]), int(p[1])) for p in pts]
                    draw.polygon(pts_px, fill=(60, 180, 75, 70), outline=(30, 120, 45, 200))

            # Draw walls
            for w in walls:
                sp = w.get("start_point", [0, 0])
                ep = w.get("end_point", [0, 0])
                col = (255, 50, 50, 220) if w.get("is_exterior", False) else (50, 50, 255, 200)
                draw.line([(int(sp[0]), int(sp[1])), (int(ep[0]), int(ep[1]))], fill=col, width=2)

            # Draw doors
            for d in doors:
                pos = d.get("position", [0, 0])
                draw.ellipse([(pos[0]-4, pos[1]-4), (pos[0]+4, pos[1]+4)], fill=(255, 200, 0, 240), outline=(0, 0, 0, 255))

            overlay_fn = f"{plan_id}_overlay.png"
            canvas.save(out_dir / overlay_fn)

            record = {
                "id": plan_id,
                "filename": filename,
                "walls_count": len(walls),
                "rooms_count": len(rooms),
                "doors_count": len(doors),
                "windows_count": len(windows),
                "shell_valid": is_shell_valid,
                "shell_status": shell_status,
                "area_m2": round(area_m2, 2),
                "perimeter_m": round(perim_m, 2),
                "triage": heuristic_triage,
                "overlay_path": str(out_dir / overlay_fn)
            }
            results.append(record)
            print(f"  -> Shell: {shell_status} | Rooms: {len(rooms)} | Doors: {len(doors)} | Area: {area_m2:.1f} m2 | Triage: {heuristic_triage}")

        except Exception as e:
            print(f"  [!] Inference failed for {plan_id}: {e}")
            results.append({
                "id": plan_id,
                "filename": filename,
                "error": str(e),
                "shell_valid": False,
                "rooms_count": 0,
                "doors_count": 0,
                "triage": "unusable"
            })

    triage_counts = {
        "usable": sum(1 for r in results if r.get("triage") == "usable"),
        "needs_fixes": sum(1 for r in results if r.get("triage") == "needs_fixes"),
        "unusable": sum(1 for r in results if r.get("triage") == "unusable")
    }

    summary = {
        "run_name": run_name,
        "total_plans": len(manifest),
        "valid_shells": valid_shells_count,
        "valid_shells_pct": round(100.0 * valid_shells_count / len(manifest), 1),
        "total_rooms_found": total_rooms_found,
        "total_doors_found": total_doors_found,
        "triage_summary": triage_counts,
        "results": results
    }

    summary_file = out_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print(f"DEV SET EVALUATION FINISHED ({run_name})")
    print(f"Valid Shells: {valid_shells_count}/{len(manifest)} ({summary['valid_shells_pct']}%)")
    print(f"Triage: {triage_counts['usable']} Usable, {triage_counts['needs_fixes']} Needs Fixes, {triage_counts['unusable']} Unusable")
    print(f"Results saved to: {summary_file}")
    print("=" * 100)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="Optional model checkpoint path")
    parser.add_argument("--run-name", type=str, default="baseline_v1_1", help="Name for this evaluation run")
    args = parser.parse_args()
    evaluate_dev_dataset(model_weights_path=args.weights, run_name=args.run_name)
