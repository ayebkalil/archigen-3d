"""
Compare Dev Plans Performance across Champion, Blends, and Challenger
Evaluates all 20 dev plans across:
  1. Champion v1.1
  2. Blend alpha=0.3
  3. Blend alpha=0.5
  4. Blend alpha=0.7
  5. Challenger
Outputs per-plan detection table, room count distribution, and failure recovery.
"""

import sys
import json
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

def run_dev_comparison():
    manifest_path = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipeline = FloorplanPipeline.get_instance()
    device = pipeline.device

    checkpoints = {
        "Champion v1.1": PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth",
        "Blend a=0.3": PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth",
        "Blend a=0.5": PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_5.pth",
        "Blend a=0.7": PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_7.pth",
        "Challenger": PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth",
    }

    # Pre-read image bytes
    images = {}
    for item in manifest:
        pid = item["id"]
        fname = item["filename"]
        img_p = manifest_path.parent / fname
        with open(img_p, "rb") as f:
            images[pid] = f.read()

    summary = {}

    for name, ckpt_p in checkpoints.items():
        print(f"\n[DEV EVAL] Running {name}...", flush=True)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        if hasattr(pipeline, "multitask_model") and pipeline.multitask_model is not None:
            pipeline.multitask_model.load_state_dict(sd)
            pipeline.multitask_model.eval()
        else:
            pipeline.model.load_state_dict(sd)
            pipeline.model.eval()

        plan_stats = {}
        valid_shells = 0
        zero_room_plans = 0
        total_rooms = 0

        for pid, raw_bytes in images.items():
            res = pipeline.process_image(raw_bytes, tta_mode="none", front_only=True)
            layout = res.get("layout", {})
            meta = layout.get("metadata", {})
            rooms = layout.get("rooms", [])
            walls = layout.get("walls", [])
            is_valid = meta.get("is_building_shell_valid", False)

            n_rooms = len(rooms)
            n_walls = len(walls)
            if is_valid:
                valid_shells += 1
            if n_rooms == 0:
                zero_room_plans += 1
            total_rooms += n_rooms

            plan_stats[pid] = {
                "rooms": n_rooms,
                "walls": n_walls,
                "valid": is_valid
            }

        summary[name] = {
            "valid_shells": valid_shells,
            "zero_room_plans": zero_room_plans,
            "total_rooms": total_rooms,
            "mean_rooms": round(total_rooms / len(images), 2),
            "plan_stats": plan_stats
        }

    # Print summary table
    print("\n" + "=" * 105)
    print("DEV SET (N=20) PERFORMANCE COMPARISON: CHAMPION vs BLENDS vs CHALLENGER")
    print("=" * 105)
    header = f"{'Plan ID':<10} | " + " | ".join([f"{n:<16}" for n in checkpoints.keys()])
    print(header)
    print("-" * 105)

    for pid in images.keys():
        row = f"{pid:<10} | "
        for name in checkpoints.keys():
            st = summary[name]["plan_stats"][pid]
            val_icon = "V" if st["valid"] else "X"
            row += f"{st['rooms']:2d} rms ({val_icon})     | "
        print(row)

    print("-" * 105)
    print(f"{'Valid Shells':<10} | " + " | ".join([f"{summary[n]['valid_shells']}/20 ({summary[n]['valid_shells']*5}%)       " for n in checkpoints.keys()]))
    print(f"{'0-Room Plans':<10} | " + " | ".join([f"{summary[n]['zero_room_plans']} plans          " for n in checkpoints.keys()]))
    print(f"{'Total Rooms':<10} | " + " | ".join([f"{summary[n]['total_rooms']} rooms (avg {summary[n]['mean_rooms']}) " for n in checkpoints.keys()]))
    print("=" * 105)

    out_json = PROJECT_ROOT / "outputs" / "dev_blends_comparison.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DEV EVAL] Results written to {out_json}")

if __name__ == "__main__":
    run_dev_comparison()
