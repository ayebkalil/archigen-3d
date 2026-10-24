"""
Authoritative Final Test Split Benchmark Script (CubiCasa5K Test N=400)
Evaluates all 4 systems on the unseen 400 CubiCasa5K test floorplans:
  1. Champion v1.1
  2. Blend alpha=0.3
  3. Challenger (Restyled)
  4. Full Routed Pipeline (Champion -> Fallback Blend a=0.3)

Strict Metric Definitions:
  - 10-Class Room mIoU: Headlined non-background room classes (includes Office, IoU=0)
  - 9-Class Active Room mIoU: Active residential rooms (excludes Office)
  - 11-Class All-Foreground Room mIoU: Includes generic 'Other' room
  - True Merged Wall IoU: Binary wall vs non-wall union IoU
  - Mean Wall IoU: Arithmetic mean of Exterior and Interior wall IoU
  - Door & Window IoU
  - Room Count MAE & Mean Signed Bias vs GT
  - Building Shell Validity (%)
  - Fallback Trigger Rate (%)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

# Class definitions
ACTIVE_9_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]     # Living, Bed, Kitchen, Bath, Entry, Balcony, Storage, Closet, Garage
HABITABLE_10_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # + Office
ALL_11_CLASSES = list(range(1, 12))                 # + Other Room

def count_gt_rooms(gt_room_mask):
    total = 0
    for c in range(1, 12):
        c_bin = (gt_room_mask == c).astype(np.uint8)
        num_c, _ = cv2.connectedComponents(c_bin)
        total += max(0, num_c - 1)
    return total

def run_test_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BENCHMARK] Executing final evaluation on CubiCasa5K TEST split (N=400)...", flush=True)
    print(f"[BENCHMARK] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    ds = CubiCasaDataset(
        root_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cubicasa5k/cubicasa5k"),
        split="test",
        img_size=512,
        augment=False,
        cache_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cached_512")
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    # Load Model Checkpoints
    champ_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    blend_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth"
    chal_p = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth"

    def load_model(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
        m.load_state_dict(ckpt.get("model_state_dict", ckpt))
        m.eval()
        return m

    m_champ = load_model(champ_p)
    m_blend = load_model(blend_p)
    m_chal = load_model(chal_p)

    vec = FloorplanVectorizer(min_area_habitable_m2=2.0)

    systems = ["Champion v1.1", "Blend a=0.3", "Challenger", "Routed System"]
    results = {
        s: {
            "room_cm": torch.zeros((12, 12), dtype=torch.int64, device=device),
            "struct_cm": torch.zeros((5, 5), dtype=torch.int64, device=device),
            "room_abs_errors": [],
            "room_signed_errors": [],
            "valid_shells": 0,
            "fallback_triggers": 0,
            "total_evaluated": 0
        }
        for s in systems
    }

    t0 = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            gt_struct = batch["struct_mask"].to(device).long()
            gt_room = batch["room_mask"].to(device).long()
            gt_room_np = batch["room_mask"].numpy().astype(np.uint8)

            out_champ = m_champ(imgs)
            out_blend = m_blend(imgs)
            out_chal = m_chal(imgs)

            pr_champ_s = torch.argmax(out_champ["struct"], dim=1)
            pr_champ_r = torch.argmax(out_champ["room"], dim=1)

            pr_blend_s = torch.argmax(out_blend["struct"], dim=1)
            pr_blend_r = torch.argmax(out_blend["room"], dim=1)

            pr_chal_s = torch.argmax(out_chal["struct"], dim=1)
            pr_chal_r = torch.argmax(out_chal["room"], dim=1)

            # Accumulate standalone CMs
            results["Champion v1.1"]["room_cm"] += torch.bincount(gt_room.view(-1) * 12 + pr_champ_r.view(-1), minlength=144).reshape(12, 12)
            results["Champion v1.1"]["struct_cm"] += torch.bincount(gt_struct.view(-1) * 5 + pr_champ_s.view(-1), minlength=25).reshape(5, 5)

            results["Blend a=0.3"]["room_cm"] += torch.bincount(gt_room.view(-1) * 12 + pr_blend_r.view(-1), minlength=144).reshape(12, 12)
            results["Blend a=0.3"]["struct_cm"] += torch.bincount(gt_struct.view(-1) * 5 + pr_blend_s.view(-1), minlength=25).reshape(5, 5)

            results["Challenger"]["room_cm"] += torch.bincount(gt_room.view(-1) * 12 + pr_chal_r.view(-1), minlength=144).reshape(12, 12)
            results["Challenger"]["struct_cm"] += torch.bincount(gt_struct.view(-1) * 5 + pr_chal_s.view(-1), minlength=25).reshape(5, 5)

            # Per-plan vectorization
            pr_champ_s_np = pr_champ_s.cpu().numpy().astype(np.uint8)
            pr_champ_r_np = pr_champ_r.cpu().numpy().astype(np.uint8)

            pr_blend_s_np = pr_blend_s.cpu().numpy().astype(np.uint8)
            pr_blend_r_np = pr_blend_r.cpu().numpy().astype(np.uint8)

            pr_chal_s_np = pr_chal_s.cpu().numpy().astype(np.uint8)
            pr_chal_r_np = pr_chal_r.cpu().numpy().astype(np.uint8)

            for i in range(len(imgs)):
                gt_n_rooms = count_gt_rooms(gt_room_np[i])

                # Standalone Champion vectorization
                m4_c = np.zeros_like(pr_champ_s_np[i])
                m4_c[(pr_champ_s_np[i] == 1) | (pr_champ_s_np[i] == 2)] = 1
                m4_c[pr_champ_s_np[i] == 3] = 2
                m4_c[pr_champ_s_np[i] == 4] = 3
                lay_c = vec.vectorize(m4_c, room_mask=pr_champ_r_np[i], struct_mask=pr_champ_s_np[i])
                n_c = len(lay_c.get("rooms", []))
                area_c = lay_c.get("metadata", {}).get("total_surface_m2", 0.0)
                valid_c = lay_c.get("metadata", {}).get("is_building_shell_valid", False)

                results["Champion v1.1"]["room_abs_errors"].append(abs(n_c - gt_n_rooms))
                results["Champion v1.1"]["room_signed_errors"].append(n_c - gt_n_rooms)
                if valid_c:
                    results["Champion v1.1"]["valid_shells"] += 1

                # Standalone Blend vectorization
                m4_bl = np.zeros_like(pr_blend_s_np[i])
                m4_bl[(pr_blend_s_np[i] == 1) | (pr_blend_s_np[i] == 2)] = 1
                m4_bl[pr_blend_s_np[i] == 3] = 2
                m4_bl[pr_blend_s_np[i] == 4] = 3
                lay_bl = vec.vectorize(m4_bl, room_mask=pr_blend_r_np[i], struct_mask=pr_blend_s_np[i])
                n_bl = len(lay_bl.get("rooms", []))
                valid_bl = lay_bl.get("metadata", {}).get("is_building_shell_valid", False)

                results["Blend a=0.3"]["room_abs_errors"].append(abs(n_bl - gt_n_rooms))
                results["Blend a=0.3"]["room_signed_errors"].append(n_bl - gt_n_rooms)
                if valid_bl:
                    results["Blend a=0.3"]["valid_shells"] += 1

                # Standalone Challenger vectorization
                m4_ch = np.zeros_like(pr_chal_s_np[i])
                m4_ch[(pr_chal_s_np[i] == 1) | (pr_chal_s_np[i] == 2)] = 1
                m4_ch[pr_chal_s_np[i] == 3] = 2
                m4_ch[pr_chal_s_np[i] == 4] = 3
                lay_ch = vec.vectorize(m4_ch, room_mask=pr_chal_r_np[i], struct_mask=pr_chal_s_np[i])
                n_ch = len(lay_ch.get("rooms", []))
                results["Challenger"]["room_abs_errors"].append(abs(n_ch - gt_n_rooms))
                results["Challenger"]["room_signed_errors"].append(n_ch - gt_n_rooms)
                if lay_ch.get("metadata", {}).get("is_building_shell_valid", False):
                    results["Challenger"]["valid_shells"] += 1

                # Refined Routed System Trigger:
                # Trigger fallback ONLY on genuine structural collapses:
                # 0 rooms, invalid shell, or single-room artifact under 8.0 m2
                is_trigger = (n_c == 0) or (not valid_c) or (n_c == 1 and area_c < 8.0)

                if is_trigger:
                    results["Routed System"]["fallback_triggers"] += 1
                    # Adopt fallback if it detected more rooms, or if primary shell was invalid and fallback is valid
                    if n_bl > n_c or (not valid_c and valid_bl):
                        chosen_lay = lay_bl
                        chosen_pr_r = pr_blend_r[i:i+1]
                        chosen_pr_s = pr_blend_s[i:i+1]
                    else:
                        chosen_lay = lay_c
                        chosen_pr_r = pr_champ_r[i:i+1]
                        chosen_pr_s = pr_champ_s[i:i+1]
                else:
                    chosen_lay = lay_c
                    chosen_pr_r = pr_champ_r[i:i+1]
                    chosen_pr_s = pr_champ_s[i:i+1]

                n_routed = len(chosen_lay.get("rooms", []))
                results["Routed System"]["room_abs_errors"].append(abs(n_routed - gt_n_rooms))
                results["Routed System"]["room_signed_errors"].append(n_routed - gt_n_rooms)
                if chosen_lay.get("metadata", {}).get("is_building_shell_valid", False):
                    results["Routed System"]["valid_shells"] += 1

                gt_r_single = gt_room[i:i+1]
                gt_s_single = gt_struct[i:i+1]
                results["Routed System"]["room_cm"] += torch.bincount(gt_r_single.view(-1) * 12 + chosen_pr_r.view(-1), minlength=144).reshape(12, 12)
                results["Routed System"]["struct_cm"] += torch.bincount(gt_s_single.view(-1) * 5 + chosen_pr_s.view(-1), minlength=25).reshape(5, 5)

            if (b_idx + 1) % 25 == 0:
                print(f"  Processed {b_idx + 1}/100 batches ({(b_idx + 1)*4}/400 floorplans)...", flush=True)

    elapsed = time.time() - t0
    print(f"[BENCHMARK] Inference and evaluation completed in {elapsed:.1f}s.")

    # Calculate metrics
    summary = {}
    for s in systems:
        r_cm = results[s]["room_cm"]
        s_cm = results[s]["struct_cm"]

        r_tp = torch.diag(r_cm).float()
        r_union = r_cm.sum(dim=1).float() + r_cm.sum(dim=0).float() - r_tp
        r_ious = torch.where(r_union > 0, r_tp / r_union, torch.zeros_like(r_tp)).cpu().numpy() * 100.0

        s_tp = torch.diag(s_cm).float()
        s_union = s_cm.sum(dim=1).float() + s_cm.sum(dim=0).float() - s_tp
        s_ious = torch.where(s_union > 0, s_tp / s_union, torch.zeros_like(s_tp)).cpu().numpy() * 100.0

        # Exact denominators
        miou_10cl = float(np.mean([r_ious[c] for c in HABITABLE_10_CLASSES]))
        miou_9cl = float(np.mean([r_ious[c] for c in ACTIVE_9_CLASSES]))
        miou_11cl = float(np.mean([r_ious[c] for c in ALL_11_CLASSES]))

        # Wall metrics
        # Merged Wall IoU: treating class 1 (interior) and 2 (exterior) as one binary wall class
        wall_tp = float(s_cm[1:3, 1:3].sum())
        wall_union = float(s_cm[1:3, :].sum() + s_cm[:, 1:3].sum() - s_cm[1:3, 1:3].sum())
        wall_merged_iou = float((wall_tp / wall_union) * 100.0) if wall_union > 0 else 0.0

        wall_int = float(s_ious[1])
        wall_ext = float(s_ious[2])
        wall_mean = float((wall_int + wall_ext) / 2.0)

        door_iou = float(s_ious[3])
        win_iou = float(s_ious[4])

        room_mae = float(np.mean(results[s]["room_abs_errors"]))
        room_bias = float(np.mean(results[s]["room_signed_errors"]))
        shell_rate = float(results[s]["valid_shells"] / 400.0 * 100.0)
        fb_rate = float(results[s]["fallback_triggers"] / 400.0 * 100.0)

        summary[s] = {
            "room_mIoU_10cl": round(miou_10cl, 2),
            "room_mIoU_9cl": round(miou_9cl, 2),
            "room_mIoU_11cl": round(miou_11cl, 2),
            "wall_merged_iou": round(wall_merged_iou, 2),
            "wall_mean_iou": round(wall_mean, 2),
            "wall_ext_iou": round(wall_ext, 2),
            "wall_int_iou": round(wall_int, 2),
            "door_iou": round(door_iou, 2),
            "win_iou": round(win_iou, 2),
            "room_count_mae": round(room_mae, 3),
            "room_count_bias": round(room_bias, 3),
            "shell_valid_pct": round(shell_rate, 2),
            "fallback_trigger_pct": round(fb_rate, 2),
            "fallback_trigger_count": results[s]["fallback_triggers"]
        }

    # Print Table
    print("\n" + "=" * 125, flush=True)
    print("AUTHORITATIVE UNBIASED BENCHMARK (CubiCasa5K TEST Split N=400, Threshold = 2.0 m2)")
    print("=" * 125, flush=True)
    header = f"{'Metric':<34} | " + " | ".join([f"{s:<18}" for s in systems])
    print(header, flush=True)
    print("-" * 125, flush=True)

    rows = [
        ("room_mIoU_10cl", "Room mIoU (10 Habitable Classes)", "%"),
        ("room_mIoU_9cl", "Room mIoU (9 Active Classes)", "%"),
        ("room_mIoU_11cl", "Room mIoU (11 All-Fg Classes)", "%"),
        ("wall_merged_iou", "True Merged Wall IoU", "%"),
        ("wall_mean_iou", "Mean Ext & Int Wall IoU", "%"),
        ("wall_ext_iou", "Wall Ext. IoU", "%"),
        ("wall_int_iou", "Wall Int. IoU", "%"),
        ("door_iou", "Door IoU", "%"),
        ("win_iou", "Window IoU", "%"),
        ("room_count_mae", "Room Count MAE", " rooms"),
        ("room_count_bias", "Room Count Signed Bias", " rooms"),
        ("shell_valid_pct", "Building Shell Valid", "%"),
        ("fallback_trigger_pct", "Fallback Trigger Rate", "%")
    ]

    for key, name, unit in rows:
        r_str = f"{name:<34} | "
        for s in systems:
            val = summary[s][key]
            r_str += f"{val:6.2f}{unit:<12} | "
        print(r_str, flush=True)
    print("=" * 125, flush=True)

    out_json = PROJECT_ROOT / "outputs" / "authoritative_test400_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[BENCHMARK] Authoritative test benchmark saved to {out_json}", flush=True)

if __name__ == "__main__":
    run_test_benchmark()
