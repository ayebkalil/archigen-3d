"""
Paired Bootstrap Significance Engine for ArchiGen 3D.
Evaluates Base Model (25ep) vs Fine-Tuned Model on the unseen Test Split (N=400 plans).
Performs 1,000 paired bootstrap resamples over plans to compute:
- 95% Confidence Intervals for Base and Fine-Tuned metrics.
- 95% Confidence Interval for Delta (Fine-Tuned - Base).
- Empirical p-value testing H0: Delta <= 0 (is the gain statistically significant?).
- Per-class breakdowns for Living, Bed, Kitchen, Bath, Garage, etc.
- Direct Footprint IoU for both models on test plans.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet

def compute_footprint_inter_union(pred_s: np.ndarray, pred_r: np.ndarray, gt_s: np.ndarray, gt_r: np.ndarray):
    # Envelope = Walls or Rooms
    p_env = ((pred_s == 1) | (pred_s == 2) | (pred_r > 0)).astype(np.uint8)
    g_env = ((gt_s == 1) | (gt_s == 2) | (gt_r > 0)).astype(np.uint8)

    def fill_contour(img):
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(img)
        if cnts:
            cv2.drawContours(filled, cnts, -1, 1, -1)
        return filled

    p_filled = fill_contour(p_env)
    g_filled = fill_contour(g_env)

    inter = int(np.logical_and(p_filled, g_filled).sum())
    union = int(np.logical_or(p_filled, g_filled).sum())
    return inter, union

def collect_per_plan_stats(ckpt_path: str, dataset: CubiCasaDataset, device: torch.device):
    print(f"[*] Extracting plan-by-plan statistics for: {Path(ckpt_path).name}")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    plans_stats = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()

            outputs = model(imgs)
            p_s = torch.argmax(outputs["struct"], dim=1).cpu().numpy().astype(np.uint8)
            p_r = torch.argmax(outputs["room"], dim=1).cpu().numpy().astype(np.uint8)

            B = imgs.shape[0]
            for b in range(B):
                # 1. Merged Wall
                p_mw = ((p_s[b] == 1) | (p_s[b] == 2))
                t_mw = ((t_s[b] == 1) | (t_s[b] == 2))
                mw_i = int(np.logical_and(p_mw, t_mw).sum())
                mw_u = int(np.logical_or(p_mw, t_mw).sum())

                # 2. Door
                d_i = int(np.logical_and(p_s[b] == 3, t_s[b] == 3).sum())
                d_u = int(np.logical_or(p_s[b] == 3, t_s[b] == 3).sum())

                # 3. Window
                w_i = int(np.logical_and(p_s[b] == 4, t_s[b] == 4).sum())
                w_u = int(np.logical_or(p_s[b] == 4, t_s[b] == 4).sum())

                # 4. Room Pixel Accuracy
                non_bg = (t_r[b] > 0)
                r_corr = int(np.logical_and(p_r[b] == t_r[b], non_bg).sum())
                r_tot = int(non_bg.sum())

                # 5. Room Classes 1..10
                r_i = np.zeros(12, dtype=np.int64)
                r_u = np.zeros(12, dtype=np.int64)
                for c in range(1, 11):
                    r_i[c] = int(np.logical_and(p_r[b] == c, t_r[b] == c).sum())
                    r_u[c] = int(np.logical_or(p_r[b] == c, t_r[b] == c).sum())

                # 6. Footprint IoU
                fp_i, fp_u = compute_footprint_inter_union(p_s[b], p_r[b], t_s[b], t_r[b])

                plans_stats.append({
                    "mw_i": mw_i, "mw_u": mw_u,
                    "d_i": d_i, "d_u": d_u,
                    "w_i": w_i, "w_u": w_u,
                    "r_corr": r_corr, "r_tot": r_tot,
                    "r_i": r_i, "r_u": r_u,
                    "fp_i": fp_i, "fp_u": fp_u
                })

            if (b_idx + 1) % 25 == 0:
                print(f"    Processed {min((b_idx+1)*4, len(dataset))}/{len(dataset)} plans...")

    return plans_stats

def run_paired_bootstrap(base_stats: List[Dict], ft_stats: List[Dict], n_bootstraps: int = 1000, seed: int = 42):
    print(f"\n[*] Running {n_bootstraps} paired bootstrap resamples over {len(base_stats)} test plans (Seed={seed})...")
    np.random.seed(seed)
    N = len(base_stats)

    inv_rooms = {v: k for k, v in ROOM_CLASSES.items()}

    # Metric accumulators for bootstrap distributions
    keys_to_track = [
        "Room Pixel Acc", "Room mIoU (10c)", "Merged Wall IoU", "Door IoU", "Window IoU", "Footprint IoU",
        "LivingRoom IoU", "Bedroom IoU", "Kitchen IoU", "Bath IoU", "Entry IoU", "Garage IoU"
    ]
    boot_base = {k: np.zeros(n_bootstraps) for k in keys_to_track}
    boot_ft = {k: np.zeros(n_bootstraps) for k in keys_to_track}
    boot_delta = {k: np.zeros(n_bootstraps) for k in keys_to_track}

    for b in range(n_bootstraps):
        idxs = np.random.choice(N, size=N, replace=True)

        def aggregate(stats_list, sample_idxs):
            mw_i = sum(stats_list[i]["mw_i"] for i in sample_idxs)
            mw_u = sum(stats_list[i]["mw_u"] for i in sample_idxs)
            d_i = sum(stats_list[i]["d_i"] for i in sample_idxs)
            d_u = sum(stats_list[i]["d_u"] for i in sample_idxs)
            w_i = sum(stats_list[i]["w_i"] for i in sample_idxs)
            w_u = sum(stats_list[i]["w_u"] for i in sample_idxs)
            r_corr = sum(stats_list[i]["r_corr"] for i in sample_idxs)
            r_tot = sum(stats_list[i]["r_tot"] for i in sample_idxs)
            fp_i = sum(stats_list[i]["fp_i"] for i in sample_idxs)
            fp_u = sum(stats_list[i]["fp_u"] for i in sample_idxs)

            r_i_tot = np.zeros(12, dtype=np.int64)
            r_u_tot = np.zeros(12, dtype=np.int64)
            for i in sample_idxs:
                r_i_tot += stats_list[i]["r_i"]
                r_u_tot += stats_list[i]["r_u"]

            room_ious = [r_i_tot[c] / max(r_u_tot[c], 1) for c in range(1, 11)]

            return {
                "Room Pixel Acc": (r_corr / max(r_tot, 1)) * 100,
                "Room mIoU (10c)": np.mean(room_ious) * 100,
                "Merged Wall IoU": (mw_i / max(mw_u, 1)) * 100,
                "Door IoU": (d_i / max(d_u, 1)) * 100,
                "Window IoU": (w_i / max(w_u, 1)) * 100,
                "Footprint IoU": (fp_i / max(fp_u, 1)) * 100,
                "LivingRoom IoU": (r_i_tot[1] / max(r_u_tot[1], 1)) * 100,
                "Bedroom IoU": (r_i_tot[2] / max(r_u_tot[2], 1)) * 100,
                "Kitchen IoU": (r_i_tot[3] / max(r_u_tot[3], 1)) * 100,
                "Bath IoU": (r_i_tot[4] / max(r_u_tot[4], 1)) * 100,
                "Entry IoU": (r_i_tot[5] / max(r_u_tot[5], 1)) * 100,
                "Garage IoU": (r_i_tot[9] / max(r_u_tot[9], 1)) * 100
            }

        res_base = aggregate(base_stats, idxs)
        res_ft = aggregate(ft_stats, idxs)

        for k in keys_to_track:
            boot_base[k][b] = res_base[k]
            boot_ft[k][b] = res_ft[k]
            boot_delta[k][b] = res_ft[k] - res_base[k]

    print("\n" + "=" * 95)
    print("PAIRED BOOTSTRAP STATISTICAL SIGNIFICANCE SUMMARY (1,000 Resamples, N=400 Test Plans)")
    print("=" * 95)
    header = f"{'Metric':<18} | {'Base Mean [95% CI]':<22} | {'Fine-Tuned [95% CI]':<22} | {'Delta [95% CI]':<20} | {'p-value':<7}"
    print(header)
    print("-" * 95)

    summary_results = {}

    for k in keys_to_track:
        b_mean = np.mean(boot_base[k])
        b_ci = (np.percentile(boot_base[k], 2.5), np.percentile(boot_base[k], 97.5))

        ft_mean = np.mean(boot_ft[k])
        ft_ci = (np.percentile(boot_ft[k], 2.5), np.percentile(boot_ft[k], 97.5))

        d_mean = np.mean(boot_delta[k])
        d_ci = (np.percentile(boot_delta[k], 2.5), np.percentile(boot_delta[k], 97.5))

        # Empirical p-value testing if Delta is significantly different from 0
        if d_mean >= 0:
            p_val = np.mean(boot_delta[k] <= 0)
        else:
            p_val = np.mean(boot_delta[k] >= 0)

        sig_flag = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        sign = "+" if d_mean >= 0 else ""

        base_str = f"{b_mean:5.2f}% [{b_ci[0]:4.1f}, {b_ci[1]:4.1f}]"
        ft_str = f"{ft_mean:5.2f}% [{ft_ci[0]:4.1f}, {ft_ci[1]:4.1f}]"
        delta_str = f"{sign}{d_mean:4.2f} [{d_ci[0]:+4.2f}, {d_ci[1]:+4.2f}]"

        print(f"{k:<18} | {base_str:<22} | {ft_str:<22} | {delta_str:<20} | {p_val:5.3f} {sig_flag}")
        summary_results[k] = {
            "base_mean": b_mean, "base_ci": b_ci,
            "ft_mean": ft_mean, "ft_ci": ft_ci,
            "delta_mean": d_mean, "delta_ci": d_ci,
            "p_val": p_val, "sig": sig_flag
        }

    print("=" * 95)
    print("Significance Legend: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. = not significant (p >= 0.05)")
    return summary_results

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {dev}")

    dataset = CubiCasaDataset(split="test", img_size=512, augment=False)

    base_ckpt = "models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth"
    ft_ckpt = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"

    base_stats = collect_per_plan_stats(base_ckpt, dataset, dev)
    ft_stats = collect_per_plan_stats(ft_ckpt, dataset, dev)

    run_paired_bootstrap(base_stats, ft_stats, n_bootstraps=1000, seed=42)

if __name__ == "__main__":
    main()
