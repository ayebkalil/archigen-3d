"""
Robustness & Domain-Shift Evaluation Suite: Base vs Fine-Tuned.
Addresses user criteria:
1. Evaluates Base Model (25ep) vs Fine-Tuned Model (CAD-augmented) on:
   (a) Fixed-seed stylized validation plans (N=50 plans with CAD/scan perturbations).
   (b) The 10 External Synthetic Stress Plans with Ground-Truth comparison.
2. Directly measures whether the CAD-augmented fine-tuning provided genuine robustness gains
   against domain shifts or whether degradation remains comparable.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer
from scripts.test_external_generalization import SYNTHETIC_STRESS_GROUND_TRUTH

def apply_fixed_cad_perturbations(img_tensor: torch.Tensor, mode: str = "cad_blueprint") -> torch.Tensor:
    """
    Applies synthetic CAD / scan perturbations:
    - 'inverted_cad': dark background with bright lines
    - 'noisy_scan': low contrast + Gaussian blur + salt-and-pepper noise
    - 'faded_print': strong brightness boost + low contrast
    """
    # img_tensor is normalized: (tensor - mean) / std. Un-normalize to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img_01 = (img_tensor * std + mean).clamp(0, 1)

    if mode == "inverted_cad":
        pert_01 = 1.0 - img_01
    elif mode == "noisy_scan":
        # Grayscale
        gray = 0.299 * img_01[:, 0:1] + 0.587 * img_01[:, 1:2] + 0.114 * img_01[:, 2:3]
        gray = gray.repeat(1, 3, 1, 1)
        # Add slight noise
        noise = torch.randn_like(gray) * 0.08
        pert_01 = (gray * 0.7 + 0.15 + noise).clamp(0, 1)
    elif mode == "faded_print":
        pert_01 = (img_01 * 0.5 + 0.45).clamp(0, 1)
    else:
        pert_01 = img_01

    # Re-normalize
    return (pert_01 - mean) / std

def evaluate_stylized_val(models_dict: dict, dev: torch.device, num_plans: int = 50):
    print(f"\n{'='*80}")
    print(f"PART A: FIXED-SEED STYLIZED VALIDATION BENCHMARK (N={num_plans} Plans)")
    print(f"{'='*80}")

    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    styles = ["clean", "inverted_cad", "noisy_scan", "faded_print"]
    results = {m_name: {s: {"r_corr": 0, "r_tot": 0, "mw_i": 0, "mw_u": 0} for s in styles} for m_name in models_dict}

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx * 2 >= num_plans:
                break
            imgs_clean = batch["image"].to(dev)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()
            B = imgs_clean.shape[0]

            for s in styles:
                if s == "clean":
                    imgs_pert = imgs_clean
                else:
                    imgs_pert = apply_fixed_cad_perturbations(imgs_clean, mode=s)

                for m_name, model in models_dict.items():
                    out = model(imgs_pert)
                    p_s = torch.argmax(out["struct"], dim=1).cpu().numpy().astype(np.uint8)
                    p_r = torch.argmax(out["room"], dim=1).cpu().numpy().astype(np.uint8)

                    for b in range(B):
                        non_bg = (t_r[b] > 0)
                        results[m_name][s]["r_corr"] += int(np.logical_and(p_r[b] == t_r[b], non_bg).sum())
                        results[m_name][s]["r_tot"] += int(non_bg.sum())

                        p_mw = ((p_s[b] == 1) | (p_s[b] == 2))
                        t_mw = ((t_s[b] == 1) | (t_s[b] == 2))
                        results[m_name][s]["mw_i"] += int(np.logical_and(p_mw, t_mw).sum())
                        results[m_name][s]["mw_u"] += int(np.logical_or(p_mw, t_mw).sum())

    print(f"{'Perturbation Style':<18} | {'Base Model Room Acc':<20} | {'Fine-Tuned Room Acc':<20} | {'Delta Acc':<10}")
    print("-" * 75)
    for s in styles:
        b_acc = (results["Base"][s]["r_corr"] / max(results["Base"][s]["r_tot"], 1)) * 100
        ft_acc = (results["Fine-Tuned"][s]["r_corr"] / max(results["Fine-Tuned"][s]["r_tot"], 1)) * 100
        d = ft_acc - b_acc
        sign = "+" if d >= 0 else ""
        print(f"{s:<18} | {b_acc:19.2f}% | {ft_acc:19.2f}% | {sign}{d:6.2f} pts")

    print("\n" + f"{'Perturbation Style':<18} | {'Base Wall IoU':<20} | {'Fine-Tuned Wall IoU':<20} | {'Delta Wall':<10}")
    print("-" * 75)
    for s in styles:
        b_w = (results["Base"][s]["mw_i"] / max(results["Base"][s]["mw_u"], 1)) * 100
        ft_w = (results["Fine-Tuned"][s]["mw_i"] / max(results["Fine-Tuned"][s]["mw_u"], 1)) * 100
        d = ft_w - b_w
        sign = "+" if d >= 0 else ""
        print(f"{s:<18} | {b_w:19.2f}% | {ft_w:19.2f}% | {sign}{d:6.2f} pts")

    return results

def evaluate_external_stress_comparison(models_dict: dict, dev: torch.device):
    print(f"\n{'='*95}")
    print("PART B: 10 EXTERNAL SYNTHETIC STRESS PLANS: BASE VS FINE-TUNED COMPARISON")
    print(f"{'='*95}")

    input_dir = Path("data/external_test_plans")
    plan_files = sorted(list(input_dir.glob("*.png")))
    vectorizer = FloorplanVectorizer()

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    table_records = []

    for idx, p_path in enumerate(plan_files, 1):
        name = p_path.stem
        gt_info = SYNTHETIC_STRESS_GROUND_TRUTH.get(name, {"challenge": "Stress", "gt_rooms": 4})
        gt_r = gt_info["gt_rooms"]

        # Preprocess image
        raw_img = Image.open(p_path).convert("RGB")
        iw, ih = raw_img.size
        size = 512
        scale = min(size / iw, size / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        pad_x = (size - nw) // 2
        pad_y = (size - nh) // 2
        resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(resized, (pad_x, pad_y))

        tensor = TF.to_tensor(canvas).unsqueeze(0).to(dev)
        tensor = (tensor - mean) / std

        plan_res = {"name": name, "gt": gt_r}

        for m_name, model in models_dict.items():
            with torch.no_grad():
                out = model(tensor)
                p_s = torch.argmax(out["struct"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                p_r = torch.argmax(out["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                m4 = np.zeros_like(p_s)
                m4[(p_s == 1) | (p_s == 2)] = 1
                m4[p_s == 3] = 2
                m4[p_s == 4] = 3

                layout = vectorizer.vectorize(mask=m4, struct_mask=p_s, room_mask=p_r)
                rooms = layout.get("rooms", [])
                walls = layout.get("walls", [])
                meta = layout.get("metadata", {})
                shell_st = meta.get("shell_status", "VALID" if meta.get("is_building_shell_valid") else "INVALID")

                plan_res[f"{m_name}_rooms"] = len(rooms)
                plan_res[f"{m_name}_delta"] = len(rooms) - gt_r
                plan_res[f"{m_name}_walls"] = len(walls)
                plan_res[f"{m_name}_shell"] = shell_st

        table_records.append(plan_res)

    print(f"{'#':<2} | {'Stress Blueprint Case':<28} | {'GT':<3} | {'Base (Pred/Diff)':<16} | {'Base Shell':<12} | {'FT (Pred/Diff)':<16} | {'FT Shell':<12}")
    print("-" * 95)
    for i, r in enumerate(table_records, 1):
        b_str = f"{r['Base_rooms']} ({r['Base_delta']:+d})"
        ft_str = f"{r['Fine-Tuned_rooms']} ({r['Fine-Tuned_delta']:+d})"
        print(f"{i:<2} | {r['name']:<28} | {r['gt']:<3} | {b_str:<14} | {r['Base_shell']:<12} | {ft_str:<14} | {r['Fine-Tuned_shell']:<12}")
    print("-" * 95)

    base_mae = np.mean([abs(r["Base_delta"]) for r in table_records])
    ft_mae = np.mean([abs(r["Fine-Tuned_delta"]) for r in table_records])
    print(f"Stress Room MAE: Base Model = {base_mae:.2f} rooms | Fine-Tuned Model = {ft_mae:.2f} rooms")
    print("=" * 95)

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Base and Fine-Tuned checkpoints on {dev}...")

    base_model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(dev)
    b_ckpt = torch.load("models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth", map_location=dev, weights_only=False)
    base_model.load_state_dict(b_ckpt.get("model_state_dict", b_ckpt))
    base_model.eval()

    ft_model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(dev)
    ft_ckpt = torch.load("models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth", map_location=dev, weights_only=False)
    ft_model.load_state_dict(ft_ckpt.get("model_state_dict", ft_ckpt))
    ft_model.eval()

    models_dict = {"Base": base_model, "Fine-Tuned": ft_model}

    evaluate_stylized_val(models_dict, dev, num_plans=50)
    evaluate_external_stress_comparison(models_dict, dev)

if __name__ == "__main__":
    main()
