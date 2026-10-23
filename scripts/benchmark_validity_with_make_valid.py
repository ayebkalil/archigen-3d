"""
benchmark_validity_with_make_valid.py
Re-measures Single-Pass vs 4-Fold TTA shell validity on all 400 validation floorplans of CubiCasa5K
under the unified strict Shapely is_valid definition with integrated make_valid repair.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def evaluate_val_validity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"RE-MEASURING SINGLE-PASS vs TTA VALIDITY WITH MAKE_VALID (Device: {device})")
    print("Dataset: CubiCasa5K Validation Split (N = 400 plans)")
    print("=" * 80)

    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_path = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    model = MultiTaskFloorplanUNet(
        encoder_name="resnet34",
        encoder_weights=None,
        num_struct_classes=5,
        num_room_classes=12
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    vectorizer = FloorplanVectorizer()

    single_valid_count = 0
    single_repaired_count = 0
    tta_valid_count = 0
    tta_repaired_count = 0

    # Discordance tracking
    single_valid_tta_invalid = []
    tta_valid_single_invalid = []
    both_valid = 0
    both_invalid = 0

    total_plans = len(dataset)
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            img = batch["image"].to(device) # (1, 3, 512, 512)

            # --- A. Single-Pass Inference ---
            out_sp = model(img)
            p_struct_sp = F.softmax(out_sp["struct"], dim=1)
            p_room_sp = F.softmax(out_sp["room"], dim=1)
            pred_s_sp = torch.argmax(p_struct_sp, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_r_sp = torch.argmax(p_room_sp, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            mask_sp = np.zeros((512, 512), dtype=np.uint8)
            mask_sp[(pred_s_sp == 1) | (pred_s_sp == 2)] = 1
            mask_sp[pred_s_sp == 3] = 2
            mask_sp[pred_s_sp == 4] = 3

            layout_sp = vectorizer.vectorize(mask=mask_sp, struct_mask=pred_s_sp, room_mask=pred_r_sp)
            sp_valid = layout_sp["metadata"]["is_building_shell_valid"]
            sp_repaired = layout_sp["metadata"].get("was_shell_repaired", False)

            if sp_valid:
                single_valid_count += 1
            if sp_repaired:
                single_repaired_count += 1

            # --- B. 4-Fold TTA Inference ---
            out_h = model(torch.flip(img, dims=[3]))
            out_v = model(torch.flip(img, dims=[2]))
            out_hv = model(torch.flip(img, dims=[2, 3]))

            p_s_tta = (p_struct_sp +
                       torch.flip(F.softmax(out_h["struct"], dim=1), dims=[3]) +
                       torch.flip(F.softmax(out_v["struct"], dim=1), dims=[2]) +
                       torch.flip(F.softmax(out_hv["struct"], dim=1), dims=[2, 3])) / 4.0

            p_r_tta = (p_room_sp +
                       torch.flip(F.softmax(out_h["room"], dim=1), dims=[3]) +
                       torch.flip(F.softmax(out_v["room"], dim=1), dims=[2]) +
                       torch.flip(F.softmax(out_hv["room"], dim=1), dims=[2, 3])) / 4.0

            pred_s_tta = torch.argmax(p_s_tta, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_r_tta = torch.argmax(p_r_tta, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            mask_tta = np.zeros((512, 512), dtype=np.uint8)
            mask_tta[(pred_s_tta == 1) | (pred_s_tta == 2)] = 1
            mask_tta[pred_s_tta == 3] = 2
            mask_tta[pred_s_tta == 4] = 3

            layout_tta = vectorizer.vectorize(mask=mask_tta, struct_mask=pred_s_tta, room_mask=pred_r_tta)
            tta_valid = layout_tta["metadata"]["is_building_shell_valid"]
            tta_repaired = layout_tta["metadata"].get("was_shell_repaired", False)

            if tta_valid:
                tta_valid_count += 1
            if tta_repaired:
                tta_repaired_count += 1

            if sp_valid and not tta_valid:
                single_valid_tta_invalid.append(idx)
            elif tta_valid and not sp_valid:
                tta_valid_single_invalid.append(idx)
            elif sp_valid and tta_valid:
                both_valid += 1
            else:
                both_invalid += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total_plans} floorplans...")

    elapsed = time.time() - t0
    sp_pct = (single_valid_count / total_plans) * 100.0
    tta_pct = (tta_valid_count / total_plans) * 100.0
    fallback_valid = both_valid + len(single_valid_tta_invalid) + len(tta_valid_single_invalid)
    fallback_pct = (fallback_valid / total_plans) * 100.0

    print("\n" + "=" * 80)
    print("EMPIRICAL VALIDITY RESULTS (N = 400 plans):")
    print("=" * 80)
    print(f"1. Single-Pass Validity:    {single_valid_count}/400 ({sp_pct:.2f}%) | Shells Repaired: {single_repaired_count}")
    print(f"2. 4-Fold TTA Validity:     {tta_valid_count}/400 ({tta_pct:.2f}%) | Shells Repaired: {tta_repaired_count}")
    print(f"3. Single Valid, TTA Broken: {len(single_valid_tta_invalid)} plans ({len(single_valid_tta_invalid)/total_plans*100:.2f}%)")
    print(f"4. TTA Valid, Single Broken: {len(tta_valid_single_invalid)} plans ({len(tta_valid_single_invalid)/total_plans*100:.2f}%)")
    print(f"5. Both Valid:              {both_valid}/400 ({both_valid/total_plans*100:.2f}%)")
    print(f"6. Both Invalid:            {both_invalid}/400 ({both_invalid/total_plans*100:.2f}%)")
    print(f"7. Combined Fallback:       {fallback_valid}/400 ({fallback_pct:.2f}%)")
    print(f"Evaluation Time: {elapsed:.2f}s ({elapsed/total_plans*1000:.1f} ms/plan)")

    results = {
        "dataset_size": total_plans,
        "single_pass_valid": single_valid_count,
        "single_pass_valid_pct": round(sp_pct, 2),
        "single_pass_repaired": single_repaired_count,
        "tta_valid": tta_valid_count,
        "tta_valid_pct": round(tta_pct, 2),
        "tta_repaired": tta_repaired_count,
        "single_valid_tta_invalid_count": len(single_valid_tta_invalid),
        "tta_valid_single_invalid_count": len(tta_valid_single_invalid),
        "both_valid": both_valid,
        "both_invalid": both_invalid,
        "fallback_valid": fallback_valid,
        "fallback_valid_pct": round(fallback_pct, 2),
        "elapsed_sec": round(elapsed, 2)
    }

    out_file = PROJECT_ROOT / "outputs/validity_with_make_valid_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to: {out_file}")

if __name__ == "__main__":
    evaluate_val_validity()
