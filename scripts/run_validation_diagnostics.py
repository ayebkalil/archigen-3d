"""
Comprehensive Diagnostic Suite on CubiCasa5K Validation Split (400 plans).
Diagnoses:
1. Exact per-class Room IoU and pixel frequencies.
2. Signed room count error (over-splitting vs merging).
3. Metric breakdown by CubiCasa drawing style (colorful, high_quality, high_quality_architectural).
4. Re-measurement of exterior perimeter error.
5. True structural class IoU values (Wall_Ext, Wall_Int, Door, Window) to verify no duplicate copy.
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, STRUCTURE_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DIAGNOSTICS] Running on device: {device}")

    # 1. Load validation dataset
    val_dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"[DATA] Validation floorplans: {len(val_dataset)}")

    # 2. Load model
    weights_path = "models/saved/floorplan_unetpp/multitask_resnet34_best.pth"
    model = MultiTaskFloorplanUNet(
        encoder_name="resnet34",
        num_struct_classes=5,
        num_room_classes=12
    ).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vectorizer = FloorplanVectorizer()

    # Metric accumulators: Global
    struct_names = {1: "Wall_Internal", 2: "Wall_External", 3: "Door", 4: "Window"}
    s_inter = {c: 0 for c in range(1, 5)}
    s_union = {c: 0 for c in range(1, 5)}
    merged_wall_inter, merged_wall_union = 0, 0

    room_names = {v: k for k, v in ROOM_CLASSES.items() if v > 0}
    r_inter = {c: 0 for c in range(1, 12)}
    r_union = {c: 0 for c in range(1, 12)}
    r_gt_pixels = {c: 0 for c in range(1, 12)}
    r_correct_px, r_total_px = 0, 0

    # Signed room error accumulators
    signed_room_errors = []
    abs_room_errors = []
    gt_room_counts = []
    pred_room_counts = []

    perimeter_errors = []

    # Style-based accumulators
    # Style is the folder name, e.g. "colorful", "high_quality", "high_quality_architectural"
    style_stats = defaultdict(lambda: {
        "count": 0,
        "m_wall_inter": 0, "m_wall_union": 0,
        "wall_int_inter": 0, "wall_int_union": 0,
        "wall_ext_inter": 0, "wall_ext_union": 0,
        "door_inter": 0, "door_union": 0,
        "win_inter": 0, "win_union": 0,
        "r_inter": {c: 0 for c in range(1, 12)},
        "r_union": {c: 0 for c in range(1, 12)},
        "r_correct_px": 0, "r_total_px": 0,
        "signed_room_errors": [],
        "perimeter_errors": []
    })

    sample_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            gt_struct = batch["struct_mask"].numpy()
            gt_room = batch["room_mask"].numpy()
            gt_mask4 = batch["mask"].numpy()
            b_size = imgs.size(0)

            outputs = model(imgs)
            pred_s5 = torch.argmax(outputs["struct"], dim=1).cpu().numpy()
            pred_r = torch.argmax(outputs["room"], dim=1).cpu().numpy()

            for b in range(b_size):
                p_s = pred_s5[b]
                p_r = pred_r[b]
                t_s = gt_struct[b]
                t_r = gt_room[b]
                t_m4 = gt_mask4[b]

                sample_path = val_dataset.valid_samples[sample_idx]
                style = sample_path.split("/")[0] if "/" in sample_path else sample_path.split("\\")[0]
                sample_idx += 1

                st = style_stats[style]
                st["count"] += 1

                # 1. Structural metrics
                p_wall_merged = (p_s == 1) | (p_s == 2)
                t_wall_merged = (t_s == 1) | (t_s == 2)
                mw_i = np.logical_and(p_wall_merged, t_wall_merged).sum()
                mw_u = np.logical_or(p_wall_merged, t_wall_merged).sum()
                merged_wall_inter += mw_i
                merged_wall_union += mw_u
                st["m_wall_inter"] += mw_i
                st["m_wall_union"] += mw_u

                for c in range(1, 5):
                    c_i = np.logical_and(p_s == c, t_s == c).sum()
                    c_u = np.logical_or(p_s == c, t_s == c).sum()
                    s_inter[c] += c_i
                    s_union[c] += c_u
                    if c == 1:
                        st["wall_int_inter"] += c_i; st["wall_int_union"] += c_u
                    elif c == 2:
                        st["wall_ext_inter"] += c_i; st["wall_ext_union"] += c_u
                    elif c == 3:
                        st["door_inter"] += c_i; st["door_union"] += c_u
                    elif c == 4:
                        st["win_inter"] += c_i; st["win_union"] += c_u

                # 2. Room metrics per class
                non_bg = (t_r > 0)
                corr_px = np.logical_and(p_r == t_r, non_bg).sum()
                tot_px = non_bg.sum()
                r_correct_px += corr_px
                r_total_px += tot_px
                st["r_correct_px"] += corr_px
                st["r_total_px"] += tot_px

                for c in range(1, 12):
                    p_c = (p_r == c)
                    t_c = (t_r == c)
                    rc_i = np.logical_and(p_c, t_c).sum()
                    rc_u = np.logical_or(p_c, t_c).sum()
                    r_inter[c] += rc_i
                    r_union[c] += rc_u
                    r_gt_pixels[c] += t_c.sum()
                    st["r_inter"][c] += rc_i
                    st["r_union"][c] += rc_u

                # 3. Vectorizer & Room Count Analysis
                p_m4 = np.zeros_like(p_s)
                p_m4[(p_s == 1) | (p_s == 2)] = 1
                p_m4[p_s == 3] = 2
                p_m4[p_s == 4] = 3

                layout = vectorizer.vectorize(mask=p_m4, struct_mask=p_s, room_mask=p_r)
                meta = layout.get("metadata", {})
                pred_scale = meta.get("scale_meters_per_pixel", 0.030)
                pred_rooms = len(layout.get("rooms", []))

                # GT room count: count contours >= 100px across all room classes
                gt_rooms = 0
                for c in range(1, 12):
                    c_mask = (t_r == c).astype(np.uint8)
                    cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    gt_rooms += sum(1 for cnt in cnts if cv2.contourArea(cnt) >= 100)

                signed_err = pred_rooms - gt_rooms
                signed_room_errors.append(signed_err)
                abs_room_errors.append(abs(signed_err))
                gt_room_counts.append(gt_rooms)
                pred_room_counts.append(pred_rooms)
                st["signed_room_errors"].append(signed_err)

                # 4. Perimeter error
                gt_scale = float(batch["gt_scale"][b]) if "gt_scale" in batch else pred_scale
                gt_ext_mask = (t_s == 2).astype(np.uint8)
                gt_cnts, _ = cv2.findContours(gt_ext_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if gt_cnts:
                    gt_perim_m = sum(cv2.arcLength(c, True) for c in gt_cnts) * gt_scale
                    pred_perim_m = meta.get("building_perimeter_m", 0)
                    if gt_perim_m > 1.0:
                        rel_p_err = abs(pred_perim_m - gt_perim_m) / gt_perim_m
                        perimeter_errors.append(min(rel_p_err, 2.0))
                        st["perimeter_errors"].append(min(rel_p_err, 2.0))

            if sample_idx % 100 == 0:
                print(f"  Processed {sample_idx}/{len(val_dataset)} plans...")

    # PRINT DIAGNOSTIC RESULTS
    print("\n" + "="*80)
    print("1. STRUCTURAL VERIFIED METRICS (VAL SPLIT, N=400)")
    print("="*80)
    mw_iou = merged_wall_inter / max(merged_wall_union, 1) * 100
    print(f"Merged Wall IoU:       {mw_iou:.2f}% (Inter: {merged_wall_inter:,}, Union: {merged_wall_union:,})")
    for c in range(1, 5):
        c_iou = s_inter[c] / max(s_union[c], 1) * 100
        print(f"  - {struct_names[c]:<15} IoU: {c_iou:.2f}% (Inter: {s_inter[c]:,}, Union: {s_union[c]:,})")
    
    # 4-class mean vs 3-class mean
    mean_4 = (s_inter[1]/s_union[1] + s_inter[2]/s_union[2] + s_inter[3]/s_union[3] + s_inter[4]/s_union[4]) / 4 * 100
    mean_3 = (mw_iou + s_inter[3]/s_union[3]*100 + s_inter[4]/s_union[4]*100) / 3
    print(f"\nStructure Mean IoU (4-class: Ext, Int, Door, Win): {mean_4:.2f}%")
    print(f"Structure Mean IoU (3-class: Merged Wall, Door, Win): {mean_3:.2f}%")

    print("\n" + "="*80)
    print("2. EXACT PER-CLASS ROOM IoU & DISTRIBUTION (VAL SPLIT, N=400)")
    print("="*80)
    print(f"{'Class ID':<10} {'Room Name':<16} {'IoU (%)':<10} {'GT Pixels':<14} {'Share (%)':<10}")
    print("-" * 60)
    tot_room_px = sum(r_gt_pixels.values())
    valid_room_ious = []
    for c in range(1, 12):
        iou_c = (r_inter[c] / max(r_union[c], 1)) * 100 if r_union[c] > 0 else 0.0
        share_c = (r_gt_pixels[c] / max(tot_room_px, 1)) * 100
        valid_room_ious.append(iou_c)
        print(f"{c:<10} {room_names[c]:<16} {iou_c:<10.2f} {r_gt_pixels[c]:<14,} {share_c:<10.2f}%")
    
    macro_room_miou = np.mean(valid_room_ious)
    overall_room_acc = (r_correct_px / max(r_total_px, 1)) * 100
    print("-" * 60)
    print(f"Macro Room mIoU:       {macro_room_miou:.2f}%")
    print(f"Room Pixel Accuracy:   {overall_room_acc:.2f}%")

    print("\n" + "="*80)
    print("3. SIGNED ROOM COUNT ERROR ANALYSIS (OVER-SPLITTING VS MERGING)")
    print("="*80)
    signed_arr = np.array(signed_room_errors)
    abs_arr = np.array(abs_room_errors)
    gt_arr = np.array(gt_room_counts)
    pred_arr = np.array(pred_room_counts)

    print(f"Mean GT Rooms per Plan:   {gt_arr.mean():.2f} (std: {gt_arr.std():.2f})")
    print(f"Mean Pred Rooms per Plan: {pred_arr.mean():.2f} (std: {pred_arr.std():.2f})")
    print(f"Mean Absolute Error (MAE):{abs_arr.mean():.2f}")
    print(f"Mean Signed Error:        {signed_arr.mean():+.2f}")
    print(f"  -> Plans Over-segmented (pred > gt): {int((signed_arr > 0).sum())} ({((signed_arr > 0).mean()*100):.1f}%)")
    print(f"  -> Plans Exact match    (pred == gt):{int((signed_arr == 0).sum())} ({((signed_arr == 0).mean()*100):.1f}%)")
    print(f"  -> Plans Under-segmented(pred < gt): {int((signed_arr < 0).sum())} ({((signed_arr < 0).mean()*100):.1f}%)")
    print(f"Median Signed Error:      {np.median(signed_arr):+.1f}")

    print("\n" + "="*80)
    print("4. EXTERIOR PERIMETER RELATIVE ERROR")
    print("="*80)
    if perimeter_errors:
        p_arr = np.array(perimeter_errors) * 100
        print(f"Mean Exterior Perimeter Rel Error:   {p_arr.mean():.2f}%")
        print(f"Median Exterior Perimeter Rel Error: {np.median(p_arr):.2f}%")

    print("\n" + "="*80)
    print("5. PERFORMANCE BREAKDOWN BY CUBICASA STYLE")
    print("="*80)
    print(f"{'Style':<28} {'Count':<6} {'Wall Merged':<12} {'Door':<8} {'Window':<8} {'Room mIoU':<10} {'Room Acc':<10} {'Signed Rm Err':<14}")
    print("-" * 100)
    for style, sdata in style_stats.items():
        n = sdata["count"]
        w_m = (sdata["m_wall_inter"] / max(sdata["m_wall_union"], 1)) * 100
        d_m = (sdata["door_inter"] / max(sdata["door_union"], 1)) * 100
        win_m = (sdata["win_inter"] / max(sdata["win_union"], 1)) * 100
        r_ious = [(sdata["r_inter"][c] / max(sdata["r_union"][c], 1)) * 100 for c in range(1, 12) if sdata["r_union"][c] > 0]
        r_miou = np.mean(r_ious) if r_ious else 0.0
        r_acc = (sdata["r_correct_px"] / max(sdata["r_total_px"], 1)) * 100
        mean_sgn = np.mean(sdata["signed_room_errors"]) if sdata["signed_room_errors"] else 0.0
        print(f"{style:<28} {n:<6} {w_m:<12.2f} {d_m:<8.2f} {win_m:<8.2f} {r_miou:<10.2f} {r_acc:<10.2f} {mean_sgn:<+14.2f}")
    print("="*100)

if __name__ == "__main__":
    run_diagnostics()
