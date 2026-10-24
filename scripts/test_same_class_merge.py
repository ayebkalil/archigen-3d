"""
Benchmark Same-Class Adjacency Merge in Vectorizer on CubiCasa Validation Plans.
Evaluates:
  1. Baseline Vectorizer (without merge)
  2. Same-Class Merged Vectorizer (merging weak-wall gaps, preserving strong walls)
Measures:
  - Instance Precision, Recall, F1 @ IoU >= 0.5
  - Room Count MAE
  - Room Count Signed Bias
"""

import sys
import json
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

def extract_gt_instances(gt_room_mask):
    instances = []
    for c in range(1, 12):
        c_bin = (gt_room_mask == c).astype(np.uint8)
        num_c, labels_c = cv2.connectedComponents(c_bin)
        for lbl in range(1, num_c):
            m = (labels_c == lbl).astype(np.uint8)
            if m.sum() >= 20:
                instances.append({"class_id": c, "mask": m})
    return instances

def compute_instance_f1(pred_masks, gt_masks, iou_thresh=0.5):
    tp = 0
    matched_gt = set()
    for p in pred_masks:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_masks):
            if g_idx in matched_gt:
                continue
            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn

def vectorize_with_optional_same_class_merge(vec, mask_4class, struct_pred, room_pred, enable_merge=False):
    # Standard vectorization first
    layout = vec.vectorize(mask=mask_4class, struct_mask=struct_pred, room_mask=room_pred)
    if not enable_merge:
        return layout

    # If merge enabled: inspect rooms of same class
    rooms = layout.get("rooms", [])
    if len(rooms) <= 1:
        return layout

    h, w = mask_4class.shape
    wall_clean = (mask_4class == 1).astype(np.uint8)

    # Render each room polygon to mask
    room_masks = []
    for r in rooms:
        pts = r.get("polygon", [])
        m = np.zeros((h, w), dtype=np.uint8)
        if len(pts) >= 3:
            pts_px = np.array(pts, dtype=np.int32)
            cv2.fillPoly(m, [pts_px], 1)
        room_masks.append({
            "class_id": r.get("class_id", 0),
            "room_type": r.get("room_type", "Room"),
            "mask": m,
            "poly": pts,
            "merged": False
        })

    # Adjacency check for same class
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged_rooms = []
    num_rooms = len(room_masks)

    for i in range(num_rooms):
        if room_masks[i]["merged"]:
            continue
        curr_m = room_masks[i]["mask"]
        curr_cls = room_masks[i]["class_id"]

        for j in range(i + 1, num_rooms):
            if room_masks[j]["merged"]:
                continue
            if room_masks[j]["class_id"] != curr_cls:
                continue

            other_m = room_masks[j]["mask"]
            # Check gap between them
            dil_i = cv2.dilate(curr_m, kernel_3, iterations=2)
            gap = np.logical_and(dil_i > 0, other_m > 0)
            if not gap.any():
                continue

            # Check wall evidence in the gap between the two same-class rooms
            # Dilate both slightly to capture the seam
            dil_j = cv2.dilate(other_m, kernel_3, iterations=2)
            seam = np.logical_and(dil_i > 0, dil_j > 0)
            if not seam.any():
                continue

            wall_in_seam = np.logical_and(seam, wall_clean > 0)
            wall_ratio = wall_in_seam.sum() / max(seam.sum(), 1)

            # Rule: only merge if wall evidence is weak (< 65% solid wall barrier in seam)
            # If wall_ratio >= 0.65, strong wall evidence exists (e.g. partition wall between two bedrooms)
            if wall_ratio < 0.65:
                # Merge room j into room i
                curr_m = np.logical_or(curr_m, other_m).astype(np.uint8)
                room_masks[j]["merged"] = True

        room_masks[i]["mask"] = curr_m
        merged_rooms.append(room_masks[i])

    # Reconstruct room instances from merged masks
    final_instances = []
    scale = layout.get("metadata", {}).get("scale_m_per_px", 0.03)
    for r in merged_rooms:
        if r["merged"]:
            continue
        cnts, _ = cv2.findContours(r["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area_m2 = cv2.contourArea(c) * (scale ** 2)
        if area_m2 >= vec.min_area_habitable_m2:
            pts = c.squeeze().tolist()
            if len(pts) >= 3:
                final_instances.append({
                    "class_id": r["class_id"],
                    "room_type": r["room_type"],
                    "polygon": pts,
                    "area_m2": round(area_m2, 2),
                    "mask": r["mask"]
                })

    layout["rooms"] = final_instances
    return layout

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST MERGE] Evaluating on {device}...")

    ds = CubiCasaDataset(split="val", img_size=512, augment=False, cache_dir="data/CubiCasa5k/cached_512")
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    champ_p = Path("models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth")
    ckpt = torch.load(champ_p, map_location=device, weights_only=False)
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    vec = FloorplanVectorizer(min_area_habitable_m2=2.0)

    # Accumulators for 100 validation plans
    N_PLANS = 100
    base_tp, base_fp, base_fn = 0, 0, 0
    merge_tp, merge_fp, merge_fn = 0, 0, 0

    base_errs, merge_errs = [], []
    base_bias, merge_bias = [], []

    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= N_PLANS:
                break
            imgs = batch["image"].to(device)
            out = model(imgs)
            pr_s = torch.argmax(out["struct"], dim=1).cpu().numpy().astype(np.uint8)
            pr_r = torch.argmax(out["room"], dim=1).cpu().numpy().astype(np.uint8)
            gt_r = batch["room_mask"].numpy().astype(np.uint8)

            for i in range(len(imgs)):
                if count >= N_PLANS:
                    break
                m4 = np.zeros_like(pr_s[i])
                m4[(pr_s[i] == 1) | (pr_s[i] == 2)] = 1
                m4[pr_s[i] == 3] = 2
                m4[pr_s[i] == 4] = 3

                # Ground truth instances
                gt_insts = extract_gt_instances(gt_r[i])
                gt_masks = [g["mask"] for g in gt_insts]
                n_gt = len(gt_masks)

                # 1. Base vectorizer
                lay_base = vec.vectorize(m4, room_mask=pr_r[i], struct_mask=pr_s[i])
                base_masks = []
                for r in lay_base.get("rooms", []):
                    pts = r.get("polygon", [])
                    m = np.zeros_like(m4)
                    if len(pts) >= 3:
                        cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 1)
                    base_masks.append(m)

                _, _, _, tp_b, fp_b, fn_b = compute_instance_f1(base_masks, gt_masks)
                base_tp += tp_b
                base_fp += fp_b
                base_fn += fn_b
                base_errs.append(abs(len(base_masks) - n_gt))
                base_bias.append(len(base_masks) - n_gt)

                # 2. Vectorizer with Same-Class Merge
                lay_merge = vectorize_with_optional_same_class_merge(vec, m4, pr_s[i], pr_r[i], enable_merge=True)
                merge_masks = []
                for r in lay_merge.get("rooms", []):
                    pts = r.get("polygon", [])
                    m = np.zeros_like(m4)
                    if len(pts) >= 3:
                        cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 1)
                    merge_masks.append(m)
                _, _, _, tp_m, fp_m, fn_m = compute_instance_f1(merge_masks, gt_masks)
                merge_tp += tp_m
                merge_fp += fp_m
                merge_fn += fn_m
                merge_errs.append(abs(len(merge_masks) - n_gt))
                merge_bias.append(len(merge_masks) - n_gt)

                count += 1

    # Final metrics
    b_prec = base_tp / max(base_tp + base_fp, 1) * 100
    b_rec = base_tp / max(base_tp + base_fn, 1) * 100
    b_f1 = 2 * b_prec * b_rec / max(b_prec + b_rec, 1e-6)

    m_prec = merge_tp / max(merge_tp + merge_fp, 1) * 100
    m_rec = merge_tp / max(merge_tp + merge_fn, 1) * 100
    m_f1 = 2 * m_prec * m_rec / max(m_prec + m_rec, 1e-6)

    print("\n" + "=" * 80)
    print(f"SAME-CLASS ADJACENCY MERGE ABLATION (N={N_PLANS} CubiCasa Validation Plans)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Base Vectorizer':<18} | {'With Same-Class Merge':<22}")
    print("-" * 80)
    print(f"{'Instance Precision @ IoU 0.5':<28} | {b_prec:6.2f}%            | {m_prec:6.2f}%")
    print(f"{'Instance Recall @ IoU 0.5':<28} | {b_rec:6.2f}%            | {m_rec:6.2f}%")
    print(f"{'Instance F1 @ IoU 0.5':<28} | {b_f1:6.2f}%            | {m_f1:6.2f}%")
    print(f"{'Room Count MAE':<28} | {np.mean(base_errs):6.3f} rooms       | {np.mean(merge_errs):6.3f} rooms")
    print(f"{'Room Count Signed Bias':<28} | {np.mean(base_bias):+6.3f} rooms       | {np.mean(merge_bias):+6.3f} rooms")
    print("=" * 80)

if __name__ == "__main__":
    main()
