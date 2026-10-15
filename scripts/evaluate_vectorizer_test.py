"""
Dedicated Vectorizer Post-Processing Ablation on Unseen Test Split (N=400 plans).
Evaluates the geometric post-processing pipeline with the neural model FIXED
to the production fine-tuned checkpoint (multitask_resnet34_finetuned_best.pth):
- Old Vectorizer: Raw per-class connected components without wall masking or fragment absorption.
- New Hybrid Vectorizer: Wall-masked semantic room partitioning + sub-threshold fragment absorption.

Metrics:
- Room Count MAE (vs Ground Truth SVG rooms)
- Mean Signed Error (measuring over-segmentation bias)
- Instance Precision, Recall, and F1 at IoU >= 0.5
- Watertight Building Shell Rate (%)
- Total Interior Area MAE (m^2)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer

def extract_old_rooms(room_mask: np.ndarray, scale_m_per_px: float):
    rooms = []
    if room_mask is None or np.max(room_mask) == 0:
        return rooms

    for r_id in range(1, 12):
        r_bin = (room_mask == r_id).astype(np.uint8)
        cnts, _ = cv2.findContours(r_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area_px = cv2.contourArea(c)
            if area_px < 100:
                continue
            r_mask = np.zeros_like(r_bin)
            cv2.drawContours(r_mask, [c], -1, 1, -1)
            rooms.append({
                "class_id": r_id,
                "area_px": area_px,
                "mask": r_mask
            })
    return rooms

def extract_hybrid_rooms(vectorizer: FloorplanVectorizer, mask_4class: np.ndarray, struct_pred: np.ndarray, room_pred: np.ndarray, scale_m_per_px: float):
    layout = vectorizer.vectorize(mask=mask_4class, struct_mask=struct_pred, room_mask=room_pred)
    return layout

def compute_instance_matching(pred_masks, gt_masks, iou_thresh=0.5):
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
    return tp, fp, fn

def main(num_plans: int = 400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Vectorizer Test Ablation on {num_plans} plans ({device})...")

    dataset = CubiCasaDataset(split="test", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_path = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    vectorizer_new = FloorplanVectorizer()
    scale_m_per_px = 0.03

    old_errors, new_errors = [], []
    old_signed, new_signed = [], []
    old_tp, old_fp, old_fn = 0, 0, 0
    new_tp, new_fp, new_fn = 0, 0, 0
    new_shells_valid = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= num_plans:
                break

            img = batch["image"].to(device)
            gt_s = batch["struct_mask"][0].numpy()
            gt_r = batch["room_mask"][0].numpy()

            outputs = model(img)
            pred_s = torch.argmax(outputs["struct"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_r = torch.argmax(outputs["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            mask_4class = np.zeros_like(pred_s)
            mask_4class[(pred_s == 1) | (pred_s == 2)] = 1
            mask_4class[pred_s == 3] = 2
            mask_4class[pred_s == 4] = 3

            # GT room instances from SVG
            gt_masks = []
            for c_id in range(1, 12):
                c_bin = (gt_r == c_id).astype(np.uint8)
                num_c, labels_c = cv2.connectedComponents(c_bin)
                for lbl in range(1, num_c):
                    m = (labels_c == lbl)
                    if m.sum() > 50:
                        gt_masks.append(m)
            n_gt = len(gt_masks)

            # Old Vectorizer rooms
            old_room_objs = extract_old_rooms(pred_r, scale_m_per_px)
            n_old = len(old_room_objs)
            old_masks = [r["mask"] for r in old_room_objs]

            # New Hybrid Vectorizer rooms
            layout_new = extract_hybrid_rooms(vectorizer_new, mask_4class, pred_s, pred_r, scale_m_per_px)
            new_room_objs = layout_new.get("rooms", [])
            n_new = len(new_room_objs)

            if layout_new.get("metadata", {}).get("is_building_shell_valid", False):
                new_shells_valid += 1

            # Convert new polygon rooms back to raster mask for instance matching
            new_masks = []
            for r in new_room_objs:
                poly = r.get("polygon_meters", [])
                if len(poly) >= 3:
                    pts_px = np.array([[int(round(pt[0] / scale_m_per_px)), int(round(pt[1] / scale_m_per_px))] for pt in poly], dtype=np.int32)
                    r_m = np.zeros_like(pred_r)
                    cv2.fillPoly(r_m, [pts_px], 1)
                    if r_m.sum() > 20:
                        new_masks.append(r_m)

            old_errors.append(abs(n_old - n_gt))
            old_signed.append(n_old - n_gt)
            new_errors.append(abs(n_new - n_gt))
            new_signed.append(n_new - n_gt)

            o_tp, o_fp, o_fn = compute_instance_matching(old_masks, gt_masks)
            n_tp, n_fp, n_fn = compute_instance_matching(new_masks, gt_masks)
            old_tp += o_tp; old_fp += o_fp; old_fn += o_fn
            new_tp += n_tp; new_fp += n_fp; new_fn += n_fn

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{num_plans} plans... (Old MAE: {np.mean(old_errors):.2f} | New MAE: {np.mean(new_errors):.2f})")

    old_prec = old_tp / max(old_tp + old_fp, 1) * 100
    old_rec = old_tp / max(old_tp + old_fn, 1) * 100
    old_f1 = 2 * old_prec * old_rec / max(old_prec + old_rec, 1e-4)

    new_prec = new_tp / max(new_tp + new_fp, 1) * 100
    new_rec = new_tp / max(new_tp + new_fn, 1) * 100
    new_f1 = 2 * new_prec * new_rec / max(new_prec + new_rec, 1e-4)

    shell_valid_pct = (new_shells_valid / num_plans) * 100

    print("\n" + "=" * 75)
    print("VECTORIZER POST-PROCESSING ABLATION (N=400 Unseen Test Plans)")
    print("Neural Model Fixed: multitask_resnet34_finetuned_best.pth")
    print("=" * 75)
    print(f"{'Metric':<35} | {'Old Vectorizer':<16} | {'New Hybrid':<16}")
    print("-" * 75)
    print(f"{'Room Count MAE':<35} | {np.mean(old_errors):<16.2f} | {np.mean(new_errors):<16.2f}")
    print(f"{'Signed Room Error (Over-split)':<35} | {np.mean(old_signed):<+16.2f} | {np.mean(new_signed):<+16.2f}")
    print(f"{'Instance Precision (IoU >= 0.5)':<35} | {old_prec:<15.2f}% | {new_prec:<15.2f}%")
    print(f"{'Instance Recall    (IoU >= 0.5)':<35} | {old_rec:<15.2f}% | {new_rec:<15.2f}%")
    print(f"{'Instance F1-Score  (IoU >= 0.5)':<35} | {old_f1:<15.2f}% | {new_f1:<15.2f}%")
    print(f"{'Watertight Building Shell Rate':<35} | {'N/A':<16} | {shell_valid_pct:<15.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    main(400)
