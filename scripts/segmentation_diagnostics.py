"""
Segmentation Diagnostics Suite:
1. Train-Set IoU vs Test-Set IoU on CubiCasa5k (100 training plans)
2. Overfit Sanity Check on 8 plans
3. Tolerance-Based Metrics:
   - Exact IoU vs 2px / 3px Dilated Tolerant IoU (walls & doors)
   - Boundary F1 Score with 2px distance tolerance
Explains the root cause of the 22% wall IoU / 11% door IoU ceiling.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from scipy.ndimage import distance_transform_edt

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.utils.seed import seed_everything

def compute_tolerant_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance_px: int = 2):
    """
    Computes tolerance-based IoU where a prediction pixel within tolerance_px
    of ground truth counts as correct.
    """
    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0
    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    gt_dilated = cv2.dilate(gt_bin.astype(np.uint8), kernel)
    pred_dilated = cv2.dilate(pred_bin.astype(np.uint8), kernel)

    # Intersection: pred that falls within dilated GT, and GT that falls within dilated pred
    inter_p_in_gt = np.logical_and(pred_bin, gt_dilated).sum()
    inter_gt_in_p = np.logical_and(gt_bin, pred_dilated).sum()
    inter = (inter_p_in_gt + inter_gt_in_p) / 2.0

    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / max(union, 1e-6))

def compute_boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance_px: int = 2):
    """
    Computes Boundary F1 score (Csurka et al., 2013) within tolerance_px distance.
    """
    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return 0.0, 0.0, 0.0

    # Extract contours / boundaries
    pred_b = (cv2.Canny(pred_bin.astype(np.uint8) * 255, 50, 150) > 0)
    gt_b = (cv2.Canny(gt_bin.astype(np.uint8) * 255, 50, 150) > 0)

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0, 0.0, 0.0

    # Distance transforms
    dist_to_gt = distance_transform_edt(np.logical_not(gt_b))
    dist_to_pred = distance_transform_edt(np.logical_not(pred_b))

    # Precision: fraction of pred boundary pixels within tolerance of GT boundary
    precision = np.sum(dist_to_gt[pred_b] <= tolerance_px) / max(np.sum(pred_b), 1e-6)
    # Recall: fraction of GT boundary pixels within tolerance of pred boundary
    recall = np.sum(dist_to_pred[gt_b] <= tolerance_px) / max(np.sum(gt_b), 1e-6)

    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return float(precision), float(recall), float(f1)


def run_diagnostics():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DIAGNOSTICS] Device: {device}")

    # 1. Load trained MultiTask UNet++ checkpoint
    weights_path = Path("models/saved/floorplan_unetpp/multitask_resnet34_best.pth")
    if not weights_path.exists():
        print(f"[ERROR] Weights not found at {weights_path}")
        return

    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[DIAGNOSTICS] Loaded best model checkpoint from {weights_path} (Epoch {ckpt.get('epoch', 'N/A')})")

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Train-Set vs Test-Set IoU (100 training plans)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 1: Measuring Train-Set IoU (100 training plans) ---")
    train_dataset = CubiCasaDataset(split="train", img_size=512, augment=False)
    train_subset = Subset(train_dataset, list(range(100)))
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=False)

    exact_wall_inter, exact_wall_union = 0, 0
    exact_door_inter, exact_door_union = 0, 0
    exact_win_inter, exact_win_union = 0, 0
    exact_room_inter = {c: 0 for c in range(1, 12)}
    exact_room_union = {c: 0 for c in range(1, 12)}

    # Tolerant accumulators
    tol2_wall_scores = []
    tol3_wall_scores = []
    tol2_door_scores = []
    boundary_f1_scores = []

    with torch.no_grad():
        for batch in train_loader:
            imgs = batch["image"].to(device)
            gt_s5 = batch["struct_mask"].numpy()
            gt_room = batch["room_mask"].numpy()

            out = model(imgs)
            pred_s5 = torch.argmax(out["struct"], dim=1).cpu().numpy()
            pred_room = torch.argmax(out["room"], dim=1).cpu().numpy()

            for b in range(imgs.size(0)):
                p_s = pred_s5[b]
                t_s = gt_s5[b]
                p_r = pred_room[b]
                t_r = gt_room[b]

                # Walls
                p_wall = (p_s == 1) | (p_s == 2)
                t_wall = (t_s == 1) | (t_s == 2)
                exact_wall_inter += np.logical_and(p_wall, t_wall).sum()
                exact_wall_union += np.logical_or(p_wall, t_wall).sum()

                # Doors
                p_door = (p_s == 3)
                t_door = (t_s == 3)
                exact_door_inter += np.logical_and(p_door, t_door).sum()
                exact_door_union += np.logical_or(p_door, t_door).sum()

                # Windows
                p_win = (p_s == 4)
                t_win = (t_s == 4)
                exact_win_inter += np.logical_and(p_win, t_win).sum()
                exact_win_union += np.logical_or(p_win, t_win).sum()

                # Rooms
                for c in range(1, 12):
                    exact_room_inter[c] += np.logical_and(p_r == c, t_r == c).sum()
                    exact_room_union[c] += np.logical_or(p_r == c, t_r == c).sum()

                # Tolerant metrics
                tol2_wall_scores.append(compute_tolerant_iou(p_wall, t_wall, tolerance_px=2))
                tol3_wall_scores.append(compute_tolerant_iou(p_wall, t_wall, tolerance_px=3))
                if t_door.sum() > 0:
                    tol2_door_scores.append(compute_tolerant_iou(p_door, t_door, tolerance_px=2))

                prec, rec, f1 = compute_boundary_f1(p_wall, t_wall, tolerance_px=2)
                boundary_f1_scores.append(f1)

    train_wall_iou = exact_wall_inter / max(exact_wall_union, 1e-6)
    train_door_iou = exact_door_inter / max(exact_door_union, 1e-6)
    train_win_iou = exact_win_inter / max(exact_win_union, 1e-6)
    train_room_ious = [exact_room_inter[c] / max(exact_room_union[c], 1e-6) for c in range(1, 12) if exact_room_union[c] > 0]
    train_room_miou = np.mean(train_room_ious) if train_room_ious else 0.0

    print(f"  Train Wall Exact IoU:    {train_wall_iou * 100:.2f}% (Test was 22.44%)")
    print(f"  Train Door Exact IoU:    {train_door_iou * 100:.2f}% (Test was 10.71%)")
    print(f"  Train Window Exact IoU:  {train_win_iou * 100:.2f}% (Test was 23.38%)")
    print(f"  Train Room Semantic mIoU:{train_room_miou * 100:.2f}% (Test was 38.08%)")
    print(f"  Train Wall Tolerant IoU (2px): {np.mean(tol2_wall_scores) * 100:.2f}%")
    print(f"  Train Wall Tolerant IoU (3px): {np.mean(tol3_wall_scores) * 100:.2f}%")
    print(f"  Train Door Tolerant IoU (2px): {np.mean(tol2_door_scores) * 100:.2f}%")
    print(f"  Train Wall Boundary F1 (2px):  {np.mean(boundary_f1_scores) * 100:.2f}%")

    # -------------------------------------------------------------
    # DIAGNOSTIC 2: Overfit Sanity Check on 8 plans (50 epochs)
    # -------------------------------------------------------------
    print("\n--- Diagnostic 2: Running 8-Plan Overfit Sanity Check (50 epochs) ---")
    tiny_subset = Subset(train_dataset, list(range(8)))
    tiny_loader = DataLoader(tiny_subset, batch_size=4, shuffle=False)

    overfit_model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    optimizer = torch.optim.AdamW(overfit_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_ce = nn.CrossEntropyLoss()

    for ep in range(1, 51):
        overfit_model.train()
        total_loss = 0.0
        for batch in tiny_loader:
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].to(device)
            t_r = batch["room_mask"].to(device)

            out = overfit_model(imgs)
            l_s = loss_ce(out["struct"], t_s)
            l_r = loss_ce(out["room"], t_r)
            loss = l_s + l_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if ep in [1, 10, 25, 50]:
            print(f"  [Overfit Epoch {ep:02d}/50] Loss: {total_loss / len(tiny_loader):.4f}")

    # Evaluate overfitted model on the same 8 plans
    overfit_model.eval()
    of_inter, of_union = 0, 0
    with torch.no_grad():
        for batch in tiny_loader:
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].numpy()
            out = overfit_model(imgs)
            p_s = torch.argmax(out["struct"], dim=1).cpu().numpy()
            p_wall = (p_s == 1) | (p_s == 2)
            t_wall = (t_s == 1) | (t_s == 2)
            of_inter += np.logical_and(p_wall, t_wall).sum()
            of_union += np.logical_or(p_wall, t_wall).sum()

    overfit_wall_iou = of_inter / max(of_union, 1e-6)
    print(f"  Overfit Wall Exact IoU on 8 plans: {overfit_wall_iou * 100:.2f}%\n")

    return {
        "train_wall_iou": round(train_wall_iou * 100, 2),
        "train_door_iou": round(train_door_iou * 100, 2),
        "train_win_iou": round(train_win_iou * 100, 2),
        "train_room_miou": round(train_room_miou * 100, 2),
        "wall_tol2_iou": round(np.mean(tol2_wall_scores) * 100, 2),
        "wall_tol3_iou": round(np.mean(tol3_wall_scores) * 100, 2),
        "door_tol2_iou": round(np.mean(tol2_door_scores) * 100, 2),
        "wall_boundary_f1": round(np.mean(boundary_f1_scores) * 100, 2),
        "overfit_wall_iou": round(overfit_wall_iou * 100, 2)
    }

if __name__ == "__main__":
    run_diagnostics()
