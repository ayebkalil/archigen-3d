"""
Official Test Split Evaluation Engine for ArchiGen 3D.
Evaluates model checkpoints strictly on the unseen test split (N=400 floorplans).
Computes exact confusion matrices and IoUs:
- Merged Wall IoU, Interior Wall IoU, Exterior Wall IoU
- Door IoU, Window IoU
- Structural 3-class foreground mIoU and 4-class complete mIoU
- 10-Class Room mIoU (Classes 1 to 10)
- Room Pixel Accuracy
- Per-class Room breakdown (Living, Bed, Kitchen, Bath, Entry, Outdoor, Storage, Closet, Garage, Office)
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, STRUCTURE_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet

def evaluate_checkpoint_on_test(ckpt_path: str, device: str = "cuda") -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"[TEST SPLIT EVALUATION] Evaluating: {ckpt_path}")
    print(f"{'='*70}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = CubiCasaDataset(split="test", img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Structure accumulators
    s_inter = np.zeros(5, dtype=np.int64)
    s_union = np.zeros(5, dtype=np.int64)
    mw_inter = 0
    mw_union = 0

    # Room accumulators
    r_inter = np.zeros(12, dtype=np.int64)
    r_union = np.zeros(12, dtype=np.int64)
    r_gt_px = np.zeros(12, dtype=np.int64)
    r_correct = 0
    r_total = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(dev)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()

            outputs = model(imgs)
            p_s = torch.argmax(outputs["struct"], dim=1).cpu().numpy().astype(np.uint8)
            p_r = torch.argmax(outputs["room"], dim=1).cpu().numpy().astype(np.uint8)

            B = imgs.shape[0]
            for b in range(B):
                # Merged Wall (classes 1 and 2)
                p_mw = ((p_s[b] == 1) | (p_s[b] == 2))
                t_mw = ((t_s[b] == 1) | (t_s[b] == 2))
                mw_inter += int(np.logical_and(p_mw, t_mw).sum())
                mw_union += int(np.logical_or(p_mw, t_mw).sum())

                # Per-class Structure
                for c in range(1, 5):
                    s_inter[c] += int(np.logical_and(p_s[b] == c, t_s[b] == c).sum())
                    s_union[c] += int(np.logical_or(p_s[b] == c, t_s[b] == c).sum())

                # Room Pixel Accuracy (non-background pixels)
                non_bg = (t_r[b] > 0)
                r_correct += int(np.logical_and(p_r[b] == t_r[b], non_bg).sum())
                r_total += int(non_bg.sum())

                # Per-class Room IoU
                for c in range(1, 12):
                    r_inter[c] += int(np.logical_and(p_r[b] == c, t_r[b] == c).sum())
                    r_union[c] += int(np.logical_or(p_r[b] == c, t_r[b] == c).sum())
                    r_gt_px[c] += int((t_r[b] == c).sum())

            if (b_idx + 1) % 25 == 0:
                print(f"  Processed {min((b_idx+1)*4, len(dataset))}/{len(dataset)} test floorplans...")

    # Calculate metrics
    wall_merged = mw_inter / max(mw_union, 1)
    wall_int = s_inter[1] / max(s_union[1], 1)
    wall_ext = s_inter[2] / max(s_union[2], 1)
    door = s_inter[3] / max(s_union[3], 1)
    win = s_inter[4] / max(s_union[4], 1)

    struct_3class = (wall_merged + door + win) / 3.0
    struct_4class = (wall_int + wall_ext + door + win) / 4.0

    room_ious = {c: (r_inter[c] / max(r_union[c], 1)) for c in range(1, 12)}
    room_miou_10 = np.mean([room_ious[c] for c in range(1, 11)])
    room_acc = r_correct / max(r_total, 1)

    inv_room_map = {v: k for k, v in ROOM_CLASSES.items()}

    print("\n" + "="*70)
    print("TEST EVALUATION RESULTS (N=400 Unseen Plans)")
    print("="*70)
    print(f"Merged Wall IoU:            {wall_merged*100:6.2f}%")
    print(f"Interior Wall IoU:          {wall_int*100:6.2f}%")
    print(f"Exterior Wall IoU:          {wall_ext*100:6.2f}%")
    print(f"Door IoU:                   {door*100:6.2f}%")
    print(f"Window IoU:                 {win*100:6.2f}%")
    print(f"Structural mIoU (3-class):  {struct_3class*100:6.2f}%")
    print(f"Structural mIoU (4-class):  {struct_4class*100:6.2f}%")
    print("-" * 70)
    print(f"Room Pixel Accuracy:        {room_acc*100:6.2f}%")
    print(f"Room mIoU (10 classes):     {room_miou_10*100:6.2f}%")
    print("\nPer-Class Room Breakdown:")
    for c in range(1, 11):
        name = inv_room_map.get(c, f"Class {c}")
        print(f"  - {name:15s}: {room_ious[c]*100:5.2f}% (GT: {r_gt_px[c]:,} px)")
    print("=" * 70)

    return {
        "checkpoint": ckpt_path,
        "merged_wall_iou": wall_merged,
        "wall_int_iou": wall_int,
        "wall_ext_iou": wall_ext,
        "door_iou": door,
        "win_iou": win,
        "struct_3class_miou": struct_3class,
        "struct_4class_miou": struct_4class,
        "room_miou_10": room_miou_10,
        "room_acc": room_acc,
        "per_class_room_ious": {inv_room_map.get(c, f"Class {c}"): room_ious[c] for c in range(1, 11)}
    }

def main():
    base_ckpt = "models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth"
    finetuned_ckpt = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"

    print("\n--- 1. Evaluating Base Checkpoint (25ep) on Test Split ---")
    base_res = evaluate_checkpoint_on_test(base_ckpt)

    print("\n--- 2. Evaluating Fine-Tuned Checkpoint on Test Split ---")
    finetuned_res = evaluate_checkpoint_on_test(finetuned_ckpt)

    print("\n" + "="*80)
    print("TEST SET COMPARISON SUMMARY (N=400 Unseen Plans)")
    print("="*80)
    print(f"{'Metric':<30} | {'Base Model (25ep)':<18} | {'Fine-Tuned Model':<18} | {'Delta':<10}")
    print("-" * 80)
    metrics_to_compare = [
        ("Merged Wall IoU", "merged_wall_iou"),
        ("Interior Wall IoU", "wall_int_iou"),
        ("Exterior Wall IoU", "wall_ext_iou"),
        ("Door IoU", "door_iou"),
        ("Window IoU", "win_iou"),
        ("Structural mIoU (3-class)", "struct_3class_miou"),
        ("Structural mIoU (4-class)", "struct_4class_miou"),
        ("Room Pixel Accuracy", "room_acc"),
        ("Room mIoU (10 classes)", "room_miou_10")
    ]
    for label, key in metrics_to_compare:
        v_base = base_res[key] * 100
        v_ft = finetuned_res[key] * 100
        delta = v_ft - v_base
        sign = "+" if delta >= 0 else ""
        print(f"{label:<30} | {v_base:16.2f}% | {v_ft:16.2f}% | {sign}{delta:6.2f} pts")
    print("=" * 80)

if __name__ == "__main__":
    main()
