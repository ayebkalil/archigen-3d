"""
Ultra-fast GPU-vectorized Per-Class Room & Structure IoU Evaluation
Evaluates Champion (v1.1), Challenger, and Blended models on CubiCasa5K validation set (N=400).
Optimized for 4GB VRAM GPU: batch_size=4, zero RAM thrashing, GPU confusion matrix via bincount.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet

ROOM_CLASSES = [
    "Background", "Living Room", "Bedroom", "Kitchen",
    "Bathroom", "Entry / Hall", "Balcony / Terrace",
    "Storage", "Closet", "Garage", "Office", "Other"
]

STRUCT_CLASSES = [
    "Background", "Wall (Interior)", "Wall (Exterior)", "Door", "Window"
]

def evaluate_model(model_path, loader, device, num_room_classes=12, num_struct_classes=5):
    t0 = time.perf_counter()
    print(f"\n[EVAL] Loading checkpoint: {model_path.name}...", flush=True)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=num_struct_classes, num_room_classes=num_room_classes).to(device)
    model.load_state_dict(sd)
    model.eval()

    room_cm = torch.zeros((num_room_classes, num_room_classes), dtype=torch.int64, device=device)
    struct_cm = torch.zeros((num_struct_classes, num_struct_classes), dtype=torch.int64, device=device)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            gt_struct = batch["struct_mask"].to(device).long()
            gt_room = batch["room_mask"].to(device).long()

            out = model(imgs)
            pred_s = torch.argmax(out["struct"], dim=1)
            pred_r = torch.argmax(out["room"], dim=1)

            # GPU Confusion matrix accumulation via 1D bincount
            room_cm += torch.bincount(
                gt_room.view(-1) * num_room_classes + pred_r.view(-1),
                minlength=num_room_classes ** 2
            ).reshape(num_room_classes, num_room_classes)

            struct_cm += torch.bincount(
                gt_struct.view(-1) * num_struct_classes + pred_s.view(-1),
                minlength=num_struct_classes ** 2
            ).reshape(num_struct_classes, num_struct_classes)

            if (b_idx + 1) % 25 == 0:
                print(f"       Batch {b_idx + 1}/{len(loader)} processed...", flush=True)

    # Compute metrics on GPU then cast to CPU
    room_tp = torch.diag(room_cm).float()
    room_gt = room_cm.sum(dim=1).float()
    room_pr = room_cm.sum(dim=0).float()
    room_union = room_gt + room_pr - room_tp
    room_ious = torch.where(room_union > 0, room_tp / room_union, torch.zeros_like(room_tp)).cpu().numpy() * 100.0

    struct_tp = torch.diag(struct_cm).float()
    struct_gt = struct_cm.sum(dim=1).float()
    struct_pr = struct_cm.sum(dim=0).float()
    struct_union = struct_gt + struct_pr - struct_tp
    struct_ious = torch.where(struct_union > 0, struct_tp / struct_union, torch.zeros_like(struct_tp)).cpu().numpy() * 100.0

    room_mIoU = float(np.mean(room_ious[1:]))  # Exclude background (class 0)
    struct_mIoU = float(np.mean(struct_ious[1:]))
    pixel_acc = float((room_tp.sum() / max(1, room_cm.sum())).item() * 100.0)
    wall_iou = float(np.mean(struct_ious[1:3]))
    door_iou = float(struct_ious[3])
    window_iou = float(struct_ious[4])

    elapsed = time.perf_counter() - t0
    print(f"[DONE] {model_path.name} in {elapsed:.1f}s | Room mIoU: {room_mIoU:.2f}% | Wall IoU: {wall_iou:.2f}% | Pixel Acc: {pixel_acc:.2f}%", flush=True)

    # Clean up model to free VRAM for next run
    del model
    del ckpt
    del sd
    torch.cuda.empty_cache()

    return {
        "room_ious": room_ious.tolist(),
        "struct_ious": struct_ious.tolist(),
        "room_mIoU": round(room_mIoU, 2),
        "struct_mIoU": round(struct_mIoU, 2),
        "room_pixel_acc": round(pixel_acc, 2),
        "wall_iou": round(wall_iou, 2),
        "door_iou": round(door_iou, 2),
        "window_iou": round(window_iou, 2),
        "elapsed_s": round(elapsed, 1)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    ds = CubiCasaDataset(
        root_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cubicasa5k/cubicasa5k"),
        split="val",
        img_size=512,
        augment=False,
        cache_dir=str(PROJECT_ROOT / "data/CubiCasa5k/cached_512")
    )
    # Batch size 4 ensures zero VRAM spilling (1.4 GB / 4.0 GB)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    models_to_test = [
        ("Champion v1.1", PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"),
        ("Blend a=0.3", PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_3.pth"),
        ("Blend a=0.5", PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_5.pth"),
        ("Blend a=0.7", PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_blend_alpha_0_7.pth"),
        ("Challenger", PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth")
    ]

    results = {}
    for label, path in models_to_test:
        if path.exists():
            results[label] = evaluate_model(path, loader, device)
        else:
            print(f"[WARN] Checkpoint not found: {path}", flush=True)

    # Print Comparative Table
    print("\n" + "=" * 115, flush=True)
    print("PER-CLASS ROOM IoU & MACRO METRICS (CubiCasa5K Validation Split N=400)", flush=True)
    print("=" * 115, flush=True)
    header = f"{'Class':<20} | " + " | ".join([f"{l:<14}" for l in results.keys()])
    print(header, flush=True)
    print("-" * 115, flush=True)

    for c, name in enumerate(ROOM_CLASSES):
        row_str = f"{name:<20} | "
        for label, res in results.items():
            val = res["room_ious"][c]
            row_str += f"{val:6.2f}%        | "
        print(row_str, flush=True)

    print("-" * 115, flush=True)
    for metric_key, metric_name in [
        ("room_mIoU", "Room mIoU (fg)"),
        ("room_pixel_acc", "Room Pixel Acc"),
        ("wall_iou", "Wall IoU"),
        ("door_iou", "Door IoU"),
        ("window_iou", "Window IoU")
    ]:
        row_str = f"{metric_name:<20} | "
        for label, res in results.items():
            val = res[metric_key]
            row_str += f"{val:6.2f}%        | "
        print(row_str, flush=True)
    print("=" * 115, flush=True)

    # Save to JSON
    out_file = PROJECT_ROOT / "outputs" / "per_class_room_iou_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Results saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
