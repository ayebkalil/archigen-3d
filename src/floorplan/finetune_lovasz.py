"""
Surgical Lovász-Softmax + Capped Class-Weighted Fine-Tuning Engine for ArchiGen 3D.
Resumes from multitask_resnet34_finetuned_best.pth.

Key Innovations:
1. Lovász-Softmax Loss (Berman et al., CVPR 2018):
   Direct submodular optimization of the Jaccard index / IoU on both Structure and Room heads.
2. Capped Square-Root Inverse Frequency Weights:
   Capped at 3.2x max (Office=3.2, Garage=2.8, Storage=2.2, Bath/Entry/Closet=1.5, Living/Bed=1.0)
   to resolve class collapse without hurting dominant class precision.
3. Gentle Cosine Schedule (lr=5e-5 -> 1e-6) over 6 epochs.
4. Frozen Encoder BatchNorm to eliminate batch=4 noise.
5. Saves epoch checkpoints and maintains multitask_resnet34_lovasz_best.pth.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES
from src.floorplan.finetune import AugmentedCubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_val_split(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    s_inter = np.zeros(5, dtype=np.int64)
    s_union = np.zeros(5, dtype=np.int64)
    mw_inter, mw_union = 0, 0

    r_inter = np.zeros(12, dtype=np.int64)
    r_union = np.zeros(12, dtype=np.int64)
    r_gt_px = np.zeros(12, dtype=np.int64)
    r_correct, r_total = 0, 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].numpy()
            t_r = batch["room_mask"].numpy()

            outputs = model(imgs)
            p_s = torch.argmax(outputs["struct"], dim=1).cpu().numpy().astype(np.uint8)
            p_r = torch.argmax(outputs["room"], dim=1).cpu().numpy().astype(np.uint8)

            B = imgs.size(0)
            for b in range(B):
                p_mw = ((p_s[b] == 1) | (p_s[b] == 2))
                t_mw = ((t_s[b] == 1) | (t_s[b] == 2))
                mw_inter += int(np.logical_and(p_mw, t_mw).sum())
                mw_union += int(np.logical_or(p_mw, t_mw).sum())

                for c in range(1, 5):
                    s_inter[c] += int(np.logical_and(p_s[b] == c, t_s[b] == c).sum())
                    s_union[c] += int(np.logical_or(p_s[b] == c, t_s[b] == c).sum())

                non_bg = (t_r[b] > 0)
                r_correct += int(np.logical_and(p_r[b] == t_r[b], non_bg).sum())
                r_total += int(non_bg.sum())

                for c in range(1, 11):
                    r_inter[c] += int(np.logical_and(p_r[b] == c, t_r[b] == c).sum())
                    r_union[c] += int(np.logical_or(p_r[b] == c, t_r[b] == c).sum())
                    r_gt_px[c] += int((t_r[b] == c).sum())

    wall_merged = (mw_inter / max(mw_union, 1)) * 100
    door = (s_inter[3] / max(s_union[3], 1)) * 100
    win = (s_inter[4] / max(s_union[4], 1)) * 100
    r_acc = (r_correct / max(r_total, 1)) * 100

    per_class_r = {c: ((r_inter[c] / max(r_union[c], 1)) * 100) for c in range(1, 11)}
    r_miou_10 = np.mean(list(per_class_r.values()))

    inv_rooms = {v: k for k, v in ROOM_CLASSES.items()}

    return {
        "merged_wall_iou": wall_merged,
        "door_iou": door,
        "win_iou": win,
        "room_acc": r_acc,
        "room_miou_10": r_miou_10,
        "per_class_room": {inv_rooms[c]: per_class_r[c] for c in range(1, 11)}
    }

def run_lovasz_finetune(
    epochs: int = 6,
    batch_size: int = 4,
    lr: float = 5e-5,
    resume_path: str = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"[SURGICAL LOVÁSZ FINE-TUNE] Starting {epochs} Epochs on {device}")
    print(f"[*] Resuming from: {resume_path}")
    print(f"[*] Base LR: {lr:.2e} | Cosine Annealing (T_max={epochs})")
    print("=" * 80)

    models_dir = PROJECT_ROOT / "models" / "saved" / "floorplan_unetpp"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "multitask_resnet34_lovasz_best.pth"

    # 1. Dataset & Loaders
    train_dataset = AugmentedCubiCasaDataset(split="train", img_size=512, augment=True)
    val_dataset = CubiCasaDataset(split="val", img_size=512, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model Initialization & Load Pretrained Weights
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(PROJECT_ROOT / resume_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"[*] Successfully loaded fine-tuned checkpoint.")

    # Freeze Encoder BatchNorm
    for m in model.base.encoder.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    print(f"[*] Frozen encoder BatchNorm statistics.")

    # 3. Loss Functions: Lovász-Softmax + Capped Class Weights
    struct_weights = torch.tensor([0.5, 4.0, 5.0, 25.0, 10.0]).to(device)
    criterion_struct_ce = nn.CrossEntropyLoss(weight=struct_weights)
    criterion_struct_dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2, 3, 4])
    criterion_struct_lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)

    # Capped Square-Root Inverse Frequency Weights (capped at 3.2 max)
    # 0: BG (0.5), 1: Living (1.0), 2: Bed (1.0), 3: Kitchen (1.2), 4: Bath (1.5), 5: Entry (1.5),
    # 6: Outdoor (1.2), 7: Storage (2.2), 8: Closet (2.5), 9: Garage (2.8), 10: Office (3.2), 11: Other (1.0)
    room_weights = torch.tensor([0.5, 1.0, 1.0, 1.2, 1.5, 1.5, 1.2, 2.2, 2.5, 2.8, 3.2, 1.0]).to(device)
    criterion_room_ce = nn.CrossEntropyLoss(weight=room_weights)
    criterion_room_dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, 12)))
    criterion_room_lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)

    def composite_loss(outputs, batch):
        s_logits, r_logits = outputs["struct"], outputs["room"]
        s_t, r_t = batch["struct_mask"], batch["room_mask"]

        # Structural loss: CE + Dice + Lovász
        l_s = (
            criterion_struct_ce(s_logits, s_t) +
            1.0 * criterion_struct_dice(s_logits, s_t) +
            0.5 * criterion_struct_lovasz(s_logits, s_t)
        )

        # Room loss: Capped Weighted CE + Dice + Lovász
        l_r = (
            criterion_room_ce(r_logits, r_t) +
            0.5 * criterion_room_dice(r_logits, r_t) +
            0.8 * criterion_room_lovasz(r_logits, r_t)
        )

        total_loss = l_s + 0.8 * l_r
        return total_loss, l_s, l_r

    # 4. Optimizer & Cosine Annealing
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Initial Validation Benchmark
    print("\n[*] Measuring initial baseline performance on 400 validation plans...")
    init_val = evaluate_val_split(model, val_loader, device)
    best_score = (
        0.35 * init_val["merged_wall_iou"] +
        0.35 * init_val["room_miou_10"] +
        0.15 * init_val["door_iou"] +
        0.15 * init_val["win_iou"]
    )
    print(f"[*] Initial Baseline Validation Score: {best_score:.2f}%")
    print(f"    Wall: {init_val['merged_wall_iou']:.2f}% | Door: {init_val['door_iou']:.2f}% | Room mIoU: {init_val['room_miou_10']:.2f}%")
    print(f"    Office: {init_val['per_class_room'].get('Office', 0.0):.2f}% | Storage: {init_val['per_class_room'].get('Storage', 0.0):.2f}% | Garage: {init_val['per_class_room'].get('Garage', 0.0):.2f}%")

    # 5. MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("floorplan-multitask-lovasz")

    with mlflow.start_run(run_name=f"lovasz_classweighted_{epochs}ep") as run:
        print(f"[MLFLOW] Started Run ID: {run.info.run_id}")
        mlflow.log_params({
            "epochs": epochs,
            "base_lr": lr,
            "batch_size": batch_size,
            "loss": "Lovasz_Softmax_plus_Capped_ClassWeighted_CE_Dice",
            "resume_path": resume_path
        })

        for epoch in range(1, epochs + 1):
            model.train()
            # Keep BN in eval mode
            for m in model.base.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

            train_loss_accum = 0.0
            t0 = time.time()
            num_batches = len(train_loader)

            for b_idx, batch in enumerate(train_loader):
                imgs = batch["image"].to(device)
                s_t = batch["struct_mask"].to(device)
                r_t = batch["room_mask"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss, l_s, l_r = composite_loss(outputs, {"struct_mask": s_t, "room_mask": r_t})

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss_accum += loss.item()

                if (b_idx + 1) % 150 == 0:
                    curr_lr = optimizer.param_groups[0]["lr"]
                    print(f"  [Epoch {epoch:02d}/{epochs:02d} | Step {b_idx+1:04d}/{num_batches}] Loss: {loss.item():.4f} (Struct: {l_s.item():.3f}, Room: {l_r.item():.3f}) | LR: {curr_lr:.2e}")

            scheduler.step()
            train_time = time.time() - t0
            avg_train_loss = train_loss_accum / max(num_batches, 1)

            # Evaluate on Validation Split
            t_eval = time.time()
            val_metrics = evaluate_val_split(model, val_loader, device)
            eval_time = time.time() - t_eval

            curr_score = (
                0.35 * val_metrics["merged_wall_iou"] +
                0.35 * val_metrics["room_miou_10"] +
                0.15 * val_metrics["door_iou"] +
                0.15 * val_metrics["win_iou"]
            )

            # Save Epoch Checkpoint
            ep_ckpt = models_dir / f"lovasz_epoch_{epoch:02d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "score": curr_score
            }, ep_ckpt)

            is_best = curr_score > best_score
            if is_best:
                best_score = curr_score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "best_score": best_score,
                    "sha256": "pending"
                }, best_ckpt_path)
                status_str = f"*** NEW BEST CHECKPOINT SAVED ({best_score:.2f}%) ***"
            else:
                status_str = f"(Best: {best_score:.2f}%)"

            # Print Detailed Epoch Summary
            print("\n" + "-" * 80)
            print(f"[EPOCH {epoch:02d}/{epochs:02d}] Completed in {train_time/60:.1f}m (Eval: {eval_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | {status_str}")
            print(f"  Wall IoU:    {val_metrics['merged_wall_iou']:6.2f}% (Baseline: {init_val['merged_wall_iou']:6.2f}%)")
            print(f"  Door IoU:    {val_metrics['door_iou']:6.2f}% (Baseline: {init_val['door_iou']:6.2f}%)")
            print(f"  Win IoU:     {val_metrics['win_iou']:6.2f}% (Baseline: {init_val['win_iou']:6.2f}%)")
            print(f"  Room Acc:    {val_metrics['room_acc']:6.2f}% (Baseline: {init_val['room_acc']:6.2f}%)")
            print(f"  Room mIoU:   {val_metrics['room_miou_10']:6.2f}% (Baseline: {init_val['room_miou_10']:6.2f}%)")
            print("  Per-Class Room IoU:")
            for r_name in ["LivingRoom", "Bedroom", "Kitchen", "Bath", "Entry", "Storage", "Closet", "Garage", "Office"]:
                v_curr = val_metrics["per_class_room"].get(r_name, 0.0)
                v_init = init_val["per_class_room"].get(r_name, 0.0)
                diff = v_curr - v_init
                print(f"    - {r_name:<12s}: {v_curr:6.2f}% ({'+' if diff >= 0 else ''}{diff:.2f}%)")
            print("-" * 80 + "\n")

            # MLflow logging
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_wall_iou": val_metrics["merged_wall_iou"],
                "val_door_iou": val_metrics["door_iou"],
                "val_win_iou": val_metrics["win_iou"],
                "val_room_acc": val_metrics["room_acc"],
                "val_room_miou_10": val_metrics["room_miou_10"],
                "val_office_iou": val_metrics["per_class_room"].get("Office", 0.0),
                "val_storage_iou": val_metrics["per_class_room"].get("Storage", 0.0),
                "val_garage_iou": val_metrics["per_class_room"].get("Garage", 0.0),
                "composite_score": curr_score
            }, step=epoch)

    print("\n" + "=" * 80)
    print(f"[+] Surgical Lovász Fine-Tuning Complete! Best Model: {best_ckpt_path} (Score: {best_score:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    run_lovasz_finetune(epochs=6, batch_size=4, lr=5e-5)
