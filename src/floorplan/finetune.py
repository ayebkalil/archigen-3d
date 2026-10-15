"""
Fine-Tuning Pipeline for MultiTask Floorplan UNet++ with CAD Image Augmentations.
Resumes from multitask_resnet34_best.pth for 12 epochs with:
1. Geometry-preserving CAD augmentations (p=0.5): grayscale, inversion, blur, JPEG noise, scale jitter, contrast.
2. Frozen Encoder BatchNorm layers to eliminate small-batch statistical noise.
3. Lower learning rate (1.5e-4 -> 1e-6) with Cosine Annealing.
4. Window IoU incorporated into checkpoint selection score:
   Score = 0.40 * Wall + 0.30 * Room + 0.15 * Door + 0.15 * Window.
5. Saves every epoch checkpoint and tracks to MLflow experiment 'floorplan-multitask-finetune'.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import mlflow

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.utils.seed import seed_everything
from scripts.verify_augmentations import CADImageAugmentor

class AugmentedCubiCasaDataset(CubiCasaDataset):
    """Wraps CubiCasaDataset with CADImageAugmentor for train split."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cad_augmentor = CADImageAugmentor(p_total=0.50)

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.augment:
            # Un-normalize tensor to PIL, augment, and re-normalize
            img_t = item["image"]
            raw_t = (img_t * self.std + self.mean).clamp(0, 1)
            raw_np = (raw_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            from PIL import Image
            pil_img = Image.fromarray(raw_np)
            aug_pil = self.cad_augmentor(pil_img)
            import torchvision.transforms.functional as TF
            new_t = TF.to_tensor(aug_pil)
            item["image"] = (new_t - self.mean) / self.std
        return item

def evaluate_val_split(model, loader, device, loss_fn):
    model.eval()
    val_loss_accum = 0.0

    s_inter = {c: 0 for c in range(1, 5)}
    s_union = {c: 0 for c in range(1, 5)}
    mw_inter, mw_union = 0, 0

    r_inter = {c: 0 for c in range(1, 12)}
    r_union = {c: 0 for c in range(1, 12)}
    r_gt_px = {c: 0 for c in range(1, 12)}
    r_correct_px, r_total_px = 0, 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            t_s = batch["struct_mask"].to(device)
            t_r = batch["room_mask"].to(device)

            outputs = model(imgs)
            loss, _, _ = loss_fn(outputs, {"struct_mask": t_s, "room_mask": t_r})
            val_loss_accum += loss.item()

            p_s = torch.argmax(outputs["struct"], dim=1).cpu().numpy()
            p_r = torch.argmax(outputs["room"], dim=1).cpu().numpy()
            t_s_np = t_s.cpu().numpy()
            t_r_np = t_r.cpu().numpy()

            for b in range(imgs.size(0)):
                # Walls
                p_wall = (p_s[b] == 1) | (p_s[b] == 2)
                t_wall = (t_s_np[b] == 1) | (t_s_np[b] == 2)
                mw_inter += np.logical_and(p_wall, t_wall).sum()
                mw_union += np.logical_or(p_wall, t_wall).sum()

                for c in range(1, 5):
                    s_inter[c] += np.logical_and(p_s[b] == c, t_s_np[b] == c).sum()
                    s_union[c] += np.logical_or(p_s[b] == c, t_s_np[b] == c).sum()

                # Rooms
                non_bg = (t_r_np[b] > 0)
                r_correct_px += np.logical_and(p_r[b] == t_r_np[b], non_bg).sum()
                r_total_px += non_bg.sum()

                for c in range(1, 12):
                    r_inter[c] += np.logical_and(p_r[b] == c, t_r_np[b] == c).sum()
                    r_union[c] += np.logical_or(p_r[b] == c, t_r_np[b] == c).sum()
                    r_gt_px[c] += (t_r_np[b] == c).sum()

    # 10-Class Room mIoU (excluding class 11 'Other' which has 0 GT pixels)
    room_ious = [r_inter[c] / max(r_union[c], 1) for c in range(1, 11)]
    macro_room_miou = np.mean(room_ious)

    wall_merged = mw_inter / max(mw_union, 1)
    wall_int = s_inter[1] / max(s_union[1], 1)
    wall_ext = s_inter[2] / max(s_union[2], 1)
    door = s_inter[3] / max(s_union[3], 1)
    win = s_inter[4] / max(s_union[4], 1)
    room_acc = r_correct_px / max(r_total_px, 1)

    return {
        "val_loss": val_loss_accum / max(len(loader), 1),
        "merged_wall_iou": wall_merged,
        "wall_int_iou": wall_int,
        "wall_ext_iou": wall_ext,
        "door_iou": door,
        "win_iou": win,
        "room_miou_10": macro_room_miou,
        "room_acc": room_acc
    }

def run_finetune(
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 1.5e-4,
    resume_path: str = "models/saved/floorplan_unetpp/multitask_resnet34_best_backup_25ep.pth"
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[FINE-TUNE] Starting {epochs}-epoch CAD-augmented fine-tune on {device}")
    print(f"[*] Resuming from: {resume_path}")

    models_dir = Path("models/saved/floorplan_unetpp")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "multitask_resnet34_finetuned_best.pth"

    # 1. Dataset & Loaders
    train_dataset = AugmentedCubiCasaDataset(split="train", img_size=512, augment=True)
    val_dataset = CubiCasaDataset(split="val", img_size=512, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model Initialization & Weight Loading
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[*] Successfully loaded base model checkpoint.")

    # Freeze Encoder BatchNorm layers to eliminate small-batch statistical noise
    for m in model.base.encoder.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    print(f"[*] Frozen encoder BatchNorm statistics to prevent batch=4 noise.")

    # 3. Loss Functions
    struct_weights = torch.tensor([0.5, 4.0, 5.0, 25.0, 10.0]).to(device)
    criterion_struct_ce = nn.CrossEntropyLoss(weight=struct_weights)
    criterion_struct_dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2, 3, 4])

    room_weights = torch.ones(12).to(device) * 1.5
    room_weights[0] = 0.5
    criterion_room_ce = nn.CrossEntropyLoss(weight=room_weights)
    criterion_room_dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, 12)))

    def multitask_loss(outputs, batch):
        s_logits, r_logits = outputs["struct"], outputs["room"]
        s_t, r_t = batch["struct_mask"], batch["room_mask"]
        l_s = criterion_struct_ce(s_logits, s_t) + 1.5 * criterion_struct_dice(s_logits, s_t)
        l_r = criterion_room_ce(r_logits, r_t) + 1.0 * criterion_room_dice(r_logits, r_t)
        return l_s + 0.8 * l_r, l_s, l_r

    # 4. Optimizer & Cosine Annealing
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Initial benchmark on validation before fine-tuning
    print("[*] Computing baseline validation score before fine-tuning...")
    initial_val = evaluate_val_split(model, val_loader, device, multitask_loss)
    best_score = (
        0.40 * initial_val["merged_wall_iou"] +
        0.30 * initial_val["room_miou_10"] +
        0.15 * initial_val["door_iou"] +
        0.15 * initial_val["win_iou"]
    )
    print(f"[*] Initial Baseline Score: {best_score*100:.2f}% (Wall: {initial_val['merged_wall_iou']*100:.1f}%, Room: {initial_val['room_miou_10']*100:.1f}%, Door: {initial_val['door_iou']*100:.1f}%, Win: {initial_val['win_iou']*100:.1f}%)")

    # 5. MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("floorplan-multitask-finetune")

    with mlflow.start_run(run_name=f"finetune_cad_aug_{epochs}ep") as run:
        print(f"[MLFLOW] Started Run ID: {run.info.run_id}")
        mlflow.log_params({
            "epochs": epochs,
            "lr_start": lr,
            "batch_size": batch_size,
            "augmentations": "grayscale,inversion,blur,jpeg,scale,contrast,1px_morph",
            "p_aug": 0.50,
            "frozen_encoder_bn": True
        })

        accum_steps = 2
        for ep in range(1, epochs + 1):
            model.train()
            # Keep encoder BN frozen in eval mode even during train()
            for m in model.base.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

            train_loss_accum = 0.0
            optimizer.zero_grad()

            for b_idx, batch in enumerate(train_loader):
                imgs = batch["image"].to(device)
                batch["struct_mask"] = batch["struct_mask"].to(device)
                batch["room_mask"] = batch["room_mask"].to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss, _, _ = multitask_loss(outputs, batch)
                    loss_scaled = loss / accum_steps

                scaler.scale(loss_scaled).backward()
                if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss_accum += loss.item()
                if (b_idx + 1) % 250 == 0:
                    print(f"  [Ep {ep:02d}/{epochs} | Batch {b_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)

            scheduler.step()
            avg_train_loss = train_loss_accum / len(train_loader)

            # Evaluate on Validation Split
            val_res = evaluate_val_split(model, val_loader, device, multitask_loss)
            current_score = (
                0.40 * val_res["merged_wall_iou"] +
                0.30 * val_res["room_miou_10"] +
                0.15 * val_res["door_iou"] +
                0.15 * val_res["win_iou"]
            )

            print(
                f"[EPOCH {ep:02d}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_res['val_loss']:.4f} | "
                f"Wall: {val_res['merged_wall_iou']*100:.2f}% | "
                f"Room mIoU(10): {val_res['room_miou_10']*100:.2f}% | "
                f"Door: {val_res['door_iou']*100:.2f}% | "
                f"Win: {val_res['win_iou']*100:.2f}% | "
                f"Score: {current_score*100:.2f}% (Best: {best_score*100:.2f}%)",
                flush=True
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_res["val_loss"],
                "val_wall_merged": val_res["merged_wall_iou"],
                "val_wall_int": val_res["wall_int_iou"],
                "val_wall_ext": val_res["wall_ext_iou"],
                "val_door": val_res["door_iou"],
                "val_win": val_res["win_iou"],
                "val_room_miou_10": val_res["room_miou_10"],
                "val_room_acc": val_res["room_acc"],
                "composite_score": current_score,
                "lr": scheduler.get_last_lr()[0]
            }, step=ep)

            # Save checkpoint every epoch for safety
            ep_ckpt = models_dir / f"finetune_epoch_{ep:02d}.pth"
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_res,
                "composite_score": current_score
            }, ep_ckpt)

            if current_score > best_score:
                best_score = current_score
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_res,
                    "composite_score": best_score
                }, best_ckpt_path)
                print(f"  -> [CHECKPOINT] Saved new best model to {best_ckpt_path}")

    print(f"\n[FINE-TUNE COMPLETE] Best checkpoint saved to {best_ckpt_path} (Score: {best_score*100:.2f}%)")

if __name__ == "__main__":
    run_finetune()
