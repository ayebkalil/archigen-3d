"""
Fine-Tuning Script with Procedural Restyling Domain Adaptation.
Trains MultiTask ResNet34 UNet++ on a 50/50 blend of original CubiCasa and
procedural vector-restyled plans (CAD blueprints, B&W drawings, hatched walls,
and dimension annotations).

Saves exclusively to:
  models/saved/floorplan_unetpp/multitask_resnet34_restyled_challenger.pth
Leaving the v1.1 production champion strictly frozen.
"""

import sys
import os
import time
import json
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

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.restyled_dataset import RestyledCubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.utils.seed import seed_everything

def evaluate_val_split(model, loader, device, loss_fn):
    model.eval()
    val_loss_accum = 0.0

    s_inter = {c: 0 for c in range(1, 5)}
    s_union = {c: 0 for c in range(1, 5)}
    mw_inter, mw_union = 0, 0

    r_inter = {c: 0 for c in range(1, 12)}
    r_union = {c: 0 for c in range(1, 12)}
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
                # Merged walls (internal + external)
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

    room_ious = [r_inter[c] / max(r_union[c], 1) for c in range(1, 11)]
    macro_room_miou = float(np.mean(room_ious))

    wall_merged = float(mw_inter / max(mw_union, 1))
    wall_int = float(s_inter[1] / max(s_union[1], 1))
    wall_ext = float(s_inter[2] / max(s_union[2], 1))
    door = float(s_inter[3] / max(s_union[3], 1))
    win = float(s_inter[4] / max(s_union[4], 1))
    room_acc = float(r_correct_px / max(r_total_px, 1))

    return {
        "val_loss": float(val_loss_accum / max(len(loader), 1)),
        "merged_wall_iou": wall_merged,
        "wall_int_iou": wall_int,
        "wall_ext_iou": wall_ext,
        "door_iou": door,
        "win_iou": win,
        "room_miou_10": macro_room_miou,
        "room_acc": room_acc
    }

def train_restyled_challenger(
    epochs: int = 6,
    batch_size: int = 4,
    lr: float = 5e-5,
    p_restyle: float = 0.50,
    resume_path: str = "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth",
    challenger_name: str = "multitask_resnet34_restyled_challenger.pth"
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"[DOMAIN ADAPTATION] Training Restyled Challenger ({epochs} epochs, lr={lr}, p_restyle={p_restyle})")
    print(f"Device: {device}")
    print(f"Base Frozen Champion: {resume_path}")
    print(f"Challenger Target: models/saved/floorplan_unetpp/{challenger_name}")
    print("=" * 90)

    out_dir = Path("models/saved/floorplan_unetpp")
    out_dir.mkdir(parents=True, exist_ok=True)
    challenger_path = out_dir / challenger_name

    # 1. Datasets & Loaders
    train_dataset = RestyledCubiCasaDataset(split="train", img_size=512, augment=True, p_restyle=p_restyle)
    val_dataset = CubiCasaDataset(split="val", img_size=512, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    # 2. Model Initialization
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    print(f"[*] Successfully loaded base weights from {resume_path}")

    # Freeze Encoder BatchNorm
    for m in model.base.encoder.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    print("[*] Frozen encoder BatchNorm statistics.")

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

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Initial Baseline Score on Val
    print("\n[*] Evaluating baseline on validation split before fine-tuning...")
    initial_val = evaluate_val_split(model, val_loader, device, multitask_loss)
    best_score = (
        0.40 * initial_val["merged_wall_iou"] +
        0.30 * initial_val["room_miou_10"] +
        0.15 * initial_val["door_iou"] +
        0.15 * initial_val["win_iou"]
    )
    print(f"[*] Initial Baseline Score: {best_score*100:.2f}% | Wall: {initial_val['merged_wall_iou']*100:.1f}% | Room: {initial_val['room_miou_10']*100:.1f}% | Door: {initial_val['door_iou']*100:.1f}% | Win: {initial_val['win_iou']*100:.1f}%", flush=True)

    training_history = {
        "initial_val": initial_val,
        "epochs": []
    }

    best_challenger_score = 0.0

    accum_steps = 2
    total_start_time = time.time()

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        for m in model.base.encoder.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

        train_loss_accum = 0.0
        optimizer.zero_grad()

        for b_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            batch["struct_mask"] = batch["struct_mask"].to(device, non_blocking=True)
            batch["room_mask"] = batch["room_mask"].to(device, non_blocking=True)

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
            if (b_idx + 1) % 200 == 0:
                print(f"  [Ep {ep:02d}/{epochs:02d} | Step {b_idx+1:04d}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        ep_duration = time.time() - ep_start
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        val_res = evaluate_val_split(model, val_loader, device, multitask_loss)
        current_score = (
            0.40 * val_res["merged_wall_iou"] +
            0.30 * val_res["room_miou_10"] +
            0.15 * val_res["door_iou"] +
            0.15 * val_res["win_iou"]
        )

        ep_record = {
            "epoch": ep,
            "train_loss": avg_train_loss,
            "val_loss": val_res["val_loss"],
            "wall_iou": val_res["merged_wall_iou"],
            "room_miou": val_res["room_miou_10"],
            "door_iou": val_res["door_iou"],
            "win_iou": val_res["win_iou"],
            "composite_score": current_score,
            "duration_s": round(ep_duration, 1)
        }
        training_history["epochs"].append(ep_record)

        print(f"--- [Epoch {ep:02d}/{epochs:02d} Finished in {ep_duration:.1f}s] ---")
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f}")
        print(f"    Wall IoU: {val_res['merged_wall_iou']*100:.2f}% | Room mIoU: {val_res['room_miou_10']*100:.2f}% | Door IoU: {val_res['door_iou']*100:.2f}% | Win IoU: {val_res['win_iou']*100:.2f}%")
        print(f"    Composite Score: {current_score*100:.2f}% (Best so far: {best_score*100:.2f}%)", flush=True)

        # Save best challenger across training epochs
        if current_score > best_challenger_score:
            best_challenger_score = current_score
            print(f"    [+] New best challenger ({current_score*100:.2f}%). Saving to {challenger_path}...", flush=True)
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "composite_score": current_score,
                "metrics": val_res
            }, challenger_path)

    total_time = time.time() - total_start_time
    print(f"\n[DONE] Restyled Training Completed in {total_time/60:.2f} min.")
    print(f"Saved Challenger: {challenger_path}")

    metrics_out = Path("outputs/restyled_training_metrics.json")
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    print(f"Training metrics saved to {metrics_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--p-restyle", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train_restyled_challenger(
        epochs=args.epochs,
        lr=args.lr,
        p_restyle=args.p_restyle,
        batch_size=args.batch_size
    )
