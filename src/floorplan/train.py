"""
Multi-Task Floorplan Segmentation Training Pipeline with SMP UNet++ & MLflow Tracking.
Jointly trains:
- Structural Head: Background, Wall_Internal, Wall_External, Door, Window (5 classes)
- Semantic Room Head: Background, LivingRoom, Bedroom, Kitchen, Bath, Entry, Outdoor, Storage, Closet, Garage, Office, Other (12 classes)
Includes Mixed Precision (AMP), Weighted Cross-Entropy + Dice Loss, and full Validation/Test evaluation.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
import mlflow

from src.floorplan.dataset import CubiCasaDataset, ROOM_CLASSES, STRUCTURE_CLASSES
from src.floorplan.models import MultiTaskFloorplanUNet
from src.utils.seed import seed_everything

def train_floorplan_multitask(
    encoder_name: str = "resnet34",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 3e-4,
    img_size: int = 512,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    device_str: Optional[str] = None
):
    seed_everything(42)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[FLOORPLAN TRAIN] Training MultiTask UNet++ ({encoder_name}) on device: {device}")
    print(f"[CONFIG] Resolution: {img_size}x{img_size} | Batch Size: {batch_size} | Epochs: {epochs}")

    # Output directory
    models_dir = Path("models/saved/floorplan_unetpp")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Loaders
    train_dataset = CubiCasaDataset(split="train", img_size=img_size, augment=True)
    val_dataset = CubiCasaDataset(split="val", img_size=img_size, augment=False)
    test_dataset = CubiCasaDataset(split="test", img_size=img_size, augment=False)

    if max_train_samples and max_train_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(max_train_samples)))
    if max_val_samples and max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, list(range(max_val_samples)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} floorplans")

    # 2. Multi-Task Architecture
    model = MultiTaskFloorplanUNet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        num_struct_classes=5,
        num_room_classes=12,
        in_channels=3
    ).to(device)

    # 3. Loss Functions
    # Structure weights: Background=1.0, Wall_Int=4.0, Wall_Ext=5.0, Door=8.0, Window=6.0
    struct_weights = torch.tensor([1.0, 4.0, 5.0, 8.0, 6.0]).to(device)
    criterion_struct_ce = nn.CrossEntropyLoss(weight=struct_weights)
    criterion_struct_dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2, 3, 4])

    # Room weights: Background=0.5, specific rooms=1.5
    room_weights = torch.ones(12).to(device) * 1.5
    room_weights[0] = 0.5
    criterion_room_ce = nn.CrossEntropyLoss(weight=room_weights)
    criterion_room_dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, 12)))

    def multitask_loss(outputs, batch):
        s_logits, r_logits = outputs["struct"], outputs["room"]
        s_targets, r_targets = batch["struct_mask"], batch["room_mask"]

        loss_s = criterion_struct_ce(s_logits, s_targets) + 1.2 * criterion_struct_dice(s_logits, s_targets)
        loss_r = criterion_room_ce(r_logits, r_targets) + 1.0 * criterion_room_dice(r_logits, r_targets)
        return loss_s + 0.8 * loss_r, loss_s, loss_r

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 5. MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("floorplan-multitask-unetpp")

    best_score = 0.0
    best_ckpt_path = models_dir / f"multitask_{encoder_name}_best.pth"

    with mlflow.start_run(run_name=f"multitask_unetpp_{encoder_name}_{epochs}ep") as run:
        print(f"[MLFLOW] Started Run ID: {run.info.run_id}")
        mlflow.log_params({
            "model": "MultiTaskFloorplanUNet",
            "backbone": encoder_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": img_size,
            "struct_classes": 5,
            "room_classes": 12,
            "device": str(device)
        })

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_accum = 0.0
            train_s_loss_accum = 0.0
            train_r_loss_accum = 0.0

            for batch in train_loader:
                imgs = batch["image"].to(device)
                batch["struct_mask"] = batch["struct_mask"].to(device)
                batch["room_mask"] = batch["room_mask"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss, loss_s, loss_r = multitask_loss(outputs, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_accum += loss.item()
                train_s_loss_accum += loss_s.item()
                train_r_loss_accum += loss_r.item()

            scheduler.step()
            n_batches = len(train_loader)
            avg_train_loss = train_loss_accum / n_batches

            # Validation Loop
            val_metrics = evaluate_loader(model, val_loader, device, multitask_loss)

            print(
                f"[EPOCH {epoch:02d}/{epochs}] "
                f"Train: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Merged Wall IoU: {val_metrics['merged_wall_iou']*100:.2f}% | "
                f"Struct mIoU: {val_metrics['struct_miou']*100:.2f}% "
                f"(Wall_Ext: {val_metrics['wall_ext_iou']*100:.1f}%, Door: {val_metrics['door_iou']*100:.1f}%, Win: {val_metrics['win_iou']*100:.1f}%) | "
                f"Room mIoU: {val_metrics['room_miou']*100:.2f}%"
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_struct_loss": train_s_loss_accum / n_batches,
                "train_room_loss": train_r_loss_accum / n_batches,
                "val_loss": val_metrics["val_loss"],
                "val_merged_wall_iou": val_metrics["merged_wall_iou"],
                "val_struct_miou": val_metrics["struct_miou"],
                "val_wall_ext_iou": val_metrics["wall_ext_iou"],
                "val_wall_int_iou": val_metrics["wall_int_iou"],
                "val_door_iou": val_metrics["door_iou"],
                "val_win_iou": val_metrics["win_iou"],
                "val_room_miou": val_metrics["room_miou"],
                "val_room_acc": val_metrics["room_acc"],
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)

            # Combined score for checkpoint selection
            combined_score = 0.6 * val_metrics["merged_wall_iou"] + 0.4 * val_metrics["room_miou"]
            if combined_score > best_score:
                best_score = combined_score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "encoder_name": encoder_name,
                    "img_size": img_size
                }, best_ckpt_path)
                print(f"  -> [CHECKPOINT] Saved best model to {best_ckpt_path} (Combined Score: {best_score*100:.2f}%)")

        # 6. Final Evaluation on Test Split using Best Checkpoint
        print(f"\n[EVALUATION] Loading best checkpoint from {best_ckpt_path} for Test Split...")
        best_data = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_data["model_state_dict"])
        test_metrics = evaluate_loader(model, test_loader, device, multitask_loss)

        print("\n========== FINAL TEST SET BENCHMARK ==========")
        print(f"Test Merged Wall IoU: {test_metrics['merged_wall_iou']*100:.2f}%")
        print(f"Test Struct mIoU:     {test_metrics['struct_miou']*100:.2f}%")
        print(f"  - Wall Internal IoU: {test_metrics['wall_int_iou']*100:.2f}%")
        print(f"  - Wall External IoU: {test_metrics['wall_ext_iou']*100:.2f}%")
        print(f"  - Door IoU:          {test_metrics['door_iou']*100:.2f}%")
        print(f"  - Window IoU:        {test_metrics['win_iou']*100:.2f}%")
        print(f"Test Room mIoU:       {test_metrics['room_miou']*100:.2f}%")
        print(f"Test Room Accuracy:   {test_metrics['room_acc']*100:.2f}%")
        print("==============================================\n")

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

    return best_ckpt_path

def evaluate_loader(model, loader, device, loss_fn) -> Dict[str, float]:
    model.eval()
    val_loss_accum = 0.0

    # Accumulators for structure classes [1: Wall_Int, 2: Wall_Ext, 3: Door, 4: Window]
    s_inter = {c: 0 for c in range(1, 5)}
    s_union = {c: 0 for c in range(1, 5)}
    merged_wall_inter = 0
    merged_wall_union = 0

    # Accumulators for room classes [1..11]
    r_inter = {c: 0 for c in range(1, 12)}
    r_union = {c: 0 for c in range(1, 12)}
    r_correct_pixels = 0
    r_total_pixels = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            batch["struct_mask"] = batch["struct_mask"].to(device)
            batch["room_mask"] = batch["room_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss, _, _ = loss_fn(outputs, batch)

            val_loss_accum += loss.item()

            s_preds = torch.argmax(outputs["struct"], dim=1)
            r_preds = torch.argmax(outputs["room"], dim=1)

            s_gt = batch["struct_mask"]
            r_gt = batch["room_mask"]

            # Structure IoUs
            for c in range(1, 5):
                p_c = (s_preds == c)
                t_c = (s_gt == c)
                s_inter[c] += (p_c & t_c).sum().item()
                s_union[c] += (p_c | t_c).sum().item()

            # Merged Wall IoU (Internal + External combined)
            p_m_wall = (s_preds == 1) | (s_preds == 2)
            t_m_wall = (s_gt == 1) | (s_gt == 2)
            merged_wall_inter += (p_m_wall & t_m_wall).sum().item()
            merged_wall_union += (p_m_wall | t_m_wall).sum().item()

            # Room IoUs
            for c in range(1, 12):
                p_c = (r_preds == c)
                t_c = (r_gt == c)
                r_inter[c] += (p_c & t_c).sum().item()
                r_union[c] += (p_c | t_c).sum().item()

            # Room Accuracy on non-background pixels
            non_bg = (r_gt > 0)
            r_correct_pixels += ((r_preds == r_gt) & non_bg).sum().item()
            r_total_pixels += non_bg.sum().item()

    n = max(len(loader), 1)
    wall_int_iou = s_inter[1] / max(s_union[1], 1)
    wall_ext_iou = s_inter[2] / max(s_union[2], 1)
    door_iou = s_inter[3] / max(s_union[3], 1)
    win_iou = s_inter[4] / max(s_union[4], 1)
    struct_miou = (wall_int_iou + wall_ext_iou + door_iou + win_iou) / 4.0
    merged_wall_iou = merged_wall_inter / max(merged_wall_union, 1)

    room_ious = [r_inter[c] / max(r_union[c], 1) for c in range(1, 12) if r_union[c] > 0]
    room_miou = sum(room_ious) / max(len(room_ious), 1) if room_ious else 0.0
    room_acc = r_correct_pixels / max(r_total_pixels, 1)

    return {
        "val_loss": val_loss_accum / n,
        "struct_miou": struct_miou,
        "merged_wall_iou": merged_wall_iou,
        "wall_int_iou": wall_int_iou,
        "wall_ext_iou": wall_ext_iou,
        "door_iou": door_iou,
        "win_iou": win_iou,
        "room_miou": room_miou,
        "room_acc": room_acc
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Floorplan UNet++ Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default 4 for 4GB RTX 3050)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="Image resolution (512 or 768)")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for fast benchmark")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples")
    args = parser.parse_args()

    train_floorplan_multitask(
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
