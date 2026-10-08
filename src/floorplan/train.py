"""
Floorplan Segmentation Training Pipeline with SMP UNet++ & MLflow Tracking.
Features:
- UNet++ Architecture with Pretrained Backbone (ImageNet).
- Combined Cross-Entropy + Dice Loss with Class Imbalance Weighting.
- Mixed Precision (AMP) on RTX 3050 GPU.
- Tracks mIoU, Wall IoU, Door IoU, Window IoU in MLflow SQLite.
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import mlflow

from src.floorplan.dataset import CubiCasaDataset
from src.utils.seed import seed_everything

def train_floorplan_model(
    encoder_name: str = "resnet34",
    num_classes: int = 4,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    img_size: int = 512,
    device: Optional[str] = None
):
    seed_everything(42)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[FLOORPLAN TRAIN] Training UNet++ ({encoder_name}) on device: {device}")

    # Output directories
    models_dir = Path("models/saved/floorplan_unetpp")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Loaders
    train_dataset = CubiCasaDataset(split="train", img_size=img_size, augment=True)
    val_dataset = CubiCasaDataset(split="val", img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)} floorplans")

    # 2. Model: UNet++ with Pretrained ImageNet Encoder
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    ).to(device)

    # 3. Loss Functions with Class Weighting (handling thin wall/door imbalance)
    # Background: 1.0, Wall: 4.0, Door: 8.0, Window: 6.0
    class_weights = torch.tensor([1.0, 4.0, 8.0, 6.0]).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2, 3])

    def composite_loss(logits, targets):
        return criterion_ce(logits, targets) + 1.5 * criterion_dice(logits, targets)

    # 4. Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 5. MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("floorplan-cubicasa5k")

    with mlflow.start_run(run_name=f"unetpp_{encoder_name}_{epochs}ep") as run:
        print(f"[MLFLOW] Started Run ID: {run.info.run_id}")
        mlflow.log_params({
            "model": "UnetPlusPlus",
            "encoder": encoder_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": img_size,
            "classes": num_classes
        })

        best_miou = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = composite_loss(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)

            # Validation Loop
            model.eval()
            total_wall_inter, total_wall_union = 0, 0
            total_door_inter, total_door_union = 0, 0
            total_win_inter, total_win_union = 0, 0
            val_loss = 0.0

            with torch.no_grad():
                for v_batch in val_loader:
                    v_imgs = v_batch["image"].to(device)
                    v_masks = v_batch["mask"].to(device)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        v_logits = model(v_imgs)
                        v_loss = composite_loss(v_logits, v_masks)

                    val_loss += v_loss.item()
                    v_preds = torch.argmax(v_logits, dim=1)

                    # Accumulate IoUs
                    for c in [1, 2, 3]:
                        p_c = (v_preds == c)
                        t_c = (v_masks == c)
                        inter = (p_c & t_c).sum().item()
                        union = (p_c | t_c).sum().item()
                        if c == 1:
                            total_wall_inter += inter; total_wall_union += union
                        elif c == 2:
                            total_door_inter += inter; total_door_union += union
                        elif c == 3:
                            total_win_inter += inter; total_win_union += union

            wall_iou = total_wall_inter / max(total_wall_union, 1)
            door_iou = total_door_inter / max(total_door_union, 1)
            win_iou = total_win_inter / max(total_win_union, 1)
            miou = (wall_iou + door_iou + win_iou) / 3.0
            avg_val_loss = val_loss / len(val_loader)

            print(f"[EPOCH {epoch:02d}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Wall IoU: {wall_iou*100:.2f}% | Door IoU: {door_iou*100:.2f}% | Win IoU: {win_iou*100:.2f}% | mIoU: {miou*100:.2f}%")

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "wall_iou": wall_iou,
                "door_iou": door_iou,
                "win_iou": win_iou,
                "miou": miou
            }, step=epoch)

            if miou > best_miou:
                best_miou = miou
                best_ckpt = models_dir / f"unetpp_{encoder_name}_best.pth"
                torch.save(model.state_dict(), best_ckpt)
                print(f"  -> [CHECKPOINT] Saved best model with mIoU: {best_miou*100:.2f}%")

        print("[SUCCESS] Floorplan training completed successfully!")

if __name__ == "__main__":
    train_floorplan_model(epochs=30)
