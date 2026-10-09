"""
Self-contained Overfit Sanity Check Script.
Trains MultiTaskFloorplanUNet on 8 fixed floorplans WITHOUT augmentation,
and evaluates training metrics at each epoch directly on those 8 plans.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.utils.seed import seed_everything

def run_overfit_test(num_samples: int = 8, epochs: int = 50, lr: float = 1e-3):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OVERFIT TEST] Device: {device} | Samples: {num_samples} | Epochs: {epochs} | LR: {lr}")

    dataset = CubiCasaDataset(split="train", img_size=512, augment=False)
    subset = Subset(dataset, list(range(num_samples)))
    loader = DataLoader(subset, batch_size=4, shuffle=True)
    eval_loader = DataLoader(subset, batch_size=4, shuffle=False)

    model = MultiTaskFloorplanUNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        num_struct_classes=5,
        num_room_classes=12
    ).to(device)

    # Weights
    struct_weights = torch.tensor([1.0, 4.0, 5.0, 10.0, 8.0]).to(device)
    criterion_struct_ce = nn.CrossEntropyLoss(weight=struct_weights)
    criterion_struct_dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2, 3, 4])

    room_weights = torch.ones(12).to(device) * 1.5
    room_weights[0] = 0.5
    criterion_room_ce = nn.CrossEntropyLoss(weight=room_weights)
    criterion_room_dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(1, 12)))

    def loss_fn(outputs, batch):
        s_logits, r_logits = outputs["struct"], outputs["room"]
        s_targets, r_targets = batch["struct_mask"], batch["room_mask"]
        loss_s = criterion_struct_ce(s_logits, s_targets) + 1.2 * criterion_struct_dice(s_logits, s_targets)
        loss_r = criterion_room_ce(r_logits, r_targets) + 1.0 * criterion_room_dice(r_logits, r_targets)
        return loss_s + 0.8 * loss_r

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)
            batch["struct_mask"] = batch["struct_mask"].to(device)
            batch["room_mask"] = batch["room_mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = loss_fn(outputs, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_loss = train_loss / len(loader)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            merged_inter = 0
            merged_union = 0
            door_inter = 0
            door_union = 0
            win_inter = 0
            win_union = 0
            room_inter = {c: 0 for c in range(1, 12)}
            room_union = {c: 0 for c in range(1, 12)}

            with torch.no_grad():
                for batch in eval_loader:
                    imgs = batch["image"].to(device)
                    s_gt = batch["struct_mask"].to(device)
                    r_gt = batch["room_mask"].to(device)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        outputs = model(imgs)

                    s_pred = torch.argmax(outputs["struct"], dim=1)
                    r_pred = torch.argmax(outputs["room"], dim=1)

                    # Merged wall (1: int, 2: ext)
                    p_m_wall = (s_pred == 1) | (s_pred == 2)
                    t_m_wall = (s_gt == 1) | (s_gt == 2)
                    merged_inter += (p_m_wall & t_m_wall).sum().item()
                    merged_union += (p_m_wall | t_m_wall).sum().item()

                    # Door (3)
                    p_door = (s_pred == 3)
                    t_door = (s_gt == 3)
                    door_inter += (p_door & t_door).sum().item()
                    door_union += (p_door | t_door).sum().item()

                    # Window (4)
                    p_win = (s_pred == 4)
                    t_win = (s_gt == 4)
                    win_inter += (p_win & t_win).sum().item()
                    win_union += (p_win | t_win).sum().item()

                    # Rooms
                    for c in range(1, 12):
                        p_c = (r_pred == c)
                        t_c = (r_gt == c)
                        room_inter[c] += (p_c & t_c).sum().item()
                        room_union[c] += (p_c | t_c).sum().item()

            merged_wall_iou = merged_inter / max(merged_union, 1)
            door_iou = door_inter / max(door_union, 1)
            win_iou = win_inter / max(win_union, 1)
            r_ious = [room_inter[c] / max(room_union[c], 1) for c in range(1, 12) if room_union[c] > 0]
            room_miou = sum(r_ious) / max(len(r_ious), 1) if r_ious else 0.0

            print(
                f"[EPOCH {epoch:02d}/{epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Merged Wall IoU: {merged_wall_iou*100:.2f}% | "
                f"Door IoU: {door_iou*100:.2f}% | "
                f"Win IoU: {win_iou*100:.2f}% | "
                f"Room mIoU: {room_miou*100:.2f}%"
            )

    print("[OVERFIT TEST COMPLETED]")

if __name__ == "__main__":
    run_overfit_test()
