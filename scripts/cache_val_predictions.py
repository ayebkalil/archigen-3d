"""
Pre-caches model predictions on all 400 validation floorplans to disk.
Saves struct logits/probabilities and room logits/probabilities to data/val_predictions_cache/.
Allows CPU-bound vectorizer development, instance-F1 tuning, and oracle studies
without competing for the 4 GB GPU while fine-tuning runs in background.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet

def cache_predictions():
    cache_out = Path("data/val_predictions_cache")
    cache_out.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREDICTION CACHE] Running on device: {device}")
    
    val_dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load("models/saved/floorplan_unetpp/multitask_resnet34_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    idx = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            outputs = model(imgs)
            
            s_probs = torch.softmax(outputs["struct"], dim=1).cpu().numpy().astype(np.float16)
            r_probs = torch.softmax(outputs["room"], dim=1).cpu().numpy().astype(np.float16)
            
            for b in range(imgs.size(0)):
                sample_id = batch["sample_id"][b]
                gt_struct = batch["struct_mask"][b].numpy().astype(np.uint8)
                gt_room = batch["room_mask"][b].numpy().astype(np.uint8)
                gt_scale = float(batch["gt_scale"][b])
                
                safe_name = sample_id.replace("/", "_").replace("\\", "_")
                file_path = cache_out / f"{safe_name}.npz"
                
                np.savez_compressed(
                    file_path,
                    sample_id=sample_id,
                    struct_probs=s_probs[b],
                    room_probs=r_probs[b],
                    gt_struct=gt_struct,
                    gt_room=gt_room,
                    gt_scale=gt_scale
                )
                idx += 1
            if idx % 100 == 0:
                print(f"  Cached {idx}/{len(val_dataset)} plans...")

    print(f"[SUCCESS] Cached all {idx} validation predictions in {cache_out}")

if __name__ == "__main__":
    cache_predictions()
