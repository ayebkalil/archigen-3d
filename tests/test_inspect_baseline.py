import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from safetensors.torch import load_file
import segmentation_models_pytorch as smp
from src.floorplan.dataset import CubiCasaDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name='resnet34', classes=4).to(device)
state_dict = load_file('models/saved/floorplan_baseline/best.safetensors')
model.load_state_dict(state_dict)
model.eval()

ds = CubiCasaDataset(split='val', img_size=512, augment=False)
for idx in [0, 1, 2, 3, 4]:
    sample = ds[idx]
    img_tensor = sample['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu().numpy()
    print(f"Sample {idx} pred classes: {np.unique(preds, return_counts=True)}")
    print(f"Sample {idx} GT classes:   {np.unique(sample['mask'].numpy(), return_counts=True)}")
