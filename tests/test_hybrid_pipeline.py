import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import segmentation_models_pytorch as smp

from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer
from src.bridge.facade_bridge import FacadeBridge

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Baseline Structure Model
    base_model = smp.Unet(encoder_name="resnet34", classes=4).to(device)
    base_model.load_state_dict(load_file("models/saved/floorplan_baseline/best.safetensors"))
    base_model.eval()

    # 2. Load MultiTask Room Model
    mt_model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    mt_ckpt = torch.load("models/saved/floorplan_unetpp/multitask_resnet34_best.pth", map_location=device)
    mt_model.load_state_dict(mt_ckpt["model_state_dict"])
    mt_model.eval()

    # Test on a real floorplan
    p = list(Path("data/CubiCasa5k").rglob("F1_scaled.png"))[1]
    raw_img = Image.open(p).convert("RGB")
    iw, ih = raw_img.size
    size = 512
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2

    resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    tensor = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        struct_preds = torch.argmax(base_model(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        room_preds = torch.argmax(mt_model(tensor)["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    print("Baseline Struct Classes:", np.unique(struct_preds, return_counts=True))
    print("MultiTask Room Classes:", np.unique(room_preds, return_counts=True))

    vectorizer = FloorplanVectorizer()
    layout = vectorizer.vectorize(mask=struct_preds, room_mask=room_preds, scale_override_m_per_px=0.03)

    walls = layout.get("walls", [])
    rooms = layout.get("rooms", [])
    print(f"HYBRID INFERENCE: {len(walls)} walls, {len(rooms)} semantic rooms")
    for r in rooms:
        print(f"  -> Room: {r['room_type']} ({r['surface_m2']} m2)")
    ext = [w for w in walls if w["is_exterior"]]
    print(f"Exterior walls identified: {len(ext)} / {len(walls)}")

if __name__ == "__main__":
    run_test()
