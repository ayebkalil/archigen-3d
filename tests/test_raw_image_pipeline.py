import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import numpy as np

from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer
from src.bridge.facade_bridge import FacadeBridge

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "models/saved/floorplan_unetpp/multitask_resnet34_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_path = Path("data/CubiCasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/333/F1_scaled.png")
    if not img_path.exists():
        # Fallback to any existing F1_scaled.png
        img_path = list(Path("data/CubiCasa5k").rglob("F1_scaled.png"))[0]
    raw_img = Image.open(img_path).convert("RGB")
    iw, ih = raw_img.size
    size = 512
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2

    resized = raw_img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    norm = ((arr - mean) / std).transpose((2, 0, 1))
    tensor = torch.from_numpy(norm).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        struct_pred = torch.argmax(out["struct"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        room_pred = torch.argmax(out["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    print("Predicted Struct Unique Classes:", np.unique(struct_pred))
    print("Predicted Room Unique Classes:", np.unique(room_pred))

    mask_4class = np.zeros_like(struct_pred)
    mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1
    mask_4class[struct_pred == 3] = 2
    mask_4class[struct_pred == 4] = 3

    vectorizer = FloorplanVectorizer()
    layout = vectorizer.vectorize(
        mask=mask_4class,
        struct_mask=struct_pred,
        room_mask=room_pred,
        scale_override_m_per_px=0.03
    )

    walls = layout.get("walls", [])
    rooms = layout.get("rooms", [])
    print(f"Extracted {len(walls)} walls and {len(rooms)} semantic rooms from raw image upload!")

    for r in rooms:
        print(f"  -> Room: {r['room_type']} ({r['surface_m2']} m2)")
    for w in walls[:5]:
        print(f"  -> Wall: {w['id']} (is_exterior: {w['is_exterior']}, openings: {len(w.get('openings', []))})")

    # Bridge to Facade Generator
    bridge = FacadeBridge()
    facade_res = bridge.process_layout_facades(layout, num_floors=2, output_dir="models/samples/facades_pred_test")
    print(f"Facade bridge processed {facade_res['exterior_walls_count']} exterior walls. Front wall: {facade_res['front_wall_id']}")

if __name__ == "__main__":
    run_test()
