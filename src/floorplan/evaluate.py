import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from safetensors.torch import load_file
from shapely.geometry import Polygon

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.vectorizer import FloorplanVectorizer


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    """Computes per-class IoU and mean IoU."""
    ious = {}
    class_names = {0: "Background", 1: "Wall", 2: "Door", 3: "Window"}
    valid_ious = []

    for c in range(num_classes):
        intersection = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        if union == 0:
            iou = 1.0  # Perfect agreement if class doesn't appear in either
        else:
            iou = float(intersection / union)
        ious[class_names[c]] = round(iou, 4)
        if c > 0:  # Foreground classes for foreground mIoU
            valid_ious.append(iou)

    ious["mIoU_foreground"] = round(float(np.mean(valid_ious)), 4)
    ious["mIoU_all"] = round(float(np.mean(list(ious.values())[:num_classes])), 4)
    return ious

def benchmark_floorplan_model(
    model_weights_path: str = "models/saved/floorplan_baseline/best.safetensors",
    split: str = "val",
    max_samples: int = 50
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BENCHMARK] Running evaluation on device: {device} (Samples: {max_samples})")

    # Load model
    model = smp.Unet(encoder_name="resnet34", classes=4).to(device)
    state_dict = load_file(model_weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset
    dataset = CubiCasaDataset(split=split, img_size=512, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    vectorizer = FloorplanVectorizer()

    total_wall_inter, total_wall_union = 0, 0
    total_door_inter, total_door_union = 0, 0
    total_win_inter, total_win_union = 0, 0

    valid_polygon_count = 0
    total_polygon_count = 0
    processed_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for b in range(preds.shape[0]):
                p_mask = preds[b]
                t_mask = masks[b]

                # Accumulate IoUs
                for c, (inter_acc, union_acc) in enumerate([
                    ("wall", (1,)), ("door", (2,)), ("win", (3,))
                ], 1):
                    inter = np.logical_and(p_mask == c, t_mask == c).sum()
                    union = np.logical_or(p_mask == c, t_mask == c).sum()
                    if c == 1:
                        total_wall_inter += inter; total_wall_union += union
                    elif c == 2:
                        total_door_inter += inter; total_door_union += union
                    elif c == 3:
                        total_win_inter += inter; total_win_union += union

                # Test Geometric Validity
                layout = vectorizer.vectorize(p_mask)
                for w in layout.get("walls", []):
                    pts = w["polygon_meters"]
                    if len(pts) >= 3:
                        poly = Polygon(pts)
                        total_polygon_count += 1
                        if poly.is_valid:
                            valid_polygon_count += 1

                processed_samples += 1
                if processed_samples >= max_samples:
                    break

            if processed_samples >= max_samples:
                break

    wall_iou = total_wall_inter / max(total_wall_union, 1)
    door_iou = total_door_inter / max(total_door_union, 1)
    win_iou = total_win_inter / max(total_win_union, 1)
    miou = (wall_iou + door_iou + win_iou) / 3.0
    validity_rate = (valid_polygon_count / max(total_polygon_count, 1)) * 100.0

    results = {
        "wall_iou": round(wall_iou, 4),
        "door_iou": round(door_iou, 4),
        "window_iou": round(win_iou, 4),
        "miou_foreground": round(miou, 4),
        "polygon_validity_rate_pct": round(validity_rate, 2),
        "num_evaluated_plans": processed_samples,
        "total_polygons_tested": total_polygon_count
    }

    print("\n" + "="*55)
    print("      CUBICASA5K QUANTITATIVE BENCHMARK REPORT      ")
    print("="*55)
    print(f"Evaluated Floorplans:           {processed_samples}")
    print(f"Wall IoU:                       {results['wall_iou'] * 100:.2f}%")
    print(f"Door IoU:                       {results['door_iou'] * 100:.2f}%")
    print(f"Window IoU:                     {results['window_iou'] * 100:.2f}%")
    print(f"Foreground mIoU:                {results['miou_foreground'] * 100:.2f}%")
    print(f"3D Shell Validity Rate:         {results['polygon_validity_rate_pct']:.2f}% ({valid_polygon_count}/{total_polygon_count} polygons valid)")
    print("="*55)

    return results

if __name__ == "__main__":
    benchmark_floorplan_model()
