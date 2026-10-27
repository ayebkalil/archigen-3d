"""
Compare layouts produced by ONNX CPU mode vs PyTorch mode on 10 floorplan plans.
Measures and reports:
- Number of rooms detected
- Room classification concordance
- Wall segment count concordance
- Mean vertex delta and polygon IoU
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from shapely.geometry import Polygon
from backend.app.floorplan_engine import FloorplanPipeline
from src.floorplan.vectorizer import FloorplanVectorizer

def find_test_plans(n=50):
    candidates = []
    # 1. Real external plans
    for p in sorted((PROJECT_ROOT / "data/real_external_plans").rglob("*.*")):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"] and not p.name.startswith("."):
            candidates.append(p)
    # 2. Dev external plans
    for p in sorted((PROJECT_ROOT / "data/dev_external_plans").rglob("*.*")):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"] and not p.name.startswith("."):
            candidates.append(p)
    # 3. CubiCasa validation plans
    cubi_dir = PROJECT_ROOT / "data/CubiCasa5k"
    if cubi_dir.exists():
        for p in sorted(cubi_dir.rglob("F1_scaled.png")):
            candidates.append(p)
            if len(candidates) >= n:
                break
    return candidates[:n]

def compute_polygon_iou(poly1_coords, poly2_coords):
    try:
        p1 = Polygon(poly1_coords)
        p2 = Polygon(poly2_coords)
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        if union <= 1e-6:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0

def main():
    plans = find_test_plans(50)
    print(f"Found {len(plans)} plans for ONNX vs PyTorch layout comparison:")
    for i, p in enumerate(plans[:10]):
        print(f"  Plan {i+1}: {p.name} ({p.parent.name})")
    if len(plans) > 10:
        print(f"  ... and {len(plans) - 10} more plans")

    # 1. Initialize PyTorch Pipeline (forcing PyTorch mode)
    print("\n[1/3] Initializing PyTorch pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_torch = FloorplanPipeline(device=device, prefer_onnx=False)

    # 2. Initialize ONNX CPU Pipeline (forcing true ONNX runtime execution)
    print("\n[2/3] Initializing ONNX CPU pipeline...")
    pipeline_onnx = FloorplanPipeline(device="cpu", prefer_onnx=True)

    results = []
    print(f"\n[3/3] Running comparative layout extraction on {len(plans)} plans...")

    for i, img_path in enumerate(plans):
        plan_name = f"{img_path.parent.name}/{img_path.name}"
        img_bytes = img_path.read_bytes()

        # PyTorch Inference
        t0 = time.perf_counter()
        res_torch = pipeline_torch.process_image(img_bytes)
        time_torch_ms = (time.perf_counter() - t0) * 1000.0
        layout_torch = res_torch.get("layout", {})

        # ONNX Inference
        t0 = time.perf_counter()
        res_onnx = pipeline_onnx.process_image(img_bytes)
        time_onnx_ms = (time.perf_counter() - t0) * 1000.0
        layout_onnx = res_onnx.get("layout", {})

        rooms_torch = layout_torch.get("rooms", [])
        rooms_onnx = layout_onnx.get("rooms", [])
        walls_torch = layout_torch.get("walls", [])
        walls_onnx = layout_onnx.get("walls", [])

        # Match rooms greedily by IoU
        matched_ious = []
        label_matches = 0
        for r_t in rooms_torch:
            poly_t = r_t.get("polygon_meters", [])
            type_t = r_t.get("room_type")
            best_iou = 0.0
            best_match = None
            for r_o in rooms_onnx:
                poly_o = r_o.get("polygon_meters", [])
                iou = compute_polygon_iou(poly_t, poly_o)
                if iou > best_iou:
                    best_iou = iou
                    best_match = r_o
            matched_ious.append(best_iou)
            if best_match and best_match.get("room_type") == type_t:
                label_matches += 1

        mean_room_iou = float(np.mean(matched_ious)) if matched_ious else 1.0
        label_concordance = float(label_matches / len(rooms_torch)) if rooms_torch else 1.0

        plan_res = {
            "plan_index": i + 1,
            "plan_name": plan_name,
            "torch_rooms_count": len(rooms_torch),
            "onnx_rooms_count": len(rooms_onnx),
            "room_count_diff": abs(len(rooms_torch) - len(rooms_onnx)),
            "torch_walls_count": len(walls_torch),
            "onnx_walls_count": len(walls_onnx),
            "wall_count_diff": abs(len(walls_torch) - len(walls_onnx)),
            "mean_room_polygon_iou": round(mean_room_iou, 4),
            "label_concordance_pct": round(label_concordance * 100, 2),
            "torch_runtime_ms": round(time_torch_ms, 1),
            "onnx_runtime_ms": round(time_onnx_ms, 1)
        }
        results.append(plan_res)
        print(f"  Plan {i+1:02d}: Rooms (PyTorch={len(rooms_torch)}, ONNX={len(rooms_onnx)}), "
              f"Walls (PyTorch={len(walls_torch)}, ONNX={len(walls_onnx)}), "
              f"Mean Room IoU: {mean_room_iou:.3f}, Label Match: {label_concordance*100:.1f}%, "
              f"ONNX Latency: {time_onnx_ms:.1f}ms")

    # Aggregate
    agg = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_plans_compared": len(results),
        "mean_room_count_diff": round(float(np.mean([r["room_count_diff"] for r in results])), 2),
        "mean_wall_count_diff": round(float(np.mean([r["wall_count_diff"] for r in results])), 2),
        "overall_mean_room_iou": round(float(np.mean([r["mean_room_polygon_iou"] for r in results])), 4),
        "overall_label_concordance_pct": round(float(np.mean([r["label_concordance_pct"] for r in results])), 2),
        "mean_torch_latency_ms": round(float(np.mean([r["torch_runtime_ms"] for r in results])), 1),
        "mean_onnx_latency_ms": round(float(np.mean([r["onnx_runtime_ms"] for r in results])), 1),
        "per_plan_details": results
    }

    out_file = PROJECT_ROOT / "outputs/design_stage4/onnx_vs_pytorch_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n========================================================")
    print("AGGREGATE ONNX vs PyTorch LAYOUT CONCORDANCE (10 Plans):")
    print(f"  Mean Room Count Diff:       {agg['mean_room_count_diff']}")
    print(f"  Mean Wall Count Diff:       {agg['mean_wall_count_diff']}")
    print(f"  Overall Mean Room IoU:      {agg['overall_mean_room_iou']*100:.2f}%")
    print(f"  Overall Label Concordance:  {agg['overall_label_concordance_pct']}%")
    print(f"  Mean PyTorch Latency:       {agg['mean_torch_latency_ms']} ms")
    print(f"  Mean ONNX CPU Latency:      {agg['mean_onnx_latency_ms']} ms")
    print("========================================================")

if __name__ == "__main__":
    main()
