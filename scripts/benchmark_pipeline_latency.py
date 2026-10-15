"""
End-to-End Latency Benchmark for ArchiGen 3D Production Pipeline.
Measures real latency across 20 representative floorplans from the validation set:
1. Raw image file loading & byte decoding.
2. Letterbox padding to 512x512 and ImageNet normalization.
3. Multi-Task ResNet-34 UNet++ neural inference (Structural 5-class + Semantic Room 12-class).
4. Geometric vectorization (morphology, Douglas-Peucker, rectilinear snapping, hybrid room partitioning, shell validation).
5. Facade Bridge (aspect-ratio sketch synthesis + Pix2Pix neural texture generation per exterior wall).
6. 3D Scene Extrusion (Three.js geometry + UV mapping + JSON serialization).

Reports:
- Per-stage breakdown (mean & median ms).
- Total pipeline latency: Mean, Median, Min, Max, P90, P95.
- Both GPU mode (CUDA) and CPU mode.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

def collect_sample_images(num_samples: int = 20) -> List[Path]:
    root_dir = PROJECT_ROOT / "data" / "CubiCasa5k" / "cubicasa5k" / "cubicasa5k"
    val_file = root_dir / "val.txt"
    if not val_file.exists():
        raise FileNotFoundError(f"val.txt not found at {val_file}")

    image_paths = []
    with open(val_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for rel_path in lines:
        clean_rel = rel_path.strip().lstrip("/").lstrip("\\")
        folder = root_dir / clean_rel
        # Look for F1_scaled.png or F1_original.png
        img_p = folder / "F1_scaled.png"
        if not img_p.exists():
            img_p = folder / "F1_original.png"
        if img_p.exists():
            image_paths.append(img_p)
            if len(image_paths) >= num_samples:
                break

    return image_paths

def benchmark_device(device_str: str, image_paths: List[Path]) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"BENCHMARKING PIPELINE ON: {device_str.upper()}")
    print(f"{'='*70}")

    pipeline = FloorplanPipeline(device=device_str)

    # Warmup with first plan
    print("[*] Running 2 warmup passes to eliminate JIT/allocation overhead...")
    if len(image_paths) > 0:
        with open(image_paths[0], "rb") as f:
            warmup_bytes = f.read()
        pipeline.process_image(warmup_bytes)
        pipeline.process_image(warmup_bytes)

    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[*] Running timed evaluation on {len(image_paths)} test floorplans...")

    total_times = []
    stage_breakdowns = {
        "preprocess_ms": [],
        "neural_inference_ms": [],
        "vectorization_ms": [],
        "facade_bridge_ms": [],
        "extrusion_3d_ms": []
    }
    ext_wall_counts = []
    room_counts = []

    for idx, img_path in enumerate(image_paths):
        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        t_start = time.perf_counter()

        # Step 1: Preprocess
        t0 = time.perf_counter()
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

        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(canvas).to(pipeline.device)
        tensor = (tensor - pipeline.mean) / pipeline.std
        tensor = tensor.unsqueeze(0)
        if device_str == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Step 2: Neural Inference
        with torch.no_grad():
            outputs = pipeline.multitask_model(tensor)
            struct_pred = torch.argmax(outputs["struct"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            room_pred = torch.argmax(outputs["room"], dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            mask_4class = np.zeros_like(struct_pred)
            mask_4class[(struct_pred == 1) | (struct_pred == 2)] = 1
            mask_4class[struct_pred == 3] = 2
            mask_4class[struct_pred == 4] = 3

        if device_str == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Step 3: Geometric Vectorization
        layout = pipeline.vectorizer.vectorize(
            mask=mask_4class,
            struct_mask=struct_pred,
            room_mask=room_pred
        )
        t3 = time.perf_counter()

        # Step 4: Facade Bridge
        facade_meta = pipeline.bridge.process_layout_facades(
            layout=layout,
            front_wall_id=None,
            num_floors=1
        )
        if device_str == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        # Step 5: 3D Scene Extrusion
        threejs_scene = pipeline.extruder.generate_threejs_scene_data(layout)
        t5 = time.perf_counter()

        total_elapsed = (t5 - t_start) * 1000.0
        total_times.append(total_elapsed)

        stage_breakdowns["preprocess_ms"].append((t1 - t0) * 1000.0)
        stage_breakdowns["neural_inference_ms"].append((t2 - t1) * 1000.0)
        stage_breakdowns["vectorization_ms"].append((t3 - t2) * 1000.0)
        stage_breakdowns["facade_bridge_ms"].append((t4 - t3) * 1000.0)
        stage_breakdowns["extrusion_3d_ms"].append((t5 - t4) * 1000.0)

        num_ext = len([w for w in layout.get("walls", []) if w.get("is_exterior")])
        num_r = len(layout.get("rooms", []))
        ext_wall_counts.append(num_ext)
        room_counts.append(num_r)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(image_paths):
            print(f"  [{idx+1:02d}/{len(image_paths)}] Total: {total_elapsed:6.1f} ms | Infer: {(t2-t1)*1000:5.1f} ms | Vec: {(t3-t2)*1000:5.1f} ms | Facade ({num_ext} walls): {(t4-t3)*1000:5.1f} ms | 3D: {(t5-t4)*1000:4.1f} ms")

    total_arr = np.array(total_times)
    metrics = {
        "device": device_str,
        "facade_backend": pipeline.bridge.device_name,
        "n_samples": len(image_paths),
        "total_mean_ms": float(np.mean(total_arr)),
        "total_median_ms": float(np.median(total_arr)),
        "total_min_ms": float(np.min(total_arr)),
        "total_max_ms": float(np.max(total_arr)),
        "total_p90_ms": float(np.percentile(total_arr, 90)),
        "total_p95_ms": float(np.percentile(total_arr, 95)),
        "avg_ext_walls": float(np.mean(ext_wall_counts)),
        "avg_rooms": float(np.mean(room_counts)),
        "stages_mean_ms": {k: float(np.mean(v)) for k, v in stage_breakdowns.items()},
        "stages_median_ms": {k: float(np.median(v)) for k, v in stage_breakdowns.items()}
    }

    print(f"\n--- {device_str.upper()} SUMMARY ---")
    print(f"Total Mean Latency:   {metrics['total_mean_ms']:.1f} ms")
    print(f"Total Median Latency: {metrics['total_median_ms']:.1f} ms")
    print(f"90th Percentile:      {metrics['total_p90_ms']:.1f} ms")
    print(f"95th Percentile:      {metrics['total_p95_ms']:.1f} ms")
    print(f"Stages (Median):")
    for s_name, s_val in metrics["stages_median_ms"].items():
        print(f"  - {s_name:22s}: {s_val:6.1f} ms")

    return metrics

def main():
    samples = collect_sample_images(20)
    print(f"Collected {len(samples)} validation floorplan images for latency benchmark.")

    results = {}
    if torch.cuda.is_available():
        results["cuda"] = benchmark_device("cuda", samples)

    results["cpu"] = benchmark_device("cpu", samples)

    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<25} | {'CUDA (GPU)':<20} | {'CPU':<20}")
    print("-" * 70)
    cuda_res = results.get("cuda", {})
    cpu_res = results.get("cpu", {})
    print(f"{'Facade Engine':<25} | {cuda_res.get('facade_backend', 'N/A'):<20} | {cpu_res.get('facade_backend', 'N/A'):<20}")
    print(f"{'Neural Infer (Median)':<25} | {cuda_res.get('stages_median_ms', {}).get('neural_inference_ms', 0):>17.1f} ms | {cpu_res.get('stages_median_ms', {}).get('neural_inference_ms', 0):>17.1f} ms")
    print(f"{'Facade Bridge (Median)':<25} | {cuda_res.get('stages_median_ms', {}).get('facade_bridge_ms', 0):>17.1f} ms | {cpu_res.get('stages_median_ms', {}).get('facade_bridge_ms', 0):>17.1f} ms")
    print(f"{'Vectorization (Median)':<25} | {cuda_res.get('stages_median_ms', {}).get('vectorization_ms', 0):>17.1f} ms | {cpu_res.get('stages_median_ms', {}).get('vectorization_ms', 0):>17.1f} ms")
    print(f"{'3D Extrusion (Median)':<25} | {cuda_res.get('stages_median_ms', {}).get('extrusion_3d_ms', 0):>17.1f} ms | {cpu_res.get('stages_median_ms', {}).get('extrusion_3d_ms', 0):>17.1f} ms")
    print("-" * 70)
    print(f"{'TOTAL PIPELINE (Median)':<25} | {cuda_res.get('total_median_ms', 0):>17.1f} ms | {cpu_res.get('total_median_ms', 0):>17.1f} ms")
    print(f"{'TOTAL PIPELINE (P95)':<25} | {cuda_res.get('total_p95_ms', 0):>17.1f} ms | {cpu_res.get('total_p95_ms', 0):>17.1f} ms")
    print("=" * 70)

if __name__ == "__main__":
    main()
