"""
Detailed Profiling and Latency Decomposition of the Facade Generation Stage.
Decomposes where the ~165 ms time is spent across 20 representative floorplans:
1. Vector geometry analysis & sketch canvas drawing (PIL ImageDraw).
2. PNG buffer encoding (sketch -> bytes).
3. PyTorch CUDA tensor preparation & GPU inference forward pass.
4. PNG buffer decoding & PIL image instantiation.
5. Disk I/O (saving PNG textures to disk).
6. Comparison: All exterior walls vs Front-Wall Only (on-demand).
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.floorplan.dataset import CubiCasaDataset
from src.floorplan.models import MultiTaskFloorplanUNet
from src.floorplan.vectorizer import FloorplanVectorizer
from src.bridge.facade_bridge import FacadeBridge

def profile_facade_pipeline(num_plans: int = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"[FACADE PROFILING] Decomposing Facade Stage on {num_plans} Plans ({device.upper()})")
    print("=" * 80)

    dataset = CubiCasaDataset(split="val", img_size=512, augment=False)
    vectorizer = FloorplanVectorizer()
    bridge = FacadeBridge(prefer_cuda=True)

    # Collect layouts with exterior walls
    layouts = []
    ckpt_path = PROJECT_ROOT / "models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth"
    model = MultiTaskFloorplanUNet(encoder_name="resnet34", num_struct_classes=5, num_room_classes=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    with torch.no_grad():
        for i in range(num_plans):
            batch = dataset[i]
            img = batch["image"].unsqueeze(0).to(device)
            out = model(img)
            p_s = torch.argmax(out["struct"], dim=1)[0].cpu().numpy().astype(np.uint8)
            p_r = torch.argmax(out["room"], dim=1)[0].cpu().numpy().astype(np.uint8)

            m4 = np.zeros_like(p_s)
            m4[(p_s == 1) | (p_s == 2)] = 1
            m4[p_s == 3] = 2
            m4[p_s == 4] = 3

            layout = vectorizer.vectorize(mask=m4, struct_mask=p_s, room_mask=p_r)
            layouts.append(layout)

    # Now profile the facade bridge operations
    t_sketch_draw = []
    t_png_encode = []
    t_gpu_infer = []
    t_png_decode = []
    t_disk_save = []
    t_total_all_walls = []
    t_total_front_wall = []
    ext_wall_counts = []

    out_temp = PROJECT_ROOT / "outputs" / "temp_profile"
    out_temp.mkdir(parents=True, exist_ok=True)

    for idx, layout in enumerate(layouts):
        ext_walls = [w for w in layout.get("walls", []) if w.get("is_exterior", False)]
        ext_wall_counts.append(len(ext_walls))
        if not ext_walls:
            continue

        # Benchmark 1: Front Wall Only (On-Demand Mode)
        front_wall = ext_walls[0]
        for w in ext_walls:
            if len(w.get("doors", [])) > 0:
                front_wall = w
                break

        t0_front = time.perf_counter()
        res_front = bridge.render_wall_facade(front_wall, is_front=True)
        t_total_front_wall.append((time.perf_counter() - t0_front) * 1000)

        # Benchmark 2: Detailed Decomposition across ALL exterior walls
        t0_all = time.perf_counter()
        for w in ext_walls:
            # Stage A: Sketch drawing
            t0 = time.perf_counter()
            sketch_img, uv = bridge.synthesize_facade_sketch(w, is_front=(w == front_wall))
            t_sketch_draw.append((time.perf_counter() - t0) * 1000)

            # Stage B: PNG buffer encode
            import io
            t0 = time.perf_counter()
            buf = io.BytesIO()
            sketch_img.save(buf, format="PNG")
            sketch_bytes = buf.getvalue()
            t_png_encode.append((time.perf_counter() - t0) * 1000)

            # Stage C: GPU Inference
            t0 = time.perf_counter()
            if bridge.inference_engine is not None:
                photo_bytes = bridge.inference_engine.generate(sketch_bytes)
            else:
                photo_bytes = sketch_bytes
            t_gpu_infer.append((time.perf_counter() - t0) * 1000)

            # Stage D: PNG buffer decode
            t0 = time.perf_counter()
            photo_img = Image.open(io.BytesIO(photo_bytes))
            photo_img.load()
            t_png_decode.append((time.perf_counter() - t0) * 1000)

            # Stage E: Disk save
            t0 = time.perf_counter()
            save_path = out_temp / f"test_{idx}_{w.get('id', 'wall')}.png"
            photo_img.save(save_path)
            t_disk_save.append((time.perf_counter() - t0) * 1000)

        t_total_all_walls.append((time.perf_counter() - t0_all) * 1000)

    # Clean up temp files
    for p in out_temp.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("FACADE LATENCY BREAKDOWN (Per Wall & End-to-End)")
    print("=" * 70)
    print(f"Average Exterior Walls / Plan: {np.mean(ext_wall_counts):.1f} (min={np.min(ext_wall_counts)}, max={np.max(ext_wall_counts)})")
    print("-" * 70)
    print("Per Single Exterior Wall Breakdown:")
    print(f"  1. Semantic Sketch Drawing (PIL):      {np.mean(t_sketch_draw):6.2f} ms")
    print(f"  2. PNG Buffer Encode (PIL -> bytes):   {np.mean(t_png_encode):6.2f} ms")
    print(f"  3. Neural Inference (PyTorch CUDA):    {np.mean(t_gpu_infer):6.2f} ms")
    print(f"  4. PNG Buffer Decode (bytes -> PIL):   {np.mean(t_png_decode):6.2f} ms")
    print(f"  5. File Save (Disk I/O per wall):      {np.mean(t_disk_save):6.2f} ms")
    print(f"  -> Total Per-Wall Time:                {np.mean(t_sketch_draw)+np.mean(t_png_encode)+np.mean(t_gpu_infer)+np.mean(t_png_decode)+np.mean(t_disk_save):6.2f} ms")
    print("-" * 70)
    print("Full Building Extrusion Stage Latency:")
    print(f"  A. ALL Exterior Walls (Current loop):  Mean = {np.mean(t_total_all_walls):6.1f} ms | Median = {np.median(t_total_all_walls):6.1f} ms | P95 = {np.percentile(t_total_all_walls, 95):6.1f} ms")
    print(f"  B. Front Wall Only (On-Demand Mode):   Mean = {np.mean(t_total_front_wall):6.1f} ms | Median = {np.median(t_total_front_wall):6.1f} ms | P95 = {np.percentile(t_total_front_wall, 95):6.1f} ms")
    print("=" * 70)

if __name__ == "__main__":
    profile_facade_pipeline(num_plans=20)
