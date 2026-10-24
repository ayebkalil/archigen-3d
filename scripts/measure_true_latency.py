"""
Accurate End-to-End Latency Profiler for Floorplan Pipeline
Measures true independent wall-clock time for:
  1. Single-pass call (96% clean path)
  2. Fallback dual-pass call (4% out-of-domain path)
Runs on CUDA (RTX 3050 Laptop GPU) with 30 warmups and 30 timed iterations.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.floorplan_engine import FloorplanPipeline

def profile_latency():
    pipeline = FloorplanPipeline.get_instance()
    device = pipeline.device
    print(f"[LATENCY] Profiling on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    manifest_p = PROJECT_ROOT / "data" / "dev_external_plans" / "dev_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # dev_01 is standard single-pass (no fallback)
    p_single = manifest_p.parent / manifest[0]["filename"]
    with open(p_single, "rb") as f:
        raw_single = f.read()

    # dev_03 is fallback trigger (white-on-blue blueprint triggers second pass)
    p_fallback = manifest_p.parent / manifest[2]["filename"]
    with open(p_fallback, "rb") as f:
        raw_fallback = f.read()

    # Warmup
    print("[LATENCY] Running 10 warmup iterations...")
    for _ in range(10):
        pipeline.process_image(raw_single, tta_mode="none", front_only=True)
        pipeline.process_image(raw_fallback, tta_mode="none", front_only=True)

    # 1. Profile Single Pass End-to-End
    N = 30
    latencies_single = []
    for _ in range(N):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = pipeline.process_image(raw_single, tta_mode="none", front_only=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_single.append((t1 - t0) * 1000.0)

    routed_single = res["layout"]["metadata"].get("routing", {})

    # 2. Profile Dual-Pass Fallback End-to-End
    latencies_fallback = []
    for _ in range(N):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res_fb = pipeline.process_image(raw_fallback, tta_mode="none", front_only=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_fallback.append((t1 - t0) * 1000.0)

    routed_fallback = res_fb["layout"]["metadata"].get("routing", {})

    print("\n" + "=" * 80)
    print("VERIFIED FULL-CALL PIPELINE LATENCY (N=30 iterations, RTX 3050 Laptop GPU)")
    print("=" * 80)
    print(f"Path 1: Standard Single-Pass Call (dev_01, routing={routed_single.get('model_used')}, fallback={routed_single.get('fallback_triggered')}):")
    print(f"  Mean Total Latency:    {np.mean(latencies_single):6.2f} ms")
    print(f"  Median Latency:        {np.median(latencies_single):6.2f} ms")
    print(f"  Std Dev:              ±{np.std(latencies_single):6.2f} ms")
    print(f"  Min / Max:             {np.min(latencies_single):6.2f} ms / {np.max(latencies_single):6.2f} ms")
    print("-" * 80)
    print(f"Path 2: Dual-Pass Fallback Call (dev_03, routing={routed_fallback.get('model_used')}, fallback={routed_fallback.get('fallback_triggered')}):")
    print(f"  Mean Total Latency:    {np.mean(latencies_fallback):6.2f} ms")
    print(f"  Median Latency:        {np.median(latencies_fallback):6.2f} ms")
    print(f"  Std Dev:              ±{np.std(latencies_fallback):6.2f} ms")
    print(f"  Min / Max:             {np.min(latencies_fallback):6.2f} ms / {np.max(latencies_fallback):6.2f} ms")
    print("=" * 80)

    out_json = PROJECT_ROOT / "outputs" / "pipeline_latency_measurement.json"
    with open(out_json, "w") as f:
        json.dump({
            "single_pass_ms": {
                "mean": round(float(np.mean(latencies_single)), 2),
                "median": round(float(np.median(latencies_single)), 2),
                "std": round(float(np.std(latencies_single)), 2),
                "min": round(float(np.min(latencies_single)), 2),
                "max": round(float(np.max(latencies_single)), 2)
            },
            "dual_pass_fallback_ms": {
                "mean": round(float(np.mean(latencies_fallback)), 2),
                "median": round(float(np.median(latencies_fallback)), 2),
                "std": round(float(np.std(latencies_fallback)), 2),
                "min": round(float(np.min(latencies_fallback)), 2),
                "max": round(float(np.max(latencies_fallback)), 2)
            }
        }, f, indent=2)
    print(f"[LATENCY] Results saved to {out_json}")

if __name__ == "__main__":
    profile_latency()
