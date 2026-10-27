import requests
import time
import json
import numpy as np
from PIL import Image
import io
from pathlib import Path

normal_plan = Path("data/real_external_plans/heldout_modern/modern_06_caroline_court_apartment.png")
with open(normal_plan, "rb") as f:
    normal_bytes = f.read()

# Blank plan for triggering collapse fallback
img_blank = Image.new("RGB", (512, 512), (255, 255, 255))
buf = io.BytesIO()
img_blank.save(buf, format="PNG")
blank_bytes = buf.getvalue()

configs = [
    {
        "setting": "Normal Plan (modern_06) - TTA OFF",
        "file_bytes": normal_bytes,
        "filename": "modern_06.png",
        "use_tta": "false",
        "tta_mode": "none",
        "fallback_triggered": False
    },
    {
        "setting": "Normal Plan (modern_06) - TTA ON (4-fold)",
        "file_bytes": normal_bytes,
        "filename": "modern_06.png",
        "use_tta": "true",
        "tta_mode": "4_fold",
        "fallback_triggered": False
    },
    {
        "setting": "Collapse Plan (Blank 512x512) - Fallback Triggered - TTA OFF",
        "file_bytes": blank_bytes,
        "filename": "blank.png",
        "use_tta": "false",
        "tta_mode": "none",
        "fallback_triggered": True
    },
    {
        "setting": "Collapse Plan (Blank 512x512) - Fallback Triggered - TTA ON (4-fold)",
        "file_bytes": blank_bytes,
        "filename": "blank.png",
        "use_tta": "true",
        "tta_mode": "4_fold",
        "fallback_triggered": True
    }
]

url = "http://localhost:8000/predict/floorplan"
results = []

for cfg in configs:
    latencies = []
    setting_name = cfg["setting"]
    print(f"Benchmarking: {setting_name}")
    for r in range(20):
        t0 = time.perf_counter()
        res = requests.post(
            url,
            files={"file": (cfg["filename"], cfg["file_bytes"], "image/png")},
            data={"use_tta": cfg["use_tta"], "tta_mode": cfg["tta_mode"], "front_only": "true"}
        )
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        print(f"  run {r+1:02d}: {lat_ms:.2f} ms")
    
    # 20 runs
    arr20 = np.array(latencies)
    # Warmup excluded (runs 4..20, N=17)
    arr_no_warmup = np.array(latencies[3:])

    res_item = {
        "setting": cfg["setting"],
        "pipeline": "ONNX Runtime CPU",
        "use_tta": cfg["use_tta"] == "true",
        "fallback_triggered": cfg["fallback_triggered"],
        "raw_latencies_ms": [round(x, 2) for x in latencies],
        "total_runs": 20,
        "stats_all_20_runs": {
            "mean_ms": round(float(np.mean(arr20)), 2),
            "median_ms": round(float(np.median(arr20)), 2),
            "p95_ms": round(float(np.percentile(arr20, 95)), 2),
            "min_ms": round(float(np.min(arr20)), 2),
            "max_ms": round(float(np.max(arr20)), 2),
        },
        "warmup_runs_excluded": 3,
        "evaluated_steady_runs": 17,
        "stats_excluding_warmup": {
            "mean_ms": round(float(np.mean(arr_no_warmup)), 2),
            "median_ms": round(float(np.median(arr_no_warmup)), 2),
            "p95_ms": round(float(np.percentile(arr_no_warmup, 95)), 2),
            "min_ms": round(float(np.min(arr_no_warmup)), 2),
            "max_ms": round(float(np.max(arr_no_warmup)), 2),
        }
    }
    results.append(res_item)

output_path = Path("outputs/design_stage4/predict_floorplan_latency_20runs.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n[SUCCESS] Saved latency benchmarks with warmup exclusion to outputs/design_stage4/predict_floorplan_latency_20runs.json")
