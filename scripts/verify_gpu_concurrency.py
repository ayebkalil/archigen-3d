"""Start a TripoSR furniture reconstruction and a Stable Diffusion facade job at the same
time on the Docker stack and check that both finish, one after the other on the GPU.

    python scripts/verify_gpu_concurrency.py
Samples device memory with nvidia-smi every 0.5 s and writes a JSON report.
"""
import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, total = map(int, out.split(","))
    return used, total


def main():
    samples, stop, results = [], threading.Event(), {}
    started = time.monotonic()

    def sampler():
        while not stop.is_set():
            samples.append((round(time.monotonic() - started, 2), gpu_used()[0]))
            time.sleep(0.5)

    def furniture():
        image = base64.b64encode((ROOT / "frontend/public/samples/furniture/chair.png").read_bytes()).decode()
        t = time.monotonic()
        response = requests.post(f"{API}/furniture/generate-3d", json={"image_base64": "data:image/png;base64," + image,
                                                                      "category": "chair", "target_height_m": 0.85}, timeout=900)
        body = response.json()
        results["furniture"] = {"http": response.status_code, "start_s": round(t - started, 2), "end_s": round(time.monotonic() - started, 2),
                                "engine": body.get("engine"), "detail": body.get("detail"),
                                "peak_device_sampled_mib": (body.get("quality") or {}).get("peak_device_sampled_mib")}

    def facade():
        wall = {"id": "concurrency", "is_exterior": True, "height_m": 2.8, "polygon_meters": [[0, 0], [7, 0], [7, .2], [0, .2]],
                "openings": [{"id": "d", "type": "door", "position_along_wall_m": 3.5, "width_m": 1.0, "height_m": 2.1, "sill_m": 0}]}
        t = time.monotonic()
        job = requests.post(f"{API}/facade/hybrid/jobs", json={"wall": wall, "style": "Traditional Brick", "seed": 4242, "use_cache": False,
                                                              "tracking_tags": {"run_kind": "concurrency_test"}}, timeout=60).json()
        phases = []
        while job["status"] not in ("succeeded", "failed", "cancelled"):
            time.sleep(0.5)
            job = requests.get(f"{API}/facade/hybrid/jobs/{job['job_id']}", timeout=30).json()
            if not phases or phases[-1][1] != job["phase"]:
                phases.append((round(time.monotonic() - started, 2), job["phase"]))
        results["facade"] = {"status": job["status"], "error": job.get("error"), "start_s": round(t - started, 2),
                             "end_s": round(time.monotonic() - started, 2), "phases": phases,
                             "timings": (job.get("result") or {}).get("timings")}

    threads = [threading.Thread(target=sampler), threading.Thread(target=furniture)]
    threads[0].start(); threads[1].start()
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0)  # let TripoSR enter its GPU section first
    facade_thread = threading.Thread(target=facade)
    facade_thread.start()
    threads[1].join(); facade_thread.join()
    stop.set(); threads[0].join()
    report = {"results": results, "peak_device_used_mib": max(v for _, v in samples), "total_mib": gpu_used()[1],
              "samples": samples}
    out = ROOT / "outputs" / "facade_engine_comparison" / f"gpu-concurrency-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    ok = results["furniture"]["http"] == 200 and results["facade"]["status"] == "succeeded"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
