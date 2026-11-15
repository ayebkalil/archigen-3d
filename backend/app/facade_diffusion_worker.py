"""CUDA Stable Diffusion material worker (Docker service ``facade-diffusion``).

It produces square, periodic *material swatches* only. Facade geometry, opening
masks and composition stay in the API. Weights are loaded once at startup into CPU
RAM; for each job the whole pipeline is moved to the GPU, run there entirely, and
parked back in CPU RAM so TripoSR keeps its VRAM between jobs. That parking is a
move between jobs, not inference offloading: no denoising step runs on the CPU.

GPU work is serialised with TripoSR through a file lock on a shared volume.
Failures are reported as failures; there is no substitute engine here.

    python -m backend.app.facade_diffusion_worker download   # one-shot weight fetch
    uvicorn backend.app.facade_diffusion_worker:app --port 8002
"""
from __future__ import annotations

import base64
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.material_diffusion import (  # noqa: E402
    config_sha256, evaluate_material, load_config, material_key, style_entry,
)

MODEL_DIR = Path(os.getenv("ARCHIGEN_SD_MODEL_DIR", ROOT / "models" / "facade_hybrid"))
CACHE_DIR = Path(os.getenv("ARCHIGEN_SD_CACHE_DIR", ROOT / "outputs" / "facade_material_cache"))
GPU_LOCK = os.getenv("ARCHIGEN_GPU_LOCK")
TERMINAL = {"succeeded", "failed", "cancelled"}


def snapshot_dir(config) -> Path:
    model = config["model"]
    return MODEL_DIR / model["repo"].replace("/", "--") / model["revision"]


# ---------------------------------------------------------------- weights
def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_weights(config, full: bool = False) -> Dict[str, Any]:
    """Size check always; SHA-256 once, remembered in a marker keyed by size+mtime."""
    root = snapshot_dir(config)
    missing = [f for f in config["model"]["files"] if not (root / f).is_file()]
    if missing:
        return {"ok": False, "missing": missing, "path": str(root)}
    marker_path = root / ".archigen-verified.json"
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
    changed = False
    for name, spec in config["model"]["weights"].items():
        path = root / name
        stat = path.stat()
        if stat.st_size != spec["size"]:
            return {"ok": False, "error": f"{name}: size {stat.st_size} != {spec['size']}", "path": str(root)}
        stamp = [stat.st_size, stat.st_mtime_ns, spec["sha256"]]
        if full or marker.get(name) != stamp:
            if _sha256(path) != spec["sha256"]:
                return {"ok": False, "error": f"{name}: SHA-256 mismatch", "path": str(root)}
            marker[name], changed = stamp, True
    if changed:
        marker_path.write_text(json.dumps(marker, indent=2))
    return {"ok": True, "path": str(root)}


def download(config) -> None:
    from huggingface_hub import hf_hub_download
    from src.models.facade_download import download_weight
    model, root = config["model"], snapshot_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    for name in model["files"]:
        target = root / name
        if name in model["weights"]:
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            download_weight(url, target, model["weights"][name])
        elif not target.is_file():
            hf_hub_download(model["repo"], name, revision=model["revision"], local_dir=root)
        print(f"ready: {name}", flush=True)
    state = verify_weights(config, full=True)
    if not state["ok"]:
        raise SystemExit(f"Weight verification failed: {state}")
    print(f"Verified {model['repo']}@{model['revision']} in {root}", flush=True)


# ---------------------------------------------------------------- GPU runtime
class GpuLock:
    """Cross-container exclusive GPU section (no-op when no lock path is configured)."""

    def __init__(self, path: Optional[str]):
        self.local = threading.Lock()
        self.file = None
        if path:
            from filelock import FileLock
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file = FileLock(path)

    @contextlib.contextmanager
    def hold(self, timeout_s: float):
        if not self.local.acquire(timeout=timeout_s):
            raise TimeoutError("GPU lock wait timed out")
        try:
            if self.file is not None:
                self.file.acquire(timeout=timeout_s)
            try:
                yield
            finally:
                if self.file is not None:
                    self.file.release()
        finally:
            self.local.release()


def process_rss_mib() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return None


def device_used_mib(torch) -> Optional[float]:
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024 ** 2
    except Exception:
        return None


class MaterialPipeline:
    """Stable Diffusion 1.5 with circular padding, resident in CPU RAM between jobs."""

    def __init__(self, config):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to the facade-diffusion worker; check NVIDIA Docker GPU access")
        self.torch = torch
        self.config = config
        started = time.monotonic()
        root = snapshot_dir(config)
        pipe = StableDiffusionPipeline.from_pretrained(
            root, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.set_progress_bar_config(disable=True)
        circular = 0
        for module in (pipe.unet, pipe.vae):
            for layer in module.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    layer.padding_mode = "circular"
                    circular += 1
        self.pipe = pipe
        self.circular_layers = circular
        self.load_seconds = time.monotonic() - started
        self.device_name = torch.cuda.get_device_name(0)
        self.total_vram_mib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2

    def run(self, prompt, negative, seed, on_step, deadline):
        torch, gen = self.torch, self.config["generation"]
        timings = {}
        free_mib = torch.cuda.mem_get_info()[0] / 1024 ** 2
        if free_mib < gen["min_free_vram_mib"]:
            raise MemoryError(f"Only {free_mib:.0f} MiB VRAM free; {gen['min_free_vram_mib']} MiB required")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        self.pipe.to("cuda")
        torch.cuda.synchronize()
        timings["gpu_load_s"] = time.monotonic() - started
        peak_device = device_used_mib(torch) or 0.0

        def callback(pipe, step, timestep, kwargs):
            nonlocal peak_device
            peak_device = max(peak_device, device_used_mib(torch) or 0.0)
            on_step(step + 1)
            if time.monotonic() > deadline:
                raise TimeoutError("Stable Diffusion attempt exceeded its time limit")
            return kwargs

        try:
            started = time.monotonic()
            with torch.inference_mode():
                image = self.pipe(
                    prompt=prompt, negative_prompt=negative, width=gen["resolution"], height=gen["resolution"],
                    num_inference_steps=gen["steps"], guidance_scale=gen["guidance_scale"],
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                    callback_on_step_end=callback,
                ).images[0]
            torch.cuda.synchronize()
            timings["generation_s"] = time.monotonic() - started
            timings["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            timings["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024 ** 2
            timings["peak_device_used_mib"] = max(peak_device, device_used_mib(torch) or 0.0)
            return image, timings
        finally:
            started = time.monotonic()
            self.pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            timings["gpu_unload_s"] = time.monotonic() - started
            timings["idle_device_used_mib"] = device_used_mib(torch)
            timings["process_rss_mib"] = process_rss_mib()


def token_lengths(pipe, prompt, negative):
    tokenizer = pipe.tokenizer
    lengths = {"positive": len(tokenizer(prompt)["input_ids"]), "negative": len(tokenizer(negative)["input_ids"])}
    if max(lengths.values()) > tokenizer.model_max_length:
        raise ValueError(f"Prompt exceeds {tokenizer.model_max_length} CLIP tokens: {lengths}")
    return lengths


# ---------------------------------------------------------------- jobs
class MaterialService:
    def __init__(self, config, pipeline_factory=MaterialPipeline):
        self.config = config
        self.config_sha256 = config_sha256()
        self.pipeline_factory = pipeline_factory
        self.pipeline = None
        self.state = "starting"
        self.error = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        self.gpu = GpuLock(GPU_LOCK)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def start(self):
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _load(self):
        try:
            self.state = "verifying_weights"
            weights = verify_weights(self.config)
            if not weights["ok"]:
                raise FileNotFoundError(f"Stable Diffusion weights unavailable: {weights}")
            self.state = "loading_model"
            self.pipeline = self.pipeline_factory(self.config)
            self.state = "ready"
        except Exception as exc:  # reported through /health; never replaced by another engine
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def health(self):
        pipeline = self.pipeline
        return {
            "status": self.state, "error": self.error, "engine": "stable_diffusion_material",
            "model": self.config["model"]["repo"], "revision": self.config["model"]["revision"],
            "license": self.config["model"]["license"], "config_sha256": self.config_sha256,
            "device": getattr(pipeline, "device_name", None),
            "total_vram_mib": getattr(pipeline, "total_vram_mib", None),
            "load_seconds": getattr(pipeline, "load_seconds", None),
            "circular_conv_layers": getattr(pipeline, "circular_layers", None),
            "execution": "full GPU per job; weights parked in CPU RAM between jobs",
            "gpu_lock": GPU_LOCK or "process-local only",
            "queued": self.queue.qsize(), "process_rss_mib": process_rss_mib(),
            "cached_materials": len(list(CACHE_DIR.glob("*.png"))),
        }

    def _cache_paths(self, key):
        return CACHE_DIR / f"{key}.png", CACHE_DIR / f"{key}.json"

    def _cached(self, key):
        png, meta = self._cache_paths(key)
        if png.is_file() and meta.is_file():
            record = json.loads(meta.read_text())
            record["image_base64"] = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
            return record
        return None

    def submit(self, style, seed, use_cache=True):
        style_entry(self.config, style)
        if self.state == "error":
            raise RuntimeError(f"Stable Diffusion worker failed to start: {self.error}")
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "style": style, "seed": int(seed), "status": "queued", "phase": "queued",
               "step": 0, "total_steps": self.config["generation"]["steps"], "attempt": 0,
               "created": time.time(), "cache_hit": False, "attempts": []}
        if use_cache:
            key = material_key(self.config, style, seed)
            cached = self._cached(key)
            if cached:
                job.update(status="succeeded", phase="cache_hit", cache_hit=True, result=cached, finished=time.time())
        with self.lock:
            self.jobs[job_id] = job
            self._prune()
        if job["status"] == "queued":
            self.queue.put(job_id)
        return self.public(job)

    def _prune(self):
        if len(self.jobs) > 200:
            done = sorted((j for j in self.jobs.values() if j["status"] in TERMINAL), key=lambda j: j["created"])
            for job in done[:len(self.jobs) - 200]:
                self.jobs.pop(job["job_id"], None)

    def public(self, job):
        out = {k: v for k, v in job.items() if k != "result"}
        if "result" in job:
            out["result"] = job["result"]
        out["queue_position"] = list(self.queue.queue).index(job["job_id"]) + 1 if job["job_id"] in self.queue.queue else 0
        return out

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job and job["status"] not in TERMINAL:
            job["cancel_requested"] = True
        return job

    def _loop(self):
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if not job:
                continue
            if job.get("cancel_requested"):
                job.update(status="cancelled", phase="cancelled", finished=time.time())
                continue
            try:
                self._run(job)
            except Exception as exc:
                job.update(status="failed", phase="failed", error=f"{type(exc).__name__}: {exc}",
                           error_type=classify_error(exc), finished=time.time())
                traceback.print_exc()

    def _run(self, job):
        gen = self.config["generation"]
        while self.state in ("starting", "verifying_weights", "loading_model"):
            job["phase"] = "loading_model"
            time.sleep(0.5)
        if self.state != "ready":
            raise RuntimeError(f"Stable Diffusion worker is not ready: {self.error}")
        entry = style_entry(self.config, job["style"])
        prompt, negative = entry["prompt"], self.config["negative_prompt"]
        lengths = token_lengths(self.pipeline.pipe, prompt, negative)
        job.update(status="running", phase="waiting_gpu", started=time.time())
        wait_started = time.monotonic()
        with self.gpu.hold(gen["gpu_wait_timeout_s"]):
            job["gpu_wait_s"] = time.monotonic() - wait_started
            for attempt in range(gen["max_attempts"]):
                if job.get("cancel_requested"):
                    job.update(status="cancelled", phase="cancelled", finished=time.time())
                    return
                seed = job["seed"] + attempt * gen["seed_stride"]
                job.update(attempt=attempt + 1, phase="generating", step=0)
                image, timings = self.pipeline.run(
                    prompt, negative, seed, lambda step: job.__setitem__("step", step),
                    time.monotonic() + gen["attempt_timeout_s"])
                job["phase"] = "checking_material"
                verdict = evaluate_material(image, entry, self.config["quality_gates"])
                record = {"seed": seed, "timings": timings, "quality": verdict}
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                record["image_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
                record["image_base64"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                job["attempts"].append(record)
                if verdict["passed"]:
                    self._store(job, record, prompt, negative, lengths)
                    return
        job.update(status="failed", phase="failed", error_type="material_quality_gate",
                   error=f"All {gen['max_attempts']} material attempts failed the automated checks: " +
                         " | ".join("; ".join(a["quality"]["failures"]) for a in job["attempts"]),
                   finished=time.time())

    def _store(self, job, record, prompt, negative, lengths):
        key = material_key(self.config, job["style"], job["seed"])
        png, meta = self._cache_paths(key)
        image_bytes = base64.b64decode(record["image_base64"].split(",", 1)[1])
        packages = {}
        for name in ("torch", "diffusers", "transformers", "accelerate"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {
            "material_key": key, "style": job["style"], "requested_seed": job["seed"], "seed": record["seed"],
            "prompt": prompt, "negative_prompt": negative, "token_lengths": lengths,
            "model": self.config["model"]["repo"], "revision": self.config["model"]["revision"],
            "license": self.config["model"]["license"], "generation": self.config["generation"],
            "config_sha256": self.config_sha256, "device": self.pipeline.device_name,
            "precision": "fp16", "offload": "none during inference; CPU RAM parking between jobs",
            "quality": record["quality"], "timings": record["timings"], "image_sha256": record["image_sha256"],
            "rejected_attempts": [{k: a[k] for k in ("seed", "quality", "timings", "image_sha256")}
                                  for a in job["attempts"][:-1]],
            "packages": packages, "created": time.time(),
        }
        tmp = png.with_suffix(".tmp")
        tmp.write_bytes(image_bytes)
        tmp.replace(png)
        meta.write_text(json.dumps(metadata, indent=2))
        result = {**metadata, "image_base64": record["image_base64"],
                  "rejected_images_base64": [a["image_base64"] for a in job["attempts"][:-1]]}
        job.update(status="succeeded", phase="succeeded", result=result, finished=time.time())


def classify_error(exc) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or isinstance(exc, MemoryError):
        return "gpu_out_of_memory"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if "cuda" in text:
        return "gpu_error"
    return "generation_error"


# ---------------------------------------------------------------- HTTP
from pydantic import BaseModel, Field  # noqa: E402


class MaterialRequest(BaseModel):
    style: str
    seed: int = Field(default=42, ge=0, le=2 ** 31 - 1)
    use_cache: bool = True


def create_app(service: Optional[MaterialService] = None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="ArchiGen Stable Diffusion material worker")
    app.state.service = service

    @app.on_event("startup")
    def startup():
        if app.state.service is None:
            app.state.service = MaterialService(load_config())
            app.state.service.start()

    @app.get("/health")
    def health():
        state = app.state.service.health()
        if state["status"] == "error":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content=state)
        return state

    @app.post("/materials")
    def submit(request: MaterialRequest):
        try:
            return app.state.service.submit(request.style, request.seed, request.use_cache)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/materials/{job_id}")
    def status(job_id: str):
        job = app.state.service.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown material job")
        return app.state.service.public(job)

    @app.delete("/materials/{job_id}")
    def cancel(job_id: str):
        job = app.state.service.cancel(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown material job")
        return app.state.service.public(job)

    return app


app = create_app()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download(load_config())
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        print(json.dumps(verify_weights(load_config(), full=True), indent=2))
    else:
        raise SystemExit("usage: python -m backend.app.facade_diffusion_worker download|verify")
