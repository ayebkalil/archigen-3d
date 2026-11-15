"""Asynchronous "Stable Diffusion — hybrid" facade jobs in the API process.

Geometry and composition run here (CPU); the CUDA worker only returns a material
swatch. A failed worker, an unreadable swatch or a failed geometry check ends the job
as ``failed`` with the reason. Nothing falls back to Pix2Pix.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from PIL import Image

from src.bridge.facade_hybrid import (
    FacadeGeometryError, compose_hybrid_facade, geometry_hash, mask_image, validate_composition, wall_spec,
)
from src.models.material_diffusion import config_sha256, load_config, style_entry

TERMINAL = {"succeeded", "failed", "cancelled"}


def _b64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _decode(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


class WorkerClient:
    def __init__(self, url: Optional[str]):
        self.url = url.rstrip("/") if url else None

    def _client(self, timeout=10):
        import httpx
        return httpx.Client(timeout=timeout, trust_env=False)

    def health(self):
        if not self.url:
            return {"status": "not_configured", "error": "ARCHIGEN_FACADE_DIFFUSION_URL is not set"}
        import httpx
        try:
            with self._client(5) as client:
                return client.get(self.url + "/health").json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}

    def submit(self, style, seed, use_cache=True):
        with self._client() as client:
            response = client.post(self.url + "/materials", json={"style": style, "seed": seed, "use_cache": use_cache})
            if response.status_code >= 400:
                raise RuntimeError(response.json().get("detail", response.text))
            return response.json()

    def status(self, job_id):
        with self._client(30) as client:
            response = client.get(f"{self.url}/materials/{job_id}")
            response.raise_for_status()
            return response.json()

    def cancel(self, job_id):
        try:
            with self._client(5) as client:
                client.delete(f"{self.url}/materials/{job_id}")
        except Exception:
            pass


class HybridFacadeJobs:
    def __init__(self, worker: Optional[WorkerClient] = None, tracker=None, poll_s: float = 0.5):
        self.worker = worker or WorkerClient(os.getenv("ARCHIGEN_FACADE_DIFFUSION_URL"))
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.poll_s = poll_s
        if tracker is None:
            from src.models.facade_tracking import log_run
            tracker = log_run
        self.tracker = tracker

    # ------------------------------------------------------------- API
    def submit(self, wall: Dict[str, Any], style: str, seed: int = 42, num_floors: int = 1,
               use_cache: bool = True, run_tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        config = load_config()
        style_entry(config, style)  # ValueError -> 422
        spec = wall_spec(wall, num_floors=num_floors, tolerance_m=config["composition"]["edge_tolerance_m"])
        if not self.worker.url:
            raise RuntimeError("Stable Diffusion worker is not configured (ARCHIGEN_FACADE_DIFFUSION_URL). "
                               "Start the GPU stack with scripts/start_docker_stack.ps1.")
        job = {
            "job_id": uuid.uuid4().hex, "engine": "stable_diffusion_hybrid", "status": "queued", "phase": "queued",
            "wall_id": spec["wall_id"], "style": style, "seed": int(seed), "num_floors": num_floors,
            "geometry_hash": geometry_hash(spec), "created": time.time(), "step": 0,
            "total_steps": config["generation"]["steps"], "attempt": 0, "tracking": {"status": "pending"},
        }
        with self.lock:
            self.jobs[job["job_id"]] = job
            self._prune()
        context = {"wall": copy.deepcopy(wall), "spec": spec, "config": config, "use_cache": use_cache,
                   "tags": run_tags or {"run_kind": "studio_generation"}}
        threading.Thread(target=self._run, args=(job, context), daemon=True).start()
        return self.public(job)

    def get(self, job_id):
        job = self.jobs.get(job_id)
        return self.public(job) if job else None

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job and job["status"] not in TERMINAL:
            job["cancel_requested"] = True
            if job.get("worker_job_id"):
                self.worker.cancel(job["worker_job_id"])
        return self.get(job_id)

    def public(self, job):
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def _prune(self):
        if len(self.jobs) > 100:
            done = sorted((j for j in self.jobs.values() if j["status"] in TERMINAL), key=lambda j: j["created"])
            for old in done[:len(self.jobs) - 100]:
                self.jobs.pop(old["job_id"], None)

    # ------------------------------------------------------------- worker
    def _fail(self, job, error_type, message):
        job.update(status="failed", phase="failed", error_type=error_type, error=message, finished=time.time())

    def _run(self, job, context):
        config, spec = context["config"], context["spec"]
        gen = config["generation"]
        started = time.monotonic()
        composed = material_info = None
        deadline = started + gen["gpu_wait_timeout_s"] + gen["max_attempts"] * gen["attempt_timeout_s"] + 120
        try:
            job.update(status="running", phase="submitting")
            remote = self.worker.submit(job["style"], job["seed"], context["use_cache"])
            job["worker_job_id"] = remote["job_id"]
            while remote["status"] not in TERMINAL:
                if job.get("cancel_requested"):
                    self.worker.cancel(remote["job_id"])
                    job.update(status="cancelled", phase="cancelled", finished=time.time())
                    return
                if time.monotonic() > deadline:
                    self.worker.cancel(remote["job_id"])
                    return self._fail(job, "timeout", "Stable Diffusion job exceeded its overall time limit")
                time.sleep(self.poll_s)
                remote = self.worker.status(remote["job_id"])
                job.update(phase=remote.get("phase", job["phase"]), step=remote.get("step", 0),
                           attempt=remote.get("attempt", 0), queue_position=remote.get("queue_position", 0))
            if remote["status"] != "succeeded":
                return self._fail(job, remote.get("error_type", "worker_" + remote["status"]),
                                  remote.get("error", f"Material job {remote['status']}"))
            if job.get("cancel_requested"):
                job.update(status="cancelled", phase="cancelled", finished=time.time())
                return
            material_info = remote["result"]
            job["phase"] = "composing"
            material = _decode(material_info["image_base64"])
            if material.size != (gen["resolution"], gen["resolution"]):
                return self._fail(job, "invalid_material", f"Material has size {material.size}")
            tile_m = float(material_info["quality"]["scale"]["tile_m"])
            compose_started = time.monotonic()
            offset_seed = int(job["geometry_hash"][:8], 16) ^ job["seed"]
            composed = compose_hybrid_facade(spec, material, tile_m, config["composition"], offset_seed)
            composition_s = time.monotonic() - compose_started
            job["phase"] = "validating_geometry"
            validate_started = time.monotonic()
            validation = validate_composition(spec, composed)
            validation_s = time.monotonic() - validate_started
            if not validation["passed"]:
                job["validation"] = validation
                return self._fail(job, "geometry_validation", "Composed texture failed geometry checks: " +
                                  "; ".join(validation["failures"]))
            timings = dict(material_info.get("timings") or {}) if not remote.get("cache_hit") else {}
            timings.update(worker_gpu_wait_s=remote.get("gpu_wait_s"), composition_s=composition_s,
                           validation_s=validation_s, total_s=time.monotonic() - started)
            width, height = composed["size"]
            result = {
                "status": "success", "engine": "stable_diffusion_hybrid", "wall_id": spec["wall_id"],
                "style": job["style"], "num_floors": job["num_floors"],
                "wall_length_m": spec["wall_length_m"], "wall_height_m": spec["wall_height_m"],
                "uv_bounds": composed["uv_bounds"], "facade_frame": spec["frame"], "openings": composed["openings"],
                "geometry_hash": job["geometry_hash"],
                "texture_base64": _b64_png(composed["image"]),
                "sketch_base64": _b64_png(mask_image(composed["labels"])),
                "material_base64": material_info["image_base64"],
                "device": f"Stable Diffusion worker / {material_info.get('device', 'CUDA')}",
                "model_info": {
                    "model": "stable_diffusion_hybrid", "base_model": material_info["model"],
                    "revision": material_info["revision"], "license": material_info.get("license"),
                    "resolution": gen["resolution"], "native_material_px": gen["resolution"],
                    "texture_size_px": [width, height], "density_px_per_m": composed["density_px_per_m"],
                    "material_tile_m": tile_m, "material_tile_px": composed["tile_px"],
                    "material_repeats": composed["repeats"], "scale": material_info["quality"]["scale"],
                    "seed": material_info["seed"], "requested_seed": job["seed"],
                    "prompt": material_info["prompt"], "negative_prompt": material_info["negative_prompt"],
                    "scheduler": gen["scheduler"], "steps": gen["steps"], "guidance_scale": gen["guidance_scale"],
                    "tiling": gen["tiling"], "precision": material_info.get("precision"),
                    "offload": material_info.get("offload"),
                    "style_conditioned": True, "style_control": "text prompt (pretrained text guidance)",
                    "effective_style": job["style"] + " (SD text prompt)",
                    "material_key": material_info["material_key"], "cache_hit": bool(remote.get("cache_hit")),
                    "rejected_attempts": len(material_info.get("rejected_attempts", [])),
                    "material_quality": material_info["quality"], "config_sha256": material_info.get("config_sha256"),
                    "pbr_maps": "none (colour texture only)",
                    "opening_rendering": "schematic fill inside masks; 3D viewer uses its door/window meshes",
                },
                "validation": validation, "timings": timings,
            }
            job.update(status="succeeded", phase="succeeded", result=result, finished=time.time())
        except FacadeGeometryError as exc:
            self._fail(job, "invalid_geometry", str(exc))
        except Exception as exc:
            name = type(exc).__name__
            kind = "worker_unavailable" if name in ("ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError") else "hybrid_error"
            self._fail(job, kind, f"{name}: {exc}")
        finally:
            threading.Thread(target=self._track, args=(job, context, composed, material_info), daemon=True).start()

    # ------------------------------------------------------------- MLflow
    def _track(self, job, context, composed, material_info):
        spec, config = context["spec"], context["config"]
        result = job.get("result") or {}
        info = result.get("model_info", {})
        wall_json = json.dumps(context["wall"], sort_keys=True, default=str)
        params = {
            "engine": "stable_diffusion_hybrid", "model": config["model"]["repo"],
            "model_revision": config["model"]["revision"], "config_sha256": config_sha256(),
            "wall_id": spec["wall_id"], "wall_length_m": round(spec["wall_length_m"], 4),
            "wall_height_m": round(spec["wall_height_m"], 4), "num_floors": spec["num_floors"],
            "openings": len(spec["openings"]), "doors": sum(o["type"] == "door" for o in spec["openings"]),
            "windows": sum(o["type"] == "window" for o in spec["openings"]),
            "geometry_hash": job["geometry_hash"], "input_sha256": hashlib.sha256(wall_json.encode()).hexdigest(),
            "style": job["style"], "prompt": style_entry(config, job["style"])["prompt"],
            "negative_prompt": config["negative_prompt"], "requested_seed": job["seed"], "seed": info.get("seed"),
            "scheduler": config["generation"]["scheduler"], "steps": config["generation"]["steps"],
            "guidance_scale": config["generation"]["guidance_scale"], "resolution": config["generation"]["resolution"],
            "tiling": config["generation"]["tiling"], "precision": "fp16",
            "offload": "none during inference; CPU RAM parking between jobs",
            "gpu": (material_info or {}).get("device"), "status": job["status"],
            "error_type": job.get("error_type"), "cache_hit": info.get("cache_hit"),
            "material_key": info.get("material_key"), "scale_method": (info.get("scale") or {}).get("method"),
            "texture_size_px": info.get("texture_size_px"),
        }
        timings = result.get("timings") or (material_info or {}).get("timings") or {}
        metrics = {k: v for k, v in timings.items() if isinstance(v, (int, float))}
        quality = (info.get("material_quality") or {}).get("metrics", {})
        metrics.update({f"material_{k}": v for k, v in quality.items()})
        validation = result.get("validation") or job.get("validation") or {}
        metrics.update(geometry_passed=float(bool(validation.get("passed"))) if validation else 0.0,
                       max_mask_error_m=validation.get("max_mask_error_m"),
                       material_tile_m=info.get("material_tile_m"), rejected_attempts=info.get("rejected_attempts"),
                       succeeded=float(job["status"] == "succeeded"))
        artifacts = {"input/wall.json": context["wall"], "input/geometry_spec.json": spec,
                     "report/job.json": {k: v for k, v in job.items() if k != "result"},
                     "report/validation.json": validation or None}
        if composed is not None:
            artifacts["input/opening_mask.png"] = mask_image(composed["labels"])
            artifacts["output/hybrid_texture.png"] = composed["image"]
        if material_info:
            artifacts["output/material.png"] = _decode(material_info["image_base64"])
            artifacts["report/material.json"] = {k: v for k, v in material_info.items()
                                                 if not k.endswith("base64")}
            for index, data in enumerate(material_info.get("rejected_images_base64", [])):
                artifacts[f"output/rejected_material_{index}.png"] = _decode(data)
        tags = {**context["tags"], "engine": "stable_diffusion_hybrid", "status": job["status"]}
        job["tracking"] = self.tracker(f"hybrid-{job['style']}-{spec['wall_id']}", params, metrics, artifacts,
                                       tags=tags, status="FINISHED" if job["status"] == "succeeded" else "FAILED")
