"""CPU contracts for the geometry-locked Stable Diffusion hybrid facade."""
import base64
import io
import math
import time

import numpy as np
import pytest
from PIL import Image

from src.bridge.facade_bridge import FacadeBridge
from src.bridge.facade_hybrid import (
    FacadeGeometryError, compose_hybrid_facade, geometry_hash, opening_masks, pix2pix_geometry_report,
    texture_size, validate_composition, wall_spec,
)
from src.models.material_diffusion import (
    course_analysis, evaluate_material, load_config, low_frequency_contrast, material_key, seam_ratio,
)

CONFIG = load_config()


def wall(length=8.0, openings=(), level=None, wall_id="w_front"):
    w = {"id": wall_id, "is_exterior": True, "height_m": 2.8, "geometry_fixed": True,
         "polygon_meters": [[0, 0], [length, 0], [length, .25], [0, .25]], "openings": list(openings)}
    if level:
        w["level_id"] = level
    return w


def op(kind, along, width=1.2, height=None, sill=None, oid=None):
    d = {"type": kind, "position_along_wall_m": along, "width_m": width, "id": oid or f"{kind}_{along}"}
    if height is not None:
        d["height_m"] = height
    if sill is not None:
        d["sill_m"] = sill
    return d


def brick_swatch(size=512, courses=16, tilt_px=0):
    """Periodic synthetic brick: exact integer number of courses per tile."""
    y, x = np.mgrid[0:size, 0:size].astype(float)
    course_h = size / courses
    row = np.floor((y + x * tilt_px / size) / course_h)
    mortar_y = ((y + x * tilt_px / size) % course_h) < 3
    brick_w = size / 8
    mortar_x = ((x + (row % 2) * brick_w / 2) % brick_w) < 3
    rng = np.random.default_rng(0)
    base = np.where(mortar_y | mortar_x, 190, 120 + (row % 3) * 6)
    noise = rng.normal(0, 6, (size, size))
    img = np.clip(np.stack([base + 40 + noise, base - 20 + noise, base - 40 + noise], -1), 0, 255)
    return Image.fromarray(img.astype(np.uint8))


def compose(spec, material=None, tile_m=1.2):
    return compose_hybrid_facade(spec, material or brick_swatch(), tile_m, CONFIG["composition"], offset_seed=3)


# ---------------------------------------------------------------- geometry
def test_masks_follow_exact_opening_meters_and_bottom_origin_uvs():
    spec = wall_spec(wall(8, [op("door", 4.0, 1.0, 2.1, 0.0), op("window", 1.5, 1.2, 1.2, 0.9)]))
    result = compose(spec)
    report = validate_composition(spec, result)
    assert report["passed"], report["failures"]
    width, height = result["size"]
    assert result["uv_bounds"] == {**result["uv_bounds"], "u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0}
    door = next(o for o in result["openings"] if o["type"] == "door")
    # Door touches the ground: bottom-origin v0 == 0, and its mask reaches the last image row.
    assert door["uv_rect"][1] == 0.0
    labels = result["labels"]
    index = [o["type"] for o in spec["openings"]].index("door") + 1
    assert labels[height - 1, round(4.0 * width / 8)] == index
    window = next(o for o in result["openings"] if o["type"] == "window")
    assert window["uv_rect"] == pytest.approx([0.9 / 8, 0.9 / 2.8, 2.1 / 8, 2.1 / 2.8])
    assert report["max_mask_error_m"] <= report["mask_tolerance_m"]


def test_density_is_isotropic_so_materials_are_never_stretched():
    for length in (2.0, 7.3, 15.9, 30.0):
        spec = wall_spec(wall(length))
        w, h, density = texture_size(spec, 256, 4096)
        assert max(w, h) <= 4096
        assert abs(w / spec["wall_length_m"] - h / spec["wall_height_m"]) < 1.0 / min(spec["wall_length_m"], 2.8) + 1e-9
        assert density <= 256


def test_wall_without_openings_is_pure_material():
    spec = wall_spec(wall(5))
    result = compose(spec)
    assert not result["mask"].any() and result["openings"] == []
    assert np.array_equal(np.asarray(result["image"]), result["wall_material"])
    assert validate_composition(spec, result)["passed"]


def test_multi_floor_repeats_windows_but_keeps_door_on_ground_floor():
    spec = wall_spec(wall(9, [op("door", 2, 1.0, 2.1, 0), op("window", 6, 1.2)]), num_floors=3)
    kinds = [(o["type"], o["floor_index"]) for o in spec["openings"]]
    assert kinds.count(("door", 0)) == 1 and ("door", 1) not in kinds
    assert [k for k in kinds if k[0] == "window"] == [("window", 0), ("window", 1), ("window", 2)]
    assert spec["wall_height_m"] == pytest.approx(8.4)
    assert validate_composition(spec, compose(spec))["passed"]


def test_opening_touching_edge_is_kept_and_reported_as_zero_clearance():
    spec = wall_spec(wall(6, [op("window", 0.6, 1.2)]))
    assert spec["openings"][0]["left_m"] == pytest.approx(0)
    report = validate_composition(spec, compose(spec))
    assert report["passed"] and report["min_edge_clearance_m"] == pytest.approx(0)


@pytest.mark.parametrize("openings,needle", [
    ([op("window", 0.3, 1.2)], "outside wall"),
    ([op("window", 2, 1.4), op("window", 2.8, 1.4)], "overlaps"),
    ([op("window", 2, 1.0, height=2.5, sill=0.9)], "outside 0.."),
])
def test_invalid_geometry_is_refused_instead_of_clipped(openings, needle):
    with pytest.raises(FacadeGeometryError) as error:
        wall_spec(wall(6, openings))
    assert any(needle in issue for issue in error.value.issues)


def test_shared_templates_give_both_engines_the_same_openings():
    w = wall(10, [op("door", 5, 1.0, 2.1, 0), op("window", 2, 1.2), op("window", 8, 1.2)])
    spec = wall_spec(w)
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = None
    _, _, _, pix_openings = bridge._build_facade_sketch(w, 1, None, True, 256)
    report = pix2pix_geometry_report(spec, pix_openings, 256)
    assert report["dropped"] == [] and report["unexpected"] == []
    # The 256 px Pix2Pix raster quantises at ~4.3 cm per pixel on a 10 m wall.
    assert report["max_error_m"] < 2 * 10 / 232 + 1e-6


def test_geometry_hash_changes_with_any_opening_or_size_change():
    base = wall_spec(wall(8, [op("window", 2, 1.2)]))
    assert geometry_hash(base) == geometry_hash(wall_spec(wall(8, [op("window", 2, 1.2)])))
    assert geometry_hash(base) != geometry_hash(wall_spec(wall(8, [op("window", 2.01, 1.2)])))
    assert geometry_hash(base) != geometry_hash(wall_spec(wall(8.1, [op("window", 2, 1.2)])))
    assert geometry_hash(base) != geometry_hash(wall_spec(wall(8, [op("window", 2, 1.2)]), num_floors=2))


def test_validation_detects_leaks_outside_masks():
    spec = wall_spec(wall(6, [op("window", 3, 1.2)]))
    result = compose(spec)
    tampered = np.asarray(result["image"]).copy()
    tampered[5, 5] = [0, 255, 0]
    result["image"] = Image.fromarray(tampered)
    report = validate_composition(spec, result)
    assert not report["passed"] and any("outside opening masks" in f for f in report["failures"])


def test_textures_are_isolated_per_wall():
    a = compose(wall_spec(wall(6, [op("window", 3, 1.2)], wall_id="a")))
    b = compose(wall_spec(wall(4, [], wall_id="b")))
    assert a["size"] != b["size"] and a["mask"].any() and not b["mask"].any()


# ---------------------------------------------------------------- material analysis
def test_periodic_swatch_is_seamless_and_cut_swatch_is_not():
    periodic = brick_swatch()
    seams = seam_ratio(periodic)
    assert max(seams.values()) < CONFIG["quality_gates"]["max_seam_ratio"]
    # A horizontal light falloff makes the right edge much brighter than the left edge.
    gradient = np.asarray(periodic).astype(float) * np.linspace(0.6, 1.2, 512)[None, :, None]
    lit = Image.fromarray(np.clip(gradient, 0, 255).astype(np.uint8))
    assert seam_ratio(lit)["x"] > CONFIG["quality_gates"]["max_seam_ratio"]
    # A gentle falloff can hide behind mortar edges in the seam ratio, but not in the
    # low-frequency contrast gate.
    gentle = np.asarray(periodic).astype(float) * np.linspace(0.85, 1.15, 512)[None, :, None]
    assert low_frequency_contrast(Image.fromarray(np.clip(gentle, 0, 255).astype(np.uint8))) > low_frequency_contrast(periodic)


def test_course_period_gives_physical_scale_and_detects_tilt():
    analysis = course_analysis(brick_swatch(courses=16), 6, 40)
    assert analysis["courses"] == 16 and abs(analysis["tilt_deg"]) < 1
    entry = CONFIG["styles"]["Traditional Brick"]
    verdict = evaluate_material(brick_swatch(courses=16), entry, CONFIG["quality_gates"])
    assert verdict["scale"]["measured"] and verdict["scale"]["tile_m"] == pytest.approx(16 * 0.075)
    tilted = evaluate_material(brick_swatch(courses=16, tilt_px=64), entry, CONFIG["quality_gates"])
    assert not tilted["passed"] and any("tilted" in f for f in tilted["failures"])


def test_window_like_dark_rectangle_raises_low_frequency_contrast():
    plain = brick_swatch()
    arr = np.asarray(plain).copy()
    arr[150:350, 180:330] = 25
    assert low_frequency_contrast(Image.fromarray(arr)) > CONFIG["quality_gates"]["max_low_frequency_contrast"]
    assert low_frequency_contrast(plain) < CONFIG["quality_gates"]["max_low_frequency_contrast"]


def test_material_cache_key_ignores_walls_but_tracks_model_prompt_and_seed():
    a = material_key(CONFIG, "Traditional Brick", 42)
    assert a == material_key(CONFIG, "Traditional Brick", 42)
    assert a != material_key(CONFIG, "Traditional Brick", 43)
    assert a != material_key(CONFIG, "Mediterranean Stone", 42)
    with pytest.raises(ValueError):
        material_key(CONFIG, "Gothic", 42)


def test_config_pins_model_and_prompts_fit_clip_budget_by_words():
    model = CONFIG["model"]
    assert len(model["revision"]) == 40 and model["license"] == "creativeml-openrail-m"
    assert all(len(spec["sha256"]) == 64 for spec in model["weights"].values())
    for entry in CONFIG["styles"].values():
        assert len(entry["prompt"].split()) < 60  # exact token count is checked by the worker
    assert len(CONFIG["negative_prompt"].split()) < 70


# ---------------------------------------------------------------- jobs (no GPU)
def png_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class FakeWorker:
    url = "http://fake"

    def __init__(self, outcome):
        self.outcome, self.polls, self.cancelled = outcome, 0, []

    def submit(self, style, seed, use_cache=True):
        if self.outcome == "unreachable":
            import httpx
            raise httpx.ConnectError("connection refused")
        return {"job_id": "remote", "status": "queued"}

    def status(self, job_id):
        self.polls += 1
        if self.polls < 2:
            return {"job_id": job_id, "status": "running", "phase": "generating", "step": 5}
        if self.outcome == "oom":
            return {"job_id": job_id, "status": "failed", "error_type": "gpu_out_of_memory",
                    "error": "CUDA out of memory"}
        swatch = brick_swatch()
        quality = evaluate_material(swatch, CONFIG["styles"]["Traditional Brick"], CONFIG["quality_gates"])
        return {"job_id": job_id, "status": "succeeded", "cache_hit": False, "gpu_wait_s": 0.1, "result": {
            "image_base64": png_url(swatch), "quality": quality, "model": CONFIG["model"]["repo"],
            "revision": CONFIG["model"]["revision"], "seed": 42, "prompt": "p", "negative_prompt": "n",
            "material_key": "k", "timings": {"generation_s": 2.0}, "device": "fake"}}

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def health(self):
        return {"status": "ready"}


def run_job(outcome, w=None):
    from backend.app.facade_hybrid_jobs import HybridFacadeJobs
    tracked = []
    jobs = HybridFacadeJobs(worker=FakeWorker(outcome), tracker=lambda *a, **k: tracked.append(a) or {"status": "logged"}, poll_s=0.01)
    submitted = jobs.submit(w or wall(8, [op("door", 4, 1.0, 2.1, 0), op("window", 1.5, 1.2)]), "Traditional Brick")
    for _ in range(500):
        job = jobs.get(submitted["job_id"])
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.01)
    return job, jobs, tracked


def test_successful_job_returns_geometry_locked_texture():
    job, _, _ = run_job("ok")
    assert job["status"] == "succeeded", job.get("error")
    result = job["result"]
    assert result["engine"] == "stable_diffusion_hybrid" and result["validation"]["passed"]
    assert result["geometry_hash"] == job["geometry_hash"]
    assert {o["type"] for o in result["openings"]} == {"door", "window"}
    assert result["model_info"]["pbr_maps"].startswith("none")


@pytest.mark.parametrize("outcome,error_type", [("oom", "gpu_out_of_memory"), ("unreachable", "worker_unavailable")])
def test_gpu_or_worker_failure_is_reported_without_pix2pix_substitute(outcome, error_type):
    job, _, _ = run_job(outcome)
    assert job["status"] == "failed" and job["error_type"] == error_type
    assert "result" not in job  # no texture from any other engine


def test_tracking_failure_does_not_lose_a_successful_generation():
    from backend.app.facade_hybrid_jobs import HybridFacadeJobs

    def broken_tracker(*args, **kwargs):
        from src.models.facade_tracking import log_run
        return log_run(*args, **kwargs, force=True, tracking_uri="http://127.0.0.1:9")

    jobs = HybridFacadeJobs(worker=FakeWorker("ok"), tracker=broken_tracker, poll_s=0.01)
    submitted = jobs.submit(wall(6, [op("window", 3, 1.2)]), "Traditional Brick")
    for _ in range(1000):
        job = jobs.get(submitted["job_id"])
        if job["status"] == "succeeded" and job["tracking"]["status"] != "pending":
            break
        time.sleep(0.01)
    assert job["status"] == "succeeded" and job["result"]["texture_base64"].startswith("data:image/png")
    assert job["tracking"]["status"] == "failed" and job["tracking"]["error"]


def test_invalid_geometry_is_rejected_before_any_gpu_work():
    from backend.app.facade_hybrid_jobs import HybridFacadeJobs
    worker = FakeWorker("ok")
    jobs = HybridFacadeJobs(worker=worker, tracker=lambda *a, **k: {"status": "disabled"})
    with pytest.raises(FacadeGeometryError):
        jobs.submit(wall(4, [op("window", 3.9, 1.2)]), "Traditional Brick")
    assert worker.polls == 0


def test_api_reports_unconfigured_worker_as_503_not_fallback(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app.main as main
    from backend.app.facade_hybrid_jobs import HybridFacadeJobs, WorkerClient
    monkeypatch.setattr(main, "_hybrid_jobs", HybridFacadeJobs(worker=WorkerClient(None), tracker=lambda *a, **k: {}))
    client = TestClient(main.app)
    response = client.post("/facade/hybrid/jobs", json={"wall": wall(6, [op("window", 3, 1.2)]), "style": "Traditional Brick"})
    assert response.status_code == 503 and "not configured" in response.json()["detail"]
    bad = client.post("/facade/hybrid/jobs", json={"wall": wall(6, [op("window", 0.2, 1.2)]), "style": "Traditional Brick"})
    assert bad.status_code == 422 and bad.json()["detail"]["error"] == "invalid_geometry"


def test_tripsr_and_diffusion_share_one_exclusive_gpu_lock(tmp_path, monkeypatch):
    """A material job waits while TripoSR holds the shared lock (and vice versa)."""
    import threading
    lock_path = str(tmp_path / "gpu.lock")
    monkeypatch.setenv("ARCHIGEN_GPU_LOCK", lock_path)
    from src.furniture.quality_engine import shared_gpu_section
    from backend.app.facade_diffusion_worker import GpuLock
    events = []
    furniture = shared_gpu_section()
    furniture.acquire()
    events.append("triposr_start")

    def diffusion():
        with GpuLock(lock_path).hold(timeout_s=5):
            events.append("diffusion_start")

    thread = threading.Thread(target=diffusion)
    thread.start()
    time.sleep(0.3)
    assert events == ["triposr_start"]  # still waiting
    events.append("triposr_end")
    furniture.release()
    thread.join(5)
    assert events == ["triposr_start", "triposr_end", "diffusion_start"]
    with pytest.raises(TimeoutError):
        with GpuLock(lock_path).hold(timeout_s=5):
            with GpuLock(lock_path).hold(timeout_s=0.2):
                pass


def test_worker_http_api_queues_generates_caches_and_reports_failures(tmp_path, monkeypatch):
    """Worker HTTP layer with a fake GPU pipeline: success, cache hit, OOM failure."""
    from fastapi.testclient import TestClient
    import backend.app.facade_diffusion_worker as worker
    monkeypatch.setattr(worker, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(worker, "verify_weights", lambda config, full=False: {"ok": True})

    class Tokenizer:
        model_max_length = 77
        def __call__(self, text):
            return {"input_ids": text.split()}

    class FakePipeline:
        device_name, total_vram_mib, load_seconds, circular_layers = "fake", 6000, 0.1, 1
        def __init__(self, config):
            self.pipe = type("P", (), {"tokenizer": Tokenizer()})()
            self.fail = False
        def run(self, prompt, negative, seed, on_step, deadline):
            if self.fail:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            on_step(25)
            return brick_swatch(), {"generation_s": 0.01}

    service = worker.MaterialService(CONFIG, pipeline_factory=FakePipeline)
    service.start()
    client = TestClient(worker.create_app(service))

    def wait(job_id):
        for _ in range(400):
            job = client.get(f"/materials/{job_id}").json()
            if job["status"] in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(0.01)
        raise AssertionError("job did not finish")

    first = wait(client.post("/materials", json={"style": "Traditional Brick", "seed": 5}).json()["job_id"])
    assert first["status"] == "succeeded" and first["result"]["quality"]["passed"]
    again = client.post("/materials", json={"style": "Traditional Brick", "seed": 5}).json()
    assert again["status"] == "succeeded" and again["cache_hit"]
    assert client.post("/materials", json={"style": "Gothic"}).status_code == 422
    service.pipeline.fail = True
    failed = wait(client.post("/materials", json={"style": "Traditional Brick", "seed": 6}).json()["job_id"])
    assert failed["status"] == "failed" and failed["error_type"] == "gpu_out_of_memory"
    assert "result" not in failed
    assert client.get("/health").json()["status"] == "ready"
