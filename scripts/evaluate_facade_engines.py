"""Reproducible Pix2Pix vs "Stable Diffusion — hybrid" facade comparison.

Runs against the *deployed* API (Docker: http://127.0.0.1:8000) so latency and memory
are those of the real stack. Wall payloads are built with the frontend's own opening
fitting (scripts/facade_payloads.mjs), i.e. exactly what the studio sends.

Splits
  tuning  sample-house ground storey: used while choosing prompts and gate thresholds.
  final   every other case; run once with the frozen configs/facade_hybrid.json.
  seeds   material-only robustness: several seeds per style (failure rate, latency,
          colour consistency). Not part of the paired wall comparison.

There is no ground-truth facade photo for these walls, so no paired image metric
(LPIPS, KID, SSIM...) is computed. Sharpness/resolution are descriptive only.

    python scripts/evaluate_facade_engines.py --split tuning
    python scripts/evaluate_facade_engines.py --split final --seeds 6
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bridge.facade_hybrid import FacadeGeometryError, pix2pix_geometry_report, wall_spec  # noqa: E402
from src.models.facade_tracking import log_run  # noqa: E402
from src.models.material_diffusion import config_sha256, load_config  # noqa: E402

STYLES = ["Modern Minimalist", "Mediterranean Stone", "Traditional Brick", "Lush Vegetation"]
PLANS = [  # (case prefix, path, scale m/px, storeys)
    ("plan_sample", "frontend/public/sample_floorplan.png", 0.03, 1),
    ("plan_cad", "data/external_test_plans/01_clean_bw_cad.png", 0.03, 2),
    ("plan_loft", "data/external_test_plans/07_open_concept_loft.png", 0.03, 1),
    ("plan_narrow", "data/external_test_plans/08_narrow_railroad_layout.png", 0.03, 1),
]


def strip(polygon_len):
    return [[0, 0], [polygon_len, 0], [polygon_len, .2], [0, .2]]


SYNTHETIC = [  # explicit edge cases, labelled synthetic
    ("syn_blank_long", {"length": 14.0, "openings": [], "floors": 1}),
    ("syn_edge_windows", {"length": 6.0, "openings": [("window", 0.6, 1.2), ("window", 5.4, 1.2), ("door", 3.0, 0.9)], "floors": 1}),
    ("syn_three_storey", {"length": 9.0, "openings": [("door", 4.5, 1.0), ("window", 1.6, 1.2), ("window", 7.4, 1.2)], "floors": 3}),
    ("syn_short", {"length": 2.4, "openings": [("window", 1.2, 0.9)], "floors": 1}),
]


def b64_image(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


def payloads_from(kind, **extra):
    process = subprocess.run(["node", str(ROOT / "scripts/facade_payloads.mjs")], input=json.dumps({"kind": kind, **extra}),
                             capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(process.stdout)


def categorise(payload):
    kinds = [o["type"] for o in payload["wall"]["openings"]]
    if not kinds:
        return "no_openings"
    if "door" in kinds and "window" in kinds:
        return "door_and_windows"
    if kinds.count("window") > 1:
        return "multiple_windows"
    return "single_" + kinds[0]


def build_cases(api):
    cases = []
    for p in payloads_from("sample", scale=0.03):
        cases.append({"case": f"sample_house/{p['wall_id']}", "plan": "sample_house (traced reference)", "payload": p,
                      "split": "tuning" if p["level_id"] == "ground" else "final", "synthetic": False})
    for prefix, path, scale, floors in PLANS:
        with open(ROOT / path, "rb") as stream:
            response = requests.post(f"{api}/predict/floorplan", files={"file": (Path(path).name, stream, "image/png")},
                                     data={"scale_override": str(scale), "num_floors": str(floors)}, timeout=180)
        response.raise_for_status()
        payloads = payloads_from("analysis", response=response.json(), scale=scale, num_floors=floors)
        chosen, seen = [], set()
        for p in sorted(payloads, key=lambda q: -len(q["wall"]["openings"])):
            category = categorise(p)
            if category in seen:
                continue
            seen.add(category)
            chosen.append(p)
            if len(chosen) == 3:
                break
        for p in chosen:
            cases.append({"case": f"{prefix}/{p['wall_id']}", "plan": path, "payload": p, "split": "final", "synthetic": False})
    for name, spec in SYNTHETIC:
        openings = [{"id": f"{kind}_{i}", "type": kind, "position_along_wall_m": along, "width_m": width,
                     **({"height_m": 2.1, "sill_m": 0.0} if kind == "door" else {})}
                    for i, (kind, along, width) in enumerate(spec["openings"])]
        wall = {"id": name, "is_exterior": True, "height_m": 2.8, "polygon_meters": strip(spec["length"]), "openings": openings}
        cases.append({"case": f"synthetic/{name}", "plan": "synthetic", "payload": {"wall": wall, "wall_id": name, "num_floors": spec["floors"]},
                      "split": "final", "synthetic": True})
    for index, case in enumerate(cases):
        case["category"] = categorise(case["payload"])
        case["index"] = index
    return cases


def run_pix2pix(api, payload, style, tags):
    started = time.perf_counter()
    response = requests.post(f"{api}/facade/generate", json={**payload, "style": style, "tracking_tags": tags}, timeout=120)
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        return {"status": "failed", "error": response.text[:500], "client_s": elapsed}
    return {"status": "succeeded", "data": response.json(), "client_s": elapsed}


def run_hybrid(api, payload, style, seed, tags, use_cache=False):
    started = time.perf_counter()
    response = requests.post(f"{api}/facade/hybrid/jobs", json={**payload, "style": style, "seed": seed,
                                                                "use_cache": use_cache, "tracking_tags": tags}, timeout=60)
    if response.status_code != 202:
        detail = response.json().get("detail") if response.headers.get("content-type", "").startswith("application/json") else response.text
        return {"status": "rejected", "http": response.status_code, "error": detail, "client_s": time.perf_counter() - started}
    job = response.json()
    while job["status"] not in ("succeeded", "failed", "cancelled"):
        time.sleep(0.5)
        job = requests.get(f"{api}/facade/hybrid/jobs/{job['job_id']}", timeout=30).json()
    elapsed = time.perf_counter() - started
    for _ in range(40):  # wait for the MLflow status of this job
        if job.get("tracking", {}).get("status") != "pending":
            break
        time.sleep(0.25)
        job = requests.get(f"{api}/facade/hybrid/jobs/{job['job_id']}", timeout=30).json()
    if job["status"] != "succeeded":
        return {"status": job["status"], "error": job.get("error"), "error_type": job.get("error_type"), "client_s": elapsed, "job": job}
    return {"status": "succeeded", "data": job["result"], "client_s": elapsed, "tracking": job.get("tracking")}


def wall_crop(data):
    image = b64_image(data["texture_base64"])
    b = data["uv_bounds"]
    w, h = image.size
    return image.crop((round(b["u_min"] * w), round((1 - b["v_max"]) * h), round(b["u_max"] * w), round((1 - b["v_min"]) * h)))


def comparison_panel(pix, hyb, title, display_width=1200):
    """Same framing (wall only), same display size, same bilinear resampling for both."""
    crops = [wall_crop(pix), wall_crop(hyb)]
    aspect = pix["wall_length_m"] / pix["wall_height_m"]
    dw = display_width if aspect >= 1 else round(display_width * 0.5)
    dh = max(60, round(dw / aspect))
    labels = [f"Pix2Pix · native {pix['model_info'].get('native_texture_px')} px, wall {crops[0].size[0]}x{crops[0].size[1]} px",
              f"SD hybrid · material {hyb['model_info']['native_material_px']}px, wall {crops[1].size[0]}x{crops[1].size[1]} px"]
    canvas = Image.new("RGB", (dw, 30 + 2 * (dh + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill="black")
    for i, (crop, label) in enumerate(zip(crops, labels)):
        y = 30 + i * (dh + 26)
        draw.text((6, y + 4), label, fill="black")
        canvas.paste(crop.resize((dw, dh), Image.Resampling.BILINEAR), (0, y + 22))
    return canvas


def zoom_panel(pix, hyb, spec, size_m=1.0, out_px=360):
    """The same 1 m x 1 m patch of the wall, native pixels shown with nearest-neighbour."""
    target = spec["openings"][0] if spec["openings"] else None
    cx = min(max(target["left_m"], size_m / 2), spec["wall_length_m"] - size_m / 2) if target else spec["wall_length_m"] / 2
    cy = min(max(target["top_m"], size_m / 2), spec["wall_height_m"] - size_m / 2) if target else spec["wall_height_m"] / 2
    patches = []
    for data in (pix, hyb):
        crop = wall_crop(data)
        sx, sy = crop.size[0] / spec["wall_length_m"], crop.size[1] / spec["wall_height_m"]
        box = (round((cx - size_m / 2) * sx), round((spec["wall_height_m"] - cy - size_m / 2) * sy),
               round((cx + size_m / 2) * sx), round((spec["wall_height_m"] - cy + size_m / 2) * sy))
        patch = crop.crop(box)
        patches.append((patch.resize((out_px, out_px), Image.Resampling.NEAREST), patch.size))
    canvas = Image.new("RGB", (2 * out_px + 10, out_px + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (patch, native) in enumerate(patches):
        draw.text((i * (out_px + 10) + 4, 4), f"{['Pix2Pix', 'SD hybrid'][i]} · same 1 m patch · native {native[0]}x{native[1]} px", fill="black")
        canvas.paste(patch, (i * (out_px + 10), 24))
    return canvas


def summary_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], "min": ordered[0], "max": ordered[-1]}


def lab_mean(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    return [float(v) for v in arr.reshape(-1, 3).mean(0) * 255]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--mlflow", default="http://127.0.0.1:5000")
    parser.add_argument("--split", choices=["tuning", "final"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=0, help="material robustness seeds per style (final split)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.output or ROOT / "outputs" / "facade_engine_comparison" / f"{args.split}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    config = load_config()
    status = requests.get(f"{args.api}/facade/hybrid/status", timeout=10).json()
    if not status["ready"]:
        raise SystemExit(f"Stable Diffusion worker not ready: {status['worker']}")
    cases = [c for c in build_cases(args.api) if c["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]
    session = {"split": args.split, "stamp": stamp, "config_sha256": config_sha256(), "seed": args.seed,
               "worker": status["worker"], "cases": len(cases)}
    tags_base = {"run_kind": "evaluation", "evaluation_split": args.split, "evaluation_session": stamp}
    print(json.dumps(session, indent=2), flush=True)

    # ---- warm-up (excluded from measurements, still logged and tagged)
    warm_payload = cases[0]["payload"]
    warmups = []
    for i in range(args.warmup):
        warmups.append({"pix2pix": run_pix2pix(args.api, warm_payload, "Traditional Brick", {**tags_base, "warmup": "true"})["client_s"],
                        "hybrid": run_hybrid(args.api, warm_payload, "Traditional Brick", 10_000 + i, {**tags_base, "warmup": "true"})["client_s"]})
    print("warm-up seconds", warmups, flush=True)

    # ---- paired wall comparison
    rows = []
    for case in cases:
        styles = STYLES if (args.split == "tuning" and case["index"] % 3 == 0) else [STYLES[case["index"] % 4]]
        for style in styles:
            slug = f"{case['case'].replace('/', '__')}__{style.split()[1].lower() if ' ' in style else style.lower()}"
            directory = out / "cases" / slug
            directory.mkdir(parents=True)
            payload = case["payload"]
            tags = {**tags_base, "case": case["case"], "category": case["category"], "warmup": "false"}
            pix = run_pix2pix(args.api, payload, style, tags)
            hyb = run_hybrid(args.api, payload, style, args.seed, tags)
            row = {"case": case["case"], "plan": case["plan"], "category": case["category"], "synthetic": case["synthetic"],
                   "style": style, "num_floors": payload["num_floors"], "pix2pix_status": pix["status"], "hybrid_status": hyb["status"],
                   "pix2pix_client_s": pix["client_s"], "hybrid_client_s": hyb["client_s"]}
            (directory / "wall.json").write_text(json.dumps(payload, indent=2))
            try:
                spec = wall_spec({**payload["wall"], "geometry_fixed": True},
                                 num_floors=1 if payload["wall"].get("level_id") else payload["num_floors"])
                row.update(wall_length_m=spec["wall_length_m"], wall_height_m=spec["wall_height_m"], openings=len(spec["openings"]),
                           min_edge_clearance_m=min((o["edge_clearance_m"] for o in spec["openings"]), default=None))
            except FacadeGeometryError as exc:
                spec = None
                row.update(geometry_invalid=exc.issues)
            if pix["status"] == "succeeded":
                d = pix["data"]
                wall_px = wall_crop(d).size
                row.update(pix2pix_server_ms=d.get("generation_ms"), pix2pix_device=d.get("device"),
                           pix2pix_wall_px=list(wall_px), pix2pix_px_per_m=wall_px[0] / d["wall_length_m"])
                b64_image(d["texture_base64"]).save(directory / "pix2pix_texture.png")
                b64_image(d["sketch_base64"]).save(directory / "pix2pix_cmp_sketch.png")
                if spec:
                    row["pix2pix_geometry"] = pix2pix_geometry_report(spec, d["openings"], d["model_info"].get("native_texture_px", [256])[0])
            if hyb["status"] == "succeeded":
                d = hyb["data"]
                info = d["model_info"]
                row.update(hybrid_validation=d["validation"], hybrid_timings=d["timings"], hybrid_px_per_m=info["density_px_per_m"],
                           hybrid_texture_px=info["texture_size_px"], hybrid_tile_m=info["material_tile_m"],
                           hybrid_scale_method=info["scale"]["method"], hybrid_courses=info["scale"].get("courses"),
                           hybrid_repeats=info["material_repeats"], hybrid_rejected=info["rejected_attempts"],
                           hybrid_material_metrics=info["material_quality"]["metrics"], hybrid_seed=info["seed"],
                           hybrid_tracking=hyb.get("tracking"), hybrid_openings_equal_spec=spec is not None and len(d["openings"]) == len(spec["openings"]))
                b64_image(d["texture_base64"]).save(directory / "hybrid_texture.png")
                b64_image(d["sketch_base64"]).save(directory / "hybrid_opening_mask.png")
                b64_image(d["material_base64"]).save(directory / "hybrid_material.png")
            else:
                row.update(hybrid_error=hyb.get("error"), hybrid_error_type=hyb.get("error_type"), hybrid_http=hyb.get("http"))
            if pix["status"] == "succeeded" and hyb["status"] == "succeeded":
                comparison_panel(pix["data"], hyb["data"], f"{case['case']} · {style} · {case['category']}").save(directory / "comparison_same_frame.png")
                zoom_panel(pix["data"], hyb["data"], spec).save(directory / "zoom_same_patch.png")
            (directory / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
            artifacts = {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.iterdir() if p.is_file()}
            run = log_run(f"compare-{slug}", {"engine": "pix2pix+stable_diffusion_hybrid", "case": case["case"], "style": style,
                                              "category": case["category"], "split": args.split, "config_sha256": session["config_sha256"],
                                              "seed": args.seed, "plan": case["plan"], "synthetic": case["synthetic"]},
                          {k: v for k, v in {"pix2pix_server_ms": row.get("pix2pix_server_ms"),
                                             "hybrid_total_s": (row.get("hybrid_timings") or {}).get("total_s"),
                                             "hybrid_generation_s": (row.get("hybrid_timings") or {}).get("generation_s"),
                                             "hybrid_composition_s": (row.get("hybrid_timings") or {}).get("composition_s"),
                                             "hybrid_peak_allocated_mib": (row.get("hybrid_timings") or {}).get("peak_allocated_mib"),
                                             "hybrid_peak_device_used_mib": (row.get("hybrid_timings") or {}).get("peak_device_used_mib"),
                                             "pix2pix_px_per_m": row.get("pix2pix_px_per_m"), "hybrid_px_per_m": row.get("hybrid_px_per_m"),
                                             "pix2pix_max_opening_error_m": (row.get("pix2pix_geometry") or {}).get("max_error_m"),
                                             "hybrid_max_mask_error_m": (row.get("hybrid_validation") or {}).get("max_mask_error_m")}.items()
                           if v is not None},
                          {f"comparison/{k}": v for k, v in artifacts.items()},
                          tags={**tags, "engine": "comparison"}, force=True, tracking_uri=args.mlflow)
            row["comparison_run"] = run
            rows.append(row)
            print(f"{slug}: pix2pix={pix['status']} hybrid={hyb['status']} "
                  f"{(row.get('hybrid_timings') or {}).get('total_s', float('nan')):.1f}s mlflow={run['status']}", flush=True)

    # ---- material robustness over seeds (final split only)
    seed_rows = []
    if args.seeds:
        probe = next(c for c in cases if c["category"] == "no_openings")["payload"]
        for style in STYLES:
            for k in range(args.seeds):
                seed = 1000 + k
                result = run_hybrid(args.api, probe, style, seed, {**tags_base, "case": "seed_study", "warmup": "false"})
                entry = {"style": style, "seed": seed, "status": result["status"], "error_type": result.get("error_type"),
                         "client_s": result["client_s"]}
                if result["status"] == "succeeded":
                    info = result["data"]["model_info"]
                    material = b64_image(result["data"]["material_base64"])
                    target = out / "seed_study" / f"{style.split()[-1].lower()}_{seed}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    material.save(target)
                    entry.update(rejected=info["rejected_attempts"], used_seed=info["seed"], tile_m=info["material_tile_m"],
                                 scale_method=info["scale"]["method"], metrics=info["material_quality"]["metrics"],
                                 mean_rgb=lab_mean(material), timings=result["data"]["timings"])
                else:
                    entry["error"] = result.get("error")
                seed_rows.append(entry)
                print(f"seed {style} {seed}: {result['status']}", flush=True)

    report = make_report(session, rows, seed_rows, warmups)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "rows.json").write_text(json.dumps(rows, indent=2, default=str))
    (out / "seed_study.json").write_text(json.dumps(seed_rows, indent=2, default=str))
    build_blind_kit(out, rows)
    summary_artifacts = {"report.json": report, "rows.json": rows, "seed_study.json": seed_rows,
                         "config/facade_hybrid.json": (ROOT / "configs/facade_hybrid.json").read_text(),
                         "blind_review/index.html": (out / "blind_review/index.html").read_text(encoding="utf-8")}
    for path in (out / "seed_study").glob("*.png") if args.seeds else []:
        summary_artifacts[f"seed_study/{path.name}"] = path.read_bytes()
    run = log_run(f"facade-engine-comparison-{args.split}-{stamp}", {"split": args.split, "config_sha256": session["config_sha256"],
                                                                     "cases": len(cases), "seed": args.seed, "seed_study_per_style": args.seeds},
                  flatten_metrics(report), summary_artifacts, tags={**tags_base, "engine": "summary"}, force=True, tracking_uri=args.mlflow)
    report["summary_run"] = run
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("geometry", "latency", "failures")}, indent=2, default=str))
    print(f"Outputs: {out}\nMLflow summary: {run}")


def make_report(session, rows, seed_rows, warmups):
    ok = [r for r in rows if r["hybrid_status"] == "succeeded"]
    timings = [r["hybrid_timings"] for r in ok]
    fresh = [t for t in timings if t.get("generation_s")]
    styles = {}
    for entry in seed_rows:
        styles.setdefault(entry["style"], []).append(entry)
    return {
        "session": session, "warmup_client_s": warmups,
        "geometry": {
            "hybrid_composed": len(ok), "hybrid_validation_passed": sum(r["hybrid_validation"]["passed"] for r in ok),
            "hybrid_max_mask_error_m": max((r["hybrid_validation"]["max_mask_error_m"] for r in ok), default=None),
            "hybrid_rejected_invalid_geometry": [{"case": r["case"], "error": r.get("hybrid_error")} for r in rows if r["hybrid_status"] == "rejected"],
            "pix2pix_max_opening_error_m": summary_stats([r["pix2pix_geometry"]["max_error_m"] for r in rows if r.get("pix2pix_geometry")]),
            "pix2pix_dropped_openings": sum(len(r["pix2pix_geometry"]["dropped"]) for r in rows if r.get("pix2pix_geometry")),
        },
        "resolution_descriptive_only": {
            "pix2pix_px_per_m": summary_stats([r.get("pix2pix_px_per_m") for r in rows]),
            "hybrid_px_per_m": summary_stats([r.get("hybrid_px_per_m") for r in rows]),
        },
        "materials": {
            "seam_ratio_max": summary_stats([max(r["hybrid_material_metrics"]["seam_ratio_x"], r["hybrid_material_metrics"]["seam_ratio_y"]) for r in ok]),
            "low_frequency_contrast": summary_stats([r["hybrid_material_metrics"]["low_frequency_contrast"] for r in ok]),
            "tile_m": summary_stats([r["hybrid_tile_m"] for r in ok]),
            "repeats_along_wall": summary_stats([r["hybrid_repeats"][0] for r in ok]),
            "scale_methods": {m: sum(r["hybrid_scale_method"] == m for r in ok) for m in {r["hybrid_scale_method"] for r in ok}},
        },
        "latency": {
            "pix2pix_server_ms": summary_stats([r.get("pix2pix_server_ms") for r in rows]),
            "pix2pix_client_s": summary_stats([r["pix2pix_client_s"] for r in rows]),
            "hybrid_client_total_s": summary_stats([r["hybrid_client_s"] for r in ok]),
            "hybrid_generation_s": summary_stats([t.get("generation_s") for t in fresh]),
            "hybrid_gpu_load_s": summary_stats([t.get("gpu_load_s") for t in fresh]),
            "hybrid_gpu_unload_s": summary_stats([t.get("gpu_unload_s") for t in fresh]),
            "hybrid_composition_s": summary_stats([t.get("composition_s") for t in timings]),
            "hybrid_validation_s": summary_stats([t.get("validation_s") for t in timings]),
        },
        "memory": {
            "hybrid_peak_allocated_mib": summary_stats([t.get("peak_allocated_mib") for t in fresh]),
            "hybrid_peak_reserved_mib": summary_stats([t.get("peak_reserved_mib") for t in fresh]),
            "hybrid_peak_device_used_mib": summary_stats([t.get("peak_device_used_mib") for t in fresh]),
            "hybrid_idle_device_used_mib": summary_stats([t.get("idle_device_used_mib") for t in fresh]),
            "worker_process_rss_mib": summary_stats([t.get("process_rss_mib") for t in fresh]),
        },
        "failures": {
            "pix2pix_failed": sum(r["pix2pix_status"] != "succeeded" for r in rows),
            "hybrid_failed": [{"case": r["case"], "type": r.get("hybrid_error_type"), "error": r.get("hybrid_error")} for r in rows if r["hybrid_status"] == "failed"],
            "hybrid_rejected_attempts_total": sum(r.get("hybrid_rejected") or 0 for r in ok),
            "seed_study": {style: {"runs": len(v), "succeeded": sum(e["status"] == "succeeded" for e in v),
                                   "rejected_attempts": sum(e.get("rejected") or 0 for e in v),
                                   "measured_scale": sum(e.get("scale_method") == "course_period" for e in v),
                                   "generation_s": summary_stats([(e.get("timings") or {}).get("generation_s") for e in v]),
                                   "mean_rgb_spread": [float(np.std([e["mean_rgb"][c] for e in v if "mean_rgb" in e])) for c in range(3)] if any("mean_rgb" in e for e in v) else None}
                           for style, v in styles.items()},
        },
        "not_measured": ["paired fidelity metrics (no ground-truth photo for these walls)",
                         "statistical superiority (too few walls and one rater at most)"],
    }


def flatten_metrics(report):
    out = {}
    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}_{k}" if prefix else k, v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix[:240]] = value
    walk("", {k: report[k] for k in ("geometry", "resolution_descriptive_only", "materials", "latency", "memory")})
    return out


def build_blind_kit(out, rows):
    """Randomised A/B pages for human raters; the key is stored separately."""
    kit = out / "blind_review"
    kit.mkdir()
    rng = random.Random(20261115)
    pairs, key = [], {}
    for directory in sorted((out / "cases").iterdir()):
        if not (directory / "pix2pix_texture.png").exists() or not (directory / "hybrid_texture.png").exists():
            continue
        metrics = json.loads((directory / "metrics.json").read_text())
        pix = Image.open(directory / "pix2pix_texture.png")
        hyb = Image.open(directory / "hybrid_texture.png")
        # Same framing: crop Pix2Pix to its wall region using the letterbox convention.
        aspect = metrics["wall_length_m"] / metrics["wall_height_m"]
        width = 1000
        height = max(60, round(width / aspect))
        pw, ph = pix.size
        scale = min((pw - 24) / metrics["wall_length_m"], (ph - 24) / metrics["wall_height_m"])
        dw, dh = round(metrics["wall_length_m"] * scale), round(metrics["wall_height_m"] * scale)
        x0, y0 = (pw - dw) // 2, (ph - dh) // 2
        images = {"pix2pix": pix.convert("RGB").crop((x0, y0, x0 + dw, y0 + dh)).resize((width, height), Image.Resampling.BILINEAR),
                  "hybrid": hyb.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)}
        order = ["pix2pix", "hybrid"]
        rng.shuffle(order)
        pair_id = hashlib.sha256(directory.name.encode()).hexdigest()[:10]
        for slot, engine in zip("AB", order):
            images[engine].save(kit / f"{pair_id}_{slot}.png")
        key[pair_id] = {"A": order[0], "B": order[1], "case": directory.name}
        pairs.append(pair_id)
    rng.shuffle(pairs)
    (out / "blind_review_key.json").write_text(json.dumps(key, indent=2))
    criteria = ["Material realism (looks like a real wall surface)", "Detail and clarity at wall scale",
                "Seams, visible repetition, stretching", "Scale plausibility (brick/stone size vs openings)",
                "Matches the requested material style"]
    blocks = "\n".join(f"""<section data-pair="{p}"><h2>Pair {i + 1}</h2><figure><figcaption>A</figcaption><img src="{p}_A.png"></figure>
<figure><figcaption>B</figcaption><img src="{p}_B.png"></figure>
<table>{''.join(f'<tr><td>{c}</td>' + ''.join(f'<td><label><input type="radio" name="{p}_{j}" value="{v}">{v}</label></td>' for v in ['A clearly', 'A slightly', 'Equal', 'B slightly', 'B clearly']) + '</tr>' for j, c in enumerate(criteria))}</table></section>"""
                       for i, p in enumerate(pairs))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Facade blind review</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1080px}}img{{width:100%;border:1px solid #888;image-rendering:auto}}
section{{border-top:2px solid #222;padding:12px 0}}td{{padding:2px 6px;font-size:12px}}</style></head><body>
<h1>Facade blind review</h1>
<p>Each pair shows the same wall with the same openings, cropped to the wall and displayed at the same size with the
same bilinear interpolation. The engine behind A and B is randomised and hidden (key stored separately). The two
engines have different native resolutions, which may be visible; rate what you see. Rate each criterion, then export.</p>
{blocks}
<button onclick="const r={{}};document.querySelectorAll('input:checked').forEach(i=>r[i.name]=i.value);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{{type:'application/json'}}));
a.download='ratings.json';a.click()">Export ratings.json</button></body></html>"""
    (kit / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
