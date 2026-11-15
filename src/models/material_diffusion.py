"""Material-swatch helpers shared by the CUDA worker, the API and the evaluation.

Importing this module never loads a model. Everything here works on plain arrays so
the CPU backend can verify a swatch it receives from the GPU worker.

The checks measure the delivered image. A prompt asking for a "seamless" texture or
for "no windows" proves nothing by itself; these gates are heuristics and do not
prove the absence of architectural objects either, so visual review is still needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "facade_hybrid.json"


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or CONFIG_PATH)
    return _load(str(path), path.stat().st_mtime_ns)


def config_sha256(path: Optional[Path] = None) -> str:
    return hashlib.sha256(Path(path or CONFIG_PATH).read_bytes()).hexdigest()


def style_entry(config: Dict[str, Any], style: str) -> Dict[str, Any]:
    if style not in config["styles"]:
        raise ValueError(f"Unsupported hybrid style {style!r}; choose one of {sorted(config['styles'])}")
    return config["styles"][style]


def material_key(config: Dict[str, Any], style: str, seed: int) -> str:
    """Cache key of a reusable material: depends on model, prompt and sampler, never on a wall."""
    entry = style_entry(config, style)
    payload = {"model": config["model"]["repo"], "revision": config["model"]["revision"],
               "generation": {k: config["generation"][k] for k in ("resolution", "steps", "guidance_scale", "scheduler", "precision", "tiling")},
               "prompt": entry["prompt"], "negative": config["negative_prompt"], "seed": int(seed)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def luminance(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def seam_ratio(image: Image.Image) -> Dict[str, float]:
    """Wrap-around step relative to the strongest ordinary neighbouring-line steps.

    The step between the last and first column (row) is compared with the 99th
    percentile of the steps between adjacent interior columns (rows). A periodic
    tile makes the wrap just another line (ratio about 1 or less, even when the wrap
    falls on a mortar joint); a cut or a light falloff makes it an outlier.
    """
    y = luminance(image)
    out = {}
    for axis, name in ((1, "x"), (0, "y")):
        lines = np.mean(np.abs(np.diff(y, axis=axis)), axis=1 - axis)
        wrap = np.mean(np.abs(np.take(y, 0, axis=axis) - np.take(y, -1, axis=axis)))
        out[name] = float(wrap / max(np.percentile(lines, 99), 1e-6))
    return out


def _periodic_blur(y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass with wrap-around (the swatch is meant to be periodic)."""
    h, w = y.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    kernel = np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(y) * kernel))


def low_frequency_contrast(image: Image.Image) -> float:
    """Std of a heavily blurred luminance relative to its mean.

    Large blotches, directional light falloff or a dark window-like rectangle all show
    up here; they are also what makes repetition obvious once the swatch is tiled.
    """
    y = luminance(image)
    blurred = _periodic_blur(y, sigma=y.shape[0] / 12)
    return float(blurred.std() / max(blurred.mean(), 1e-6))


def detail_std(image: Image.Image) -> float:
    """High-frequency energy; a near-constant output is rejected (not a quality score)."""
    y = luminance(image)
    return float((y - _periodic_blur(y, sigma=2.0)).std())


def course_analysis(image: Image.Image, min_courses: int, max_courses: int) -> Dict[str, Any]:
    """Dominant number of horizontal courses per tile and their tilt.

    In a periodic swatch courses are an integer 2D frequency (kx, ky) of the
    vertical-gradient image: ky courses per tile height, tilted by atan(kx / ky).
    """
    y = luminance(image)
    gy = np.abs(np.diff(np.vstack([y, y[:1]]), axis=0))
    spectrum = np.abs(np.fft.fft2(gy - gy.mean()))
    h, w = gy.shape
    lo, hi = max(1, min_courses), min(h // 2 - 1, max_courses)
    if hi < lo:
        return {"courses": None, "prominence": 0.0, "tilt_deg": None}
    kxs = np.arange(-8, 9)
    band = np.array([[spectrum[ky, kx % w] for kx in kxs] for ky in range(lo, hi + 1)])
    iy, ix = np.unravel_index(np.argmax(band), band.shape)
    ky, kx = lo + iy, int(kxs[ix])
    prominence = float(band[iy, ix] / max(np.median(band), 1e-6))
    return {"courses": int(ky), "prominence": prominence, "tilt_deg": float(math.degrees(math.atan2(kx, ky)))}


def physical_tile_m(image: Image.Image, scale: Dict[str, Any]) -> Dict[str, Any]:
    """Physical size of the square swatch, measured when the material allows it."""
    if scale["method"] == "assumed":
        return {"tile_m": float(scale["tile_m"]), "method": "assumed", "measured": False}
    analysis = course_analysis(image, scale["min_courses"], scale["max_courses"])
    if analysis["courses"] and analysis["prominence"] >= 3.0:
        return {"tile_m": analysis["courses"] * scale["course_m"], "method": "course_period", "measured": True,
                "course_m": scale["course_m"], **analysis}
    return {"tile_m": float(scale["default_tile_m"]), "method": "default_after_weak_periodicity",
            "measured": False, **analysis}


def evaluate_material(image: Image.Image, style_cfg: Dict[str, Any], gates: Dict[str, Any]) -> Dict[str, Any]:
    seams = seam_ratio(image)
    metrics = {"seam_ratio_x": seams["x"], "seam_ratio_y": seams["y"],
               "low_frequency_contrast": low_frequency_contrast(image), "detail_std": detail_std(image)}
    scale = physical_tile_m(image, style_cfg["scale"])
    failures = []
    if max(seams.values()) > gates["max_seam_ratio"]:
        failures.append(f"visible wrap seam (ratio {max(seams.values()):.2f} > {gates['max_seam_ratio']})")
    if metrics["low_frequency_contrast"] > gates["max_low_frequency_contrast"]:
        failures.append(f"large-scale shading or object (contrast {metrics['low_frequency_contrast']:.3f} > {gates['max_low_frequency_contrast']})")
    if metrics["detail_std"] < gates["min_detail_std"]:
        failures.append("nearly flat output")
    if style_cfg["scale"]["method"] == "course_period":
        if not scale["measured"]:
            failures.append("no regular horizontal courses detected")
        elif abs(scale["tilt_deg"]) > gates["max_course_tilt_deg"]:
            failures.append(f"courses tilted by {scale['tilt_deg']:.1f} deg")
    return {"passed": not failures, "failures": failures, "metrics": metrics, "scale": scale}
