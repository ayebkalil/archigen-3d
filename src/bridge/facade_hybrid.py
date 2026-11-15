"""Geometry-locked hybrid facade composition.

Stable Diffusion only supplies a periodic *material swatch*. Everything about the
building comes from the wall geometry: physical size, storeys and the exact door and
window rectangles. Nothing here moves, resizes, adds or removes an opening; invalid
inputs are reported instead of being clipped.

Texture convention (same fields as the Pix2Pix bridge, consumed unchanged by
Viewer3D/createFacadeGeometry): ``uv_bounds`` locate the wall inside the image with
bottom-origin V, and each opening carries a bottom-origin ``uv_rect``. The hybrid
texture covers the wall exactly, so ``uv_bounds`` is the full unit square and the
image keeps the wall's aspect ratio instead of a square letterbox.

Outputs are colour images only. No normal, roughness or other PBR map is produced.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.floorplan.dimensions import STOREY_HEIGHT_M


class FacadeGeometryError(ValueError):
    """The wall cannot be composed without changing its openings."""

    def __init__(self, issues: List[str]):
        super().__init__("; ".join(issues))
        self.issues = issues


def wall_spec(wall: Dict[str, Any], num_floors: int = 1, floor_height_m: Optional[float] = None,
              is_front: bool = True, tolerance_m: float = 0.001) -> Dict[str, Any]:
    """Exact physical facade description shared by masks, validation and UVs."""
    from src.bridge.facade_bridge import FacadeBridge, wall_opening_templates

    issues = []
    if not isinstance(num_floors, int) or not 1 <= num_floors <= 10:
        raise FacadeGeometryError([f"num_floors must be an integer between 1 and 10, got {num_floors!r}"])
    frame = FacadeBridge.facade_frame(wall)
    length = float(frame["length_m"])
    floor_height = float(floor_height_m if floor_height_m is not None else wall.get("height_m", STOREY_HEIGHT_M))
    if not (math.isfinite(length) and length >= 0.3):
        issues.append(f"wall length {length:.3f} m is too small or invalid")
    if not (math.isfinite(floor_height) and 1.5 <= floor_height <= 8.0):
        issues.append(f"storey height {floor_height!r} m is outside 1.5..8 m")
    if issues:
        raise FacadeGeometryError(issues)
    height = num_floors * floor_height
    openings = []
    for template in wall_opening_templates(wall, frame, is_front):
        for floor in range(num_floors):
            if template["type"] == "door" and floor > 0:
                continue
            half = template["width_m"] / 2
            left, right = template["position_along_wall_m"] - half, template["position_along_wall_m"] + half
            bottom = floor * floor_height + template["sill_m"]
            top = bottom + template["height_m"]
            opening_id = template["id"] if floor == 0 else f"{template['id']}@{floor}"
            values = [left, right, bottom, top]
            if not all(math.isfinite(v) for v in values) or right - left <= 0 or top - bottom <= 0:
                issues.append(f"{opening_id}: non-positive or invalid size")
                continue
            if left < -tolerance_m or right > length + tolerance_m:
                issues.append(f"{opening_id}: spans {left:.3f}..{right:.3f} m outside wall 0..{length:.3f} m")
            if bottom < -tolerance_m or top > height + tolerance_m:
                issues.append(f"{opening_id}: spans height {bottom:.3f}..{top:.3f} m outside 0..{height:.3f} m")
            if top > (floor + 1) * floor_height + tolerance_m:
                issues.append(f"{opening_id}: crosses into the storey above")
            openings.append({
                "id": opening_id, "source_id": template["id"], "type": template["type"], "floor_index": floor,
                "left_m": left, "right_m": right, "bottom_m": bottom, "top_m": top,
                # Same field semantics as FacadeBridge rendered openings (sill from wall base).
                "position_along_wall_m": (left + right) / 2, "width_m": right - left,
                "height_m": top - bottom, "sill_m": bottom,
            })
    for i, a in enumerate(openings):
        for b in openings[i + 1:]:
            if min(a["right_m"], b["right_m"]) - max(a["left_m"], b["left_m"]) > tolerance_m and \
                    min(a["top_m"], b["top_m"]) - max(a["bottom_m"], b["bottom_m"]) > tolerance_m:
                issues.append(f"{a['id']} overlaps {b['id']}")
    if issues:
        raise FacadeGeometryError(issues)
    for opening in openings:
        opening["uv_rect"] = [opening["left_m"] / length, opening["bottom_m"] / height,
                              opening["right_m"] / length, opening["top_m"] / height]
        opening["edge_clearance_m"] = min(opening["left_m"], length - opening["right_m"],
                                          opening["bottom_m"], height - opening["top_m"])
    return {"wall_id": wall.get("id", "wall"), "frame": frame, "wall_length_m": length,
            "wall_height_m": height, "floor_height_m": floor_height, "num_floors": num_floors,
            "openings": openings}


def geometry_hash(spec: Dict[str, Any]) -> str:
    """Stable fingerprint of everything that determines masks and UVs."""
    keep = {
        "frame": spec["frame"], "wall_length_m": spec["wall_length_m"], "wall_height_m": spec["wall_height_m"],
        "openings": [{k: o[k] for k in ("id", "type", "left_m", "right_m", "bottom_m", "top_m")} for o in spec["openings"]],
    }
    text = json.dumps(keep, sort_keys=True, default=lambda v: round(float(v), 6))
    return hashlib.sha256(text.encode()).hexdigest()


def texture_size(spec: Dict[str, Any], max_px_per_m: float = 256, max_px: int = 4096) -> Tuple[int, int, float]:
    """Isotropic pixel density: one density for both axes, so materials never stretch."""
    density = min(max_px_per_m, max_px / spec["wall_length_m"], max_px / spec["wall_height_m"])
    return max(1, round(spec["wall_length_m"] * density)), max(1, round(spec["wall_height_m"] * density)), density


def opening_pixel_box(opening: Dict[str, Any], width: int, height: int, spec: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Top-origin, end-exclusive pixel box of an opening in a full-wall texture."""
    sx, sy = width / spec["wall_length_m"], height / spec["wall_height_m"]
    x0, x1 = round(opening["left_m"] * sx), round(opening["right_m"] * sx)
    y0, y1 = round(height - opening["top_m"] * sy), round(height - opening["bottom_m"] * sy)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def opening_masks(spec: Dict[str, Any], width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Label image (0 wall, i+1 opening i) and a binary union mask."""
    labels = np.zeros((height, width), np.int32)
    for index, opening in enumerate(spec["openings"]):
        x0, y0, x1, y1 = opening_pixel_box(opening, width, height, spec)
        labels[y0:y1, x0:x1] = index + 1
    return labels, labels > 0


def tile_material(material: Image.Image, tile_m: float, width: int, height: int, density: float,
                  offset_px: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Periodically repeat a square swatch at its physical size (no mirroring)."""
    if material.width != material.height:
        raise ValueError("Material swatch must be square")
    if not (math.isfinite(tile_m) and tile_m > 0):
        raise ValueError("Physical tile size must be positive")
    side = max(4, round(tile_m * density))
    tile = np.asarray(material.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))
    reps_y, reps_x = math.ceil((height + side) / side), math.ceil((width + side) / side)
    big = np.tile(tile, (reps_y, reps_x, 1))
    ox, oy = offset_px[0] % side, offset_px[1] % side
    return big[oy:oy + height, ox:ox + width].copy()


def draw_openings(image: np.ndarray, spec: Dict[str, Any], labels: np.ndarray, colors: Dict[str, List[int]]) -> None:
    """Schematic glass/door placeholders strictly inside each opening mask."""
    height, width = labels.shape
    for index, opening in enumerate(spec["openings"]):
        x0, y0, x1, y1 = opening_pixel_box(opening, width, height, spec)
        if x1 <= x0 or y1 <= y0:
            continue
        kind = opening["type"]
        region = image[y0:y1, x0:x1]
        region[:] = colors[f"{kind}_frame"]
        border = max(1, round(min(x1 - x0, y1 - y0) * 0.06))
        inner = region[border:max(border, region.shape[0] - border), border:max(border, region.shape[1] - border)]
        inner[:] = colors[kind]
        if kind == "window" and inner.shape[1] > 6:
            mid = inner.shape[1] // 2
            inner[:, max(0, mid - border // 2):mid + max(1, border // 2)] = colors["window_frame"]


def compose_hybrid_facade(spec: Dict[str, Any], material: Image.Image, tile_m: float,
                          composition: Dict[str, Any], offset_seed: int = 0) -> Dict[str, Any]:
    width, height, density = texture_size(spec, composition["max_px_per_m"], composition["max_texture_px"])
    labels, mask = opening_masks(spec, width, height)
    rng = np.random.default_rng(offset_seed)
    side = max(4, round(tile_m * density))
    image = tile_material(material, tile_m, width, height, density,
                          (int(rng.integers(0, side)), int(rng.integers(0, side))))
    wall_only = image.copy()
    draw_openings(image, spec, labels, composition["opening_fill"])
    openings = [{k: v for k, v in o.items() if k not in ("left_m", "right_m", "bottom_m", "top_m")}
                for o in spec["openings"]]
    uv_bounds = {"u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0,
                 "wall_length_m": spec["wall_length_m"], "wall_height_m": spec["wall_height_m"],
                 "aspect_ratio": round(spec["wall_length_m"] / spec["wall_height_m"], 3)}
    return {"image": Image.fromarray(image), "wall_material": wall_only, "labels": labels, "mask": mask,
            "uv_bounds": uv_bounds, "openings": openings, "density_px_per_m": density,
            "tile_px": side, "size": [width, height],
            "repeats": [spec["wall_length_m"] / tile_m, spec["wall_height_m"] / tile_m]}


def mask_image(labels: np.ndarray) -> Image.Image:
    """Diagnostic mask: black wall, white openings."""
    return Image.fromarray(np.where(labels > 0, 255, 0).astype(np.uint8))


def validate_composition(spec: Dict[str, Any], result: Dict[str, Any], material_tile_px: Optional[int] = None) -> Dict[str, Any]:
    """Automated geometry checks on the composed texture (not on the prompt)."""
    width, height = result["size"]
    labels = result["labels"]
    image = np.asarray(result["image"])
    checks, failures = {}, []
    sx, sy = width / spec["wall_length_m"], height / spec["wall_height_m"]
    checks["isotropic_density"] = abs(sx - sy) / max(sx, sy) < 0.01 or abs(sx - sy) <= 1.0 / min(spec["wall_length_m"], spec["wall_height_m"])
    if not checks["isotropic_density"]:
        failures.append(f"non-isotropic pixel density {sx:.2f} vs {sy:.2f} px/m")
    tolerance_m = 0.5 / min(sx, sy) + 1e-6
    max_error = 0.0
    for index, opening in enumerate(spec["openings"]):
        ys, xs = np.nonzero(labels == index + 1)
        if not len(xs):
            failures.append(f"{opening['id']}: opening smaller than one pixel")
            continue
        measured = {"left_m": xs.min() / sx, "right_m": (xs.max() + 1) / sx,
                    "top_m": (height - ys.min()) / sy, "bottom_m": (height - ys.max() - 1) / sy}
        error = max(abs(measured[k] - opening[k]) for k in measured)
        max_error = max(max_error, error)
        if error > tolerance_m:
            failures.append(f"{opening['id']}: mask differs from geometry by {error * 1000:.1f} mm")
        expected_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if expected_area != len(xs):
            failures.append(f"{opening['id']}: mask is not one rectangle")
        u0, v0, u1, v1 = result["openings"][index]["uv_rect"]
        uv_error = max(abs(u0 * spec["wall_length_m"] - opening["left_m"]), abs(u1 * spec["wall_length_m"] - opening["right_m"]),
                       abs(v0 * spec["wall_height_m"] - opening["bottom_m"]), abs(v1 * spec["wall_height_m"] - opening["top_m"]))
        if uv_error > 1e-6:
            failures.append(f"{opening['id']}: uv_rect disagrees with geometry")
    checks["opening_count"] = len(result["openings"]) == len(spec["openings"])
    if not checks["opening_count"]:
        failures.append("opening count changed during composition")
    # Pixels outside masks must be pure material: nothing from an opening renderer leaks.
    outside = labels == 0
    leaked = int(np.count_nonzero(np.any(image[outside] != result["wall_material"][outside], axis=-1)))
    checks["outside_mask_is_material_only"] = leaked == 0
    if leaked:
        failures.append(f"{leaked} wall pixels were modified outside opening masks")
    return {"passed": not failures, "failures": failures, "checks": checks,
            "max_mask_error_m": max_error, "mask_tolerance_m": tolerance_m,
            "openings": len(spec["openings"]),
            "min_edge_clearance_m": min((o["edge_clearance_m"] for o in spec["openings"]), default=None)}


def pix2pix_geometry_report(spec: Dict[str, Any], pix2pix_openings: List[Dict[str, Any]], texture_px: int) -> Dict[str, Any]:
    """How far the 256 px Pix2Pix raster moved each opening from the exact geometry."""
    exact = {o["id"]: o for o in spec["openings"]}
    errors, missing = [], []
    for opening in pix2pix_openings:
        key = opening.get("id") if opening.get("floor_index", 0) == 0 else f"{opening.get('id')}@{opening.get('floor_index')}"
        target = exact.get(key)
        if target is None:
            missing.append(key)
            continue
        left = opening["position_along_wall_m"] - opening["width_m"] / 2
        right = opening["position_along_wall_m"] + opening["width_m"] / 2
        bottom, top = opening["sill_m"], opening["sill_m"] + opening["height_m"]
        errors.append(max(abs(left - target["left_m"]), abs(right - target["right_m"]),
                          abs(bottom - target["bottom_m"]), abs(top - target["top_m"])))
    found = {(o.get("id") if o.get("floor_index", 0) == 0 else f"{o.get('id')}@{o.get('floor_index')}") for o in pix2pix_openings}
    return {"max_error_m": max(errors, default=0.0), "mean_error_m": float(np.mean(errors)) if errors else 0.0,
            "dropped": sorted(set(exact) - found), "unexpected": missing, "texture_px": texture_px}
