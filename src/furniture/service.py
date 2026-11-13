"""Shared API response for local and GPU worker reconstruction."""

import base64
import io
import time

from PIL import Image
from .tracking import GenerationRun


def generate_furniture_response(request, generator, *, tracking_tags=None):
    run = GenerationRun(request, generator, tags=tracking_tags)
    try:
        return _generate(request, generator, run)
    except Exception as exc:
        run.fail(exc)
        raise


def _generate(request, generator, run):
    image_bytes = base64.b64decode(request.image_base64.split(",")[-1])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    run.input(image_bytes, image)
    started = time.perf_counter()
    result = generator.generate_from_image(
        image=image,
        name=request.name or "Custom Furniture",
        category=request.category or "chair",
        remove_bg=bool(request.remove_bg),
        target_height_m=request.target_height_m,
    )
    duration = time.perf_counter() - started
    run.finish(result, duration)
    return {
        "status": "success",
        "id": result["id"],
        "name": result["name"],
        "category": result["category"],
        "dimensions_m": result["dimensions_m"],
        "bounds": result["bounds"],
        "num_vertices": result["num_vertices"],
        "num_faces": result["num_faces"],
        "roughness": result.get("roughness", 0.35),
        "engine": result["engine"],
        "device": result["device"],
        "model_glb_base64": result["glb_base64"],
        "preview_image_base64": result["preview_image_base64"],
        "quality": result.get("quality"),
    }
