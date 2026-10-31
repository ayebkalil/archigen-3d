"""Offline experiment helpers. Importing this module never loads model weights.

Control images guide diffusion; they do NOT guarantee window alignment or make
pretrained text styles into CMP-supervised style classes.
"""
import base64
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.data.facade_palette import FACADE_STYLES

ROOT = Path(__file__).resolve().parents[2]
STYLE_PROMPTS = {
    "neutral": "restrained architectural materials, painted plaster, clear glass",
    FACADE_STYLES[0]: "modern minimalist facade, white fine stucco, slim dark metal window frames",
    FACADE_STYLES[1]: "Mediterranean limestone facade, natural pale stone, restrained stone detailing",
    FACADE_STYLES[2]: "traditional red brick facade, crisp brick mortar, painted wood window frames",
    FACADE_STYLES[3]: "plaster facade with climbing vegetation between windows, visible window frames",
}
NEGATIVE_PROMPT = (
    "perspective view, angled building, street, people, cars, text, watermark, "
    "extra windows, extra doors, bent frames, warped walls, blurry, painting, illustration"
)


def prompt_for(style):
    return (
        "Straight-on orthographic architectural facade elevation, flat rectangular wall, "
        "exact window and door arrangement from the outline, " + STYLE_PROMPTS[style] +
        ", realistic material texture, crisp glass reflections, even diffuse daylight, "
        "minimal baked shadows, no surrounding scene."
    )


def uv_box(rect, size):
    """Three.js bottom-origin UVs -> PIL top-origin pixel coordinates."""
    u0, v0, u1, v1 = map(float, rect)
    if not all(np.isfinite([u0, v0, u1, v1])) or not (0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1):
        raise ValueError(f"Invalid UV rectangle: {rect}")
    # UVs denote pixel boundaries, not pixel-centre indices. Clamp outer borders.
    return tuple(min(size - 1, max(0, round(p * size))) for p in (u0, 1-v1, u1, 1-v0))


def fixture_control(fixture, size=1024):
    """Use authoritative physical opening UVs, not edges extracted from blurry GAN output."""
    edge = Image.new("L", (size, size))
    mask = Image.new("L", (size, size))
    draw, fill = ImageDraw.Draw(edge), ImageDraw.Draw(mask)
    b = fixture["uv_bounds"]
    draw.rectangle(uv_box([b['u_min'], b['v_min'], b['u_max'], b['v_max']], size), outline=255)
    for opening in fixture["openings"]:
        box = uv_box(opening["uv_rect"], size)
        draw.rectangle(box, outline=255)
        fill.rectangle(box, fill=255)
    return edge.convert("RGB"), mask


def cmp_control(labels, size=1024):
    """Binary boundaries from original indexed masks; CMP colours are not Canny inputs."""
    labels = np.asarray(labels)
    if (labels.ndim != 2 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() > 12):
        raise ValueError("Expected original CMP class indices 0..12, not an RGB image")
    labels = np.asarray(Image.fromarray(labels.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
    edges = np.zeros((size, size), np.uint8)
    for label in np.unique(labels):
        edges |= cv2.Canny((labels == label).astype(np.uint8) * 255, 100, 200)
    opening = np.isin(labels, [3, 4]).astype(np.uint8) * 255
    return Image.fromarray(edges).convert("RGB"), Image.fromarray(opening)


def read_fixture_image(fixture, key):
    return Image.open(io.BytesIO(base64.b64decode(fixture[key].split(",", 1)[1]))).convert("RGB")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_memory():
    """Device-wide MiB; includes graphics and other processes, unlike torch stats."""
    result = subprocess.run([
        "nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=10)
    total, used = [int(v.strip()) for v in result.stdout.strip().split(",")]
    return {"total_mib": total, "used_mib": used}


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def resource_report(cache, *, download=False, cached=False, disk_dir=None):
    import psutil
    free_disk = shutil.disk_usage(existing_parent(cache)).free
    free_ram = psutil.virtual_memory().available
    errors = []
    # Conservative headroom includes environment setup, temporary download data,
    # and Windows HF cache copies when symlinks are unavailable.
    required_disk = (2 if cached else 15) * 1024**3
    if free_disk < required_disk:
        errors.append(f"Need {required_disk / 1024**3:.0f} GiB free on model-cache drive; have {free_disk / 1024**3:.2f}")
    gpu = None
    if not download:
        required_ram = 3 if disk_dir else 8
        if free_ram < required_ram * 1024**3:
            errors.append(f"Need {required_ram} GiB available RAM for SDXL offload preflight; have {free_ram / 1024**3:.2f}")
        if disk_dir:
            # Weight cache and offload directory may share a drive. Require room for both
            # when downloading is still necessary, otherwise just for the disk copy.
            offload_free = shutil.disk_usage(existing_parent(disk_dir)).free
            required_offload = (12 if cached else 22) * 1024**3
            if offload_free < required_offload:
                errors.append(f"Need {required_offload / 1024**3:.0f} GiB free on disk-offload drive")
        try:
            gpu = gpu_memory()
            if min(3500, gpu['total_mib']) - gpu['used_mib'] - 512 < 1024:
                errors.append("Need at least 1536 MiB headroom below the 3500 MiB total GPU budget")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"Cannot measure CUDA GPU: {exc}")
    return {"cache": str(Path(cache).resolve()), "offload": "disk" if disk_dir else "CPU",
            "free_disk_gib": free_disk / 1024**3,
            "available_ram_gib": free_ram / 1024**3, "gpu": gpu,
            "ready": not errors, "blockers": errors}


def cached_models(config, cache):
    paths = {}
    for name, model in config.items():
        path = Path(cache) / ("models--" + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']
        if not all((path / file).is_file() for file in model['files']):
            return None
        if any((path / file).stat().st_size != spec['size'] for file, spec in model.get('weights', {}).items()):
            return None
        paths[name] = path
    return paths


def prepare_cases(output, fixture_path, manifest_path, count=4):
    """Fixed validation prefix, never test samples or photo-derived control edges."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(Path(fixture_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    selected = sorted((r for r in manifest['samples'] if r['split'] == 'val'), key=lambda r: r['id'])[:count]
    if len(selected) != count:
        raise ValueError("Not enough validation samples")
    cases = []
    control, mask = fixture_control(fixture)
    control.save(output / 'viewer_control.png')
    mask.save(output / 'viewer_openings.png')
    read_fixture_image(fixture, 'texture_base64').save(output / 'viewer_baseline512.png')
    cases.append({"id": "viewer", "control": "viewer_control.png", "openings": "viewer_openings.png",
                  "baseline": "viewer_baseline512.png", "source_fixture": str(Path(fixture_path).resolve()),
                  "source_sha256": sha256(fixture_path)})
    for index, row in enumerate(selected):
        stem = f"val_{index:02d}"
        label_path = ROOT / Path(row['photo']).with_suffix('.png')
        with Image.open(label_path) as image:
            control, mask = cmp_control(np.array(image))
        control.save(output / f'{stem}_control.png')
        mask.save(output / f'{stem}_openings.png')
        with Image.open(ROOT / row['photo']) as photo:
            photo.convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS).save(output / f'{stem}_target.png')
        cases.append({"id": stem, "cmp_id": row['id'], "split": "val", "control": f'{stem}_control.png',
                      "openings": f'{stem}_openings.png', "target": f'{stem}_target.png',
                      "source_sha256": sha256(label_path), "native_size": row['native_size']})
    for case in cases:
        case['control_sha256'] = sha256(output / case['control'])
    report = {"resolution": 1024, "selection": "first sorted validation IDs; no test selection",
              "manifest_sha256": sha256(manifest_path), "cases": cases,
              "style_status": "pretrained text conditioning only; no CMP style training",
              "status": "controls prepared; no diffusion images generated"}
    (output / 'cases.json').write_text(json.dumps(report, indent=2))
    return report


def comparison(control, result, reference, path, reference_label):
    canvas = Image.new('RGB', (1536, 548), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(zip([control, reference, result],
                                      ['Geometry control', reference_label, 'SDXL candidate (not yet accepted)'])):
        canvas.paste(im.resize((512, 512), Image.Resampling.LANCZOS), (512*i, 36))
        draw.text((512*i+8, 10), title, fill='black')
    canvas.save(path)
