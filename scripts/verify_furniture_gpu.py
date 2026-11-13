"""Exercise two consecutive high-quality CUDA reconstructions through the API.

Does not write to the user's furniture catalog. Stores GLBs and measurements in
the selected output directory; the second request exercises opaque-photo masking.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import trimesh


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--image', type=Path, default=root / 'frontend/public/samples/furniture/chair.png')
    parser.add_argument('--output', type=Path, default=root / 'outputs/furniture_quality/rtx4050')
    parser.add_argument('--check-mlflow', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tracking_client = None
    if args.check_mlflow:
        from src.furniture.tracking import get_client, get_experiment_id
        tracking_client = get_client()
        experiment_id = get_experiment_id(tracking_client)
    original = Image.open(args.image).convert('RGBA')
    opaque = Image.new('RGBA', original.size, 'white')
    opaque.alpha_composite(original)
    results = []
    for label, image in (('reference', original), ('opaque', opaque.convert('RGB'))):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        payload = {
            'image_base64': base64.b64encode(buffer.getvalue()).decode(),
            'name': 'GPU verification chair', 'category': 'chair',
            'remove_bg': True, 'target_height_m': .85,
        }
        request = urllib.request.Request(
            args.base_url.rstrip('/') + '/furniture/generate-3d',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'},
        )
        print(f'Generating {label} on CUDA...', flush=True)
        if tracking_client:
            before_runs = {run.info.run_id for run in tracking_client.search_runs([experiment_id])}
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=850) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        quality = result['quality']
        assert result['status'] == 'success'
        assert 'TripoSR Quality' in result['engine'] and 'cuda' in result['device']
        assert quality['checkpoint_loading'] == 'strict' and quality['preset'] == 'high'
        assert quality['density_resolution'] == 384 and quality['texture_resolution'] == 2048
        assert result['num_vertices'] > 0 and result['num_faces'] > 0
        assert 0 < quality['peak_device_sampled_mib'] <= 3500
        if label == 'opaque':
            assert 'u2netp' in quality['mask_source']
        glb = base64.b64decode(result.pop('model_glb_base64').split(',')[-1])
        assert glb[:4] == b'glTF'
        scene = trimesh.load(io.BytesIO(glb), file_type='glb')
        assert len(scene.geometry) > 0
        assert np.isclose(scene.extents[1], .85, atol=.001)
        for mesh in scene.geometry.values():
            assert mesh.visual.material.baseColorTexture.size == (2048, 2048)
        (args.output / f'{label}.glb').write_bytes(glb)
        if tracking_client:
            runs = tracking_client.search_runs([experiment_id], filter_string=f"tags.furniture_id = '{result['id']}'")
            assert len(runs) == 1 and runs[0].info.status == 'FINISHED'
            run = runs[0]
            after_runs = {item.info.run_id for item in tracking_client.search_runs([experiment_id])}
            assert after_runs - before_runs == {run.info.run_id}, 'One API request must create exactly one run'
            assert run.data.params['device_type'] == 'cuda' and run.data.params['preset'] == 'high'
            assert run.data.params['gpu_name'].startswith('NVIDIA GeForce RTX 4050')
            assert run.data.metrics['num_faces'] == result['num_faces']
            artifact_dir = args.output / label / 'tracked_artifacts'
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact = tracking_client.download_artifacts(run.info.run_id, 'output/model.glb', str(artifact_dir))
            assert Path(artifact).read_bytes() == glb
            result['mlflow_run_id'] = run.info.run_id
        preview = base64.b64decode(result.pop('preview_image_base64').split(',')[-1])
        (args.output / f'{label}-mask.png').write_bytes(preview)
        result.update(case=label, http_elapsed_seconds=time.monotonic()-started, glb_bytes=len(glb))
        (args.output / f'{label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    (args.output / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('Two consecutive CUDA reconstructions and GLB validation passed.', flush=True)


if __name__ == '__main__':
    main()
