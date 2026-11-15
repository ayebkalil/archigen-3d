"""Track inference ablations explicitly separately from training/quality benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess

from experiments.facade_sd15_controlnet.data import digest, write_json
from experiments.facade_sd15_controlnet.tracking import RunTracker


def record(output, uri):
    output = Path(output).resolve()
    profile = json.loads((output / 'profile.json').read_text())
    checks = json.loads((output / 'viewer-checks.json').read_text())
    proxies = json.loads((output / 'appearance-proxies.json').read_text())
    if not profile.get('complete') or checks['errors']:
        raise ValueError('Cannot record a successful incomplete or broken house test')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    tracker = RunTracker(output / 'tracking', 'ArchiGen Facade V3 Inference Constraints',
        params={'weights_changed': False, 'weights_sha256': profile['weights_sha256'],
            'lora_sha256': profile['lora_sha256'], **profile['settings'],
            'material_id': profile['shared_material']['id'], 'trim_m': profile['shared_material']['trim_m'],
            'wall_reference_lighting': profile['shared_material'].get('reference_wall_lighting', 'lit exploratory reference'),
            'house_sha256': digest(output / 'house.json'), 'panels_sha256': digest(output / 'panels.json')},
        tags={'git_commit': revision, 'git_dirty': dirty, 'branch': 'codex/facade-v3-reference-locked',
            'experiment_type': 'inference_compositing_ablation', 'case': 'single_bundled_house',
            'quality_claim': 'protected pixels unchanged by construction; no learned accuracy claim'}, tracking_uri=uri)
    try:
        tracker.client.set_tag(tracker.run_id, 'mlflow.note.content',
            'Local RTX 3050 inference with unchanged V2 weights. No new training. '
            'Protection metrics describe compositing; colour/Sobel values are limited proxies. '
            'Neither a held-out benchmark nor evidence of improved photorealism.')
        tracker.metrics({'panel_count': checks['panels'], 'physical_opening_count': checks['openings'],
            'max_uv_error': checks['maxUvError'], 'javascript_webgl_errors': len(checks['errors']),
            'protected_wall_pixel_max_error': max(p['outside_allowed_mask_max_error'] for p in profile['panels']),
            'blank_walls_exact_match': sum(p['blank_wall_exact_material_match'] is True for p in profile['panels']),
            'generation_mean_seconds': sum(p['seconds'] for p in profile['panels']) / len(profile['panels']),
            'total_seconds': profile['total_seconds'], 'max_torch_allocated_mib': profile['max_allocated_mib'],
            'sampled_peak_device_mib': profile['sampled_peak_device_mib'],
            **{f'{mode}_proxy_{key}': value for mode, values in proxies['modes'].items() for key, value in values.items()}}, 0)
        for name in ('profile.json', 'material.json', 'settings.json', 'viewer-checks.json', 'appearance-proxies.json', 'comparison.png'):
            tracker.artifact(output / name)
        for name in ('v2-house-overview.png', 'guided-house-overview.png', 'locked-house-overview.png', 'locked-house-opposite.png'):
            tracker.artifact(output / name, 'viewer')
        tracker.finish()
    except Exception:
        tracker.finish('FAILED'); raise
    result = {'mlflow_run_id': tracker.run_id, 'source_commit': revision, 'source_dirty': dirty,
        'weights_changed': False, 'held_out_benchmark': False}
    write_json(output / 'tracking-result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--tracking-uri', required=True)
    args = parser.parse_args(); record(args.output, args.tracking_uri)
