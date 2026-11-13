"""Track the real volumetric-CPU -> TripoSR-GPU migration, not a hardware speedup.

Requires the tracked native GPU stack. Does not save models to the user catalog.
GPU requests reuse the live worker, avoiding a second resident neural model.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / 'outputs/runtime/huggingface'))
os.environ.setdefault('U2NET_HOME', str(ROOT / 'outputs/runtime/rembg'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'frontend/public/samples/furniture/chair.png')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--api-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/furniture_migration')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    os.environ['ARCHIGEN_MLFLOW_ENABLED'] = '1'
    from src.furniture.tracking import get_client, get_experiment_id
    from src.furniture.image_to_3d import ImageTo3DGenerator
    from src.furniture.service import generate_furniture_response

    client = get_client()
    experiment_id = get_experiment_id(client)
    parent = client.create_run(experiment_id, tags={
        'mlflow.runName': 'CPU volumetric vs GPU TripoSR', 'run_kind': 'migration_comparison',
        'comparison_type': 'different_engines_before_after',
    }).info.run_id
    from mlflow.entities import Param
    client.log_batch(parent, params=[Param('repetitions_per_engine', str(args.repetitions)),
                                    Param('warmups_per_engine', '1'), Param('target_height_m', '0.85')])
    request = SimpleNamespace(image_base64=base64.b64encode(args.image.read_bytes()).decode(),
                              name='Migration comparison chair', category='chair', remove_bg=True, target_height_m=.85)
    rows = []

    def find_run(item_id):
        found = client.search_runs([experiment_id], filter_string=f"tags.furniture_id = '{item_id}'")
        if len(found) != 1:
            raise RuntimeError(f'Expected exactly one tracked generation for {item_id}, found {len(found)}')
        run = found[0]
        if run.info.status != 'FINISHED' or 'generation_seconds' not in run.data.metrics:
            raise RuntimeError(f'Generation tracking is incomplete: {run.info.run_id}')
        return run

    try:
        baseline = ImageTo3DGenerator(use_gpu=False, engine_mode='volumetric')
        for engine in ('volumetric_cpu', 'triposr_gpu'):
            for index in range(args.repetitions + 1):
                warmup = index == 0
                print(f'{engine}: {"warmup" if warmup else f"measurement {index}/{args.repetitions}"}', flush=True)
                started = time.perf_counter()
                if engine == 'volumetric_cpu':
                    result = generate_furniture_response(request, baseline, tracking_tags={'mlflow.parentRunId': parent})
                else:
                    http_request = urllib.request.Request(
                        args.api_url.rstrip('/') + '/furniture/generate-3d', data=json.dumps(vars(request)).encode(),
                        headers={'Content-Type': 'application/json'},
                    )
                    try:
                        with urllib.request.urlopen(http_request, timeout=850) as response:
                            result = json.load(response)
                    except urllib.error.HTTPError as exc:
                        raise RuntimeError(exc.read().decode()) from exc
                    if 'cuda' not in result['device'] or (result.get('quality') or {}).get('preset') != 'high':
                        raise RuntimeError('The comparison requires the TripoSR CUDA worker in high mode')
                elapsed = time.perf_counter() - started
                run = find_run(result['id'])
                for key, value in {'mlflow.parentRunId': parent, 'run_kind': 'benchmark_warmup' if warmup else 'benchmark_measurement',
                                   'benchmark_configuration': engine, 'benchmark_index': str(index)}.items():
                    client.set_tag(run.info.run_id, key, value)
                rows.append({'configuration': engine, 'warmup': warmup, 'run_id': run.info.run_id,
                             'generation_seconds': run.data.metrics['generation_seconds'],
                             'request_and_tracking_seconds': elapsed, 'engine': result['engine'],
                             'device': result['device'], 'num_faces': result['num_faces']})
        summary = {'parent_run_id': parent, 'comparison_type': 'different_engines_before_after',
                   'interpretation': 'Different algorithms and hardware; duration ratio is not an isolated GPU acceleration.',
                   'repetitions_per_engine': args.repetitions, 'warmups_excluded': True, 'configurations': {}, 'runs': rows}
        for engine in ('volumetric_cpu', 'triposr_gpu'):
            values = [row['generation_seconds'] for row in rows if row['configuration'] == engine and not row['warmup']]
            stats = {'median_seconds': statistics.median(values), 'min_seconds': min(values), 'max_seconds': max(values)}
            summary['configurations'][engine] = stats
            for key, value in stats.items():
                client.log_metric(parent, engine + '_' + key, value)
        ratio = summary['configurations']['volumetric_cpu']['median_seconds'] / summary['configurations']['triposr_gpu']['median_seconds']
        summary['cpu_to_gpu_median_duration_ratio'] = ratio
        client.log_metric(parent, 'cpu_to_gpu_median_duration_ratio', ratio)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f'{parent}.json'
        path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        client.log_artifact(parent, str(path), 'comparison')
        client.set_terminated(parent, 'FINISHED')
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        client.set_tag(parent, 'error', f'{type(exc).__name__}: {exc}'[:1000])
        client.set_terminated(parent, 'FAILED')
        raise


if __name__ == '__main__':
    main()
