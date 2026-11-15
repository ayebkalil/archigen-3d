"""Prepare and generate the actual two-storey house's exterior panels locally."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

from experiments.facade_sd15_controlnet.data import digest, write_json


def prepare(project, output, api='http://127.0.0.1:8000'):
    import httpx
    from src.bridge.facade_bridge import FacadeBridge
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Use the current production opening-fitting rules rather than the old,
    # narrow raw sample detections. The production source is only read here.
    command = '''import {createSampleHouse} from './frontend/src/geometry/sampleHousePlan.js';
import {buildingLevels,fitHouseOpenings} from './frontend/src/geometry/houseGeometry.js';
const layout=createSampleHouse();
const fitted=buildingLevels(layout).flatMap(level=>fitHouseOpenings(level,layout.metadata.scale_meters_per_pixel));
for(const wall of layout.walls) wall.openings=fitted.filter(op=>op.associated_wall===wall.id);
layout.doors=fitted.filter(op=>op.type==='door');layout.windows=fitted.filter(op=>op.type==='window');
console.log(JSON.stringify(layout));'''
    result = subprocess.run(['node', '--input-type=module', '-e', command], cwd=project,
                            capture_output=True, text=True, check=True)
    layout = json.loads(result.stdout)
    write_json(output / 'house.json', layout)
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.inference_engine = SimpleNamespace(resolution=512, neural_style=True)
    bridge.enhance_textures = False
    panels = []
    with httpx.Client(timeout=120, trust_env=False) as client:
        for wall in layout['walls']:
            if not wall.get('is_exterior'):
                continue
            wall = {**wall, 'geometry_fixed': True}
            target = output / wall['id']; target.mkdir(exist_ok=True)
            sketch, bounds, frame, openings = bridge._build_facade_sketch(
                wall, num_floors=1, is_front=wall.get('is_front', False), target_size=512)
            sketch.save(target / 'sketch.png')
            spec = {'wall_id': wall['id'], 'level_id': wall['level_id'], 'uv_bounds': bounds,
                'facade_frame': frame, 'openings': openings, 'wall_height_m': bounds['wall_height_m'],
                'wall_length_m': bounds['wall_length_m'], 'sketch_sha256': digest(target / 'sketch.png')}
            write_json(target / 'probe.json', spec)
            response = client.post(api.rstrip('/') + '/facade/generate',
                json={'wall': wall, 'num_floors': 1, 'style': None})
            response.raise_for_status()
            baseline = response.json()
            (target / 'baseline.png').write_bytes(base64.b64decode(baseline['texture_base64'].split(',')[-1]))
            baseline.pop('texture_base64', None); baseline.pop('sketch_base64', None)
            write_json(target / 'baseline.json', baseline)
            panels.append(spec)
            print(f'Prepared {wall["id"]}: {len(openings)} openings, {frame["length_m"]:.2f} m', flush=True)
    from PIL import Image
    Image.new('RGB', (512, 512), (243, 240, 233)).save(output / 'plain.png')
    write_json(output / 'panels.json', panels)
    print(f'Prepared {len(panels)} exterior panels with {sum(len(p["openings"]) for p in panels)} fixed openings.', flush=True)


def generate(model, output):
    from .runtime import SemanticFacadeRuntime
    from src.models.facade_diffusion import gpu_memory
    output = Path(output).resolve()
    panels = json.loads((output / 'panels.json').read_text())
    runtime = SemanticFacadeRuntime(model, allow_download=False)
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                samples.append({'time': time.perf_counter(), **gpu_memory()})
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    report = {'architecture': runtime.metadata['architecture'], 'checkpoint_step': runtime.metadata['selected_step'],
        'weights_sha256': runtime.metadata['weights_sha256'], 'lora_sha256': runtime.metadata['lora_sha256'],
        'seed': runtime.config['seed'], 'prompt': runtime.config['prompt'], 'inference_steps': runtime.config['inference_steps'],
        'style': None, 'panels': [], 'device': 'Local RTX 3050 / FP16 / sequential CPU offload',
        'measurement_note': 'Device-wide samples every 0.5 s are not an instantaneous peak guarantee.'}
    started = time.perf_counter()
    try:
        runtime._load(); report['load_seconds'] = time.perf_counter() - started
        print(f'V2 loaded in {report["load_seconds"]:.1f}s; generating {len(panels)} sides.', flush=True)
        for index, spec in enumerate(panels):
            directory = output / spec['wall_id']
            metadata = directory / 'candidate.json'
            if metadata.is_file() and (directory / 'candidate.png').is_file():
                previous = json.loads(metadata.read_text())
                if previous.get('weights_sha256') == report['weights_sha256'] and previous.get('lora_sha256') == report['lora_sha256'] and previous.get('sketch_sha256') == spec['sketch_sha256'] and previous.get('seed') == report['seed']:
                    report['panels'].append(previous)
                    print('Using matching previously generated side:', spec['wall_id'], flush=True)
                    continue
            first_sample = len(samples)
            image = runtime.generate_sketch((directory / 'sketch.png').read_bytes(), style=None)
            image.save(directory / 'candidate.png')
            peak = max([s['used_mib'] for s in samples[first_sample:]] or [gpu_memory()['used_mib']])
            if peak > 3500:
                raise RuntimeError('Sampled device-wide VRAM exceeds 3500 MiB')
            result = {**spec, **runtime.last_profile, 'sampled_peak_device_mib': peak,
                      'seed': report['seed'], 'weights_sha256': report['weights_sha256'], 'lora_sha256': report['lora_sha256']}
            write_json(metadata, result); report['panels'].append(result)
            write_json(output / 'profile.json', report)
            print(json.dumps({'side': spec['wall_id'], 'completed': index + 1, 'total': len(panels),
                'seconds': result['seconds'], 'sampled_peak_device_mib': peak}), flush=True)
        report['sampled_peak_device_mib'] = max(s['used_mib'] for s in samples)
        report['total_seconds'] = time.perf_counter() - started
        report['complete'] = len(report['panels']) == len(panels)
        if report['sampled_peak_device_mib'] > 3500:
            raise RuntimeError('Device-wide VRAM budget exceeded during model loading/generation')
        write_json(output / 'profile.json', report)
    finally:
        stop.set(); thread.join(timeout=12)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    source = sub.add_parser('prepare'); source.add_argument('--project', required=True); source.add_argument('--api', default='http://127.0.0.1:8000')
    gen = sub.add_parser('generate'); gen.add_argument('--model', required=True)
    for item in (source, gen):
        item.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.project, args.output, args.api)
    else:
        generate(args.model, args.output)


if __name__ == '__main__': main()
