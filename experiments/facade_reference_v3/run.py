"""Reference-anchored inference ablation; trained V2 weights are never modified."""
import argparse
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import digest, write_json, labels_from_bridge
from experiments.facade_semantic_v2.conditioning import cmp_labels, control_channels
from .materials import facade_chains, material_surface, lock_surface, fingerprint

SETTINGS = {'seed': 42, 'steps': 25, 'strength': .35, 'guidance_scale': 5., 'control_scale': 1.7,
    'prompt': 'Architectural front elevation, uniform warm white fine stucco wall, clean black window frames, clear reflective glass, simple wood door, consistent wall material, neutral diffuse light.',
    'negative_prompt': 'extra windows, extra doors, extra openings, plants, trees, ivy, vegetation, roof, street, people, cars, text, watermark, brick cladding, stone cladding, wood wall cladding, strong shadows, perspective, warped windows'}


def prepare(project, source, output):
    project, source, output = Path(project).resolve(), Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('house.json', 'panels.json'):
        shutil.copyfile(source / name, output / name)
    house, panels = json.loads((output / 'house.json').read_text()), json.loads((output / 'panels.json').read_text())
    phases = facade_chains(house, panels)
    bank = project / 'frontend/public/showcase/tex-1k/white_stucco'
    material = {'id': 'white_stucco', 'source': 'existing project PBR material', 'reference_is_synthetic_render': True,
                'weights_changed': False, 'trim_m': .045, 'surface_resolution': 1024,
                'tile_period': '2 m target, adjusted per closed perimeter to avoid a closure seam',
                'reference_wall_lighting': 'unlit albedo; actual opening meshes retain their materials',
                'maps': {name: digest(bank / (name + '.jpg')) for name in ('diff', 'nor', 'arm')}}
    write_json(output / 'material.json', material)
    write_json(output / 'settings.json', SETTINGS)
    for spec in panels:
        folder = output / spec['wall_id']; folder.mkdir(exist_ok=True)
        for name in ('probe.json', 'sketch.png'):
            shutil.copyfile(source / spec['wall_id'] / name, folder / name)
        shutil.copyfile(source / spec['wall_id'] / 'candidate.png', folder / 'v2.png')
        for name, output_name, background in [('diff', 'base', (127,127,127)), ('nor', 'normal', (128,128,255)), ('arm', 'arm', (255,210,0))]:
            surface = material_surface(spec, Image.open(bank / (name + '.jpg')), phases[spec['wall_id']],
                                       size=material['surface_resolution'], neutral=background)
            if name == 'nor' and np.dot(spec['facade_frame']['direction'], phases[spec['wall_id']]['direction']) < 0:
                # Tangent X reverses when the physically sampled tile runs against UV U.
                pixels = np.asarray(surface).copy(); pixels[:, :, 0] = 255 - pixels[:, :, 0]
                surface = Image.fromarray(pixels)
            surface.save(folder / (output_name + '.png'))
        write_json(folder / 'phase.json', phases[spec['wall_id']])
    print('Prepared shared material surfaces for', len(panels), 'panels. Next: capture PBR references.', flush=True)


def generate(model, output):
    import torch
    from .runtime import ReferenceFacadeRuntime
    from src.models.facade_diffusion import gpu_memory
    output = Path(output).resolve()
    panels = json.loads((output / 'panels.json').read_text())
    settings = json.loads((output / 'settings.json').read_text())
    material = json.loads((output / 'material.json').read_text())
    runtime = ReferenceFacadeRuntime(model, allow_download=False)
    report = {'pipeline': 'reference_guided_v3_with_material_lock', 'weights_changed': False,
        'parent_architecture': runtime.metadata['architecture'], 'weights_sha256': runtime.metadata['weights_sha256'],
        'lora_sha256': runtime.metadata['lora_sha256'], 'settings': settings, 'shared_material': material,
        'device': 'Local RTX 3050 / FP16 / sequential CPU offload', 'panels': [],
        'measurement_note': 'Device-wide samples every 0.5 s; not an instantaneous peak guarantee.'}
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                samples.append(gpu_memory()['used_mib'])
            except Exception:
                pass
            stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start(); started = time.perf_counter()
    try:
        runtime._load(); report['load_seconds'] = time.perf_counter() - started
        for i, spec in enumerate(panels):
            folder = output / spec['wall_id']
            contract = {'protocol': 'reference_v3_1', 'weights': report['weights_sha256'], 'lora': report['lora_sha256'],
                'sketch': digest(folder / 'sketch.png'), 'reference': digest(folder / 'reference.png'),
                'base': digest(folder / 'base.png'), 'settings': settings, 'material': material}
            key = fingerprint(contract)
            prior = json.loads((folder / 'proposal.json').read_text()) if (folder / 'proposal.json').is_file() else {}
            if prior.get('contract') == key and (folder / 'guided.png').is_file() and prior.get('image_sha256') == digest(folder / 'guided.png'):
                profile = json.loads((folder / 'proposal.json').read_text())
                proposal = Image.open(folder / 'guided.png').convert('RGB')
            else:
                channels = control_channels(cmp_labels(labels_from_bridge(Image.open(folder / 'sketch.png'))))
                proposal = runtime.generate_reference(torch.from_numpy(channels)[None],
                    Image.open(folder / 'reference.png').convert('RGB'), settings)
                proposal.save(folder / 'guided.png')
                profile = {'contract': key, 'image_sha256': digest(folder / 'guided.png'), **runtime.last_profile}
                write_json(folder / 'proposal.json', profile)
            base = Image.open(folder / 'base.png').convert('RGB')
            locked, mask = lock_surface(base, proposal, spec, material['trim_m'])
            locked.save(folder / 'locked.png'); Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'allowed.png')
            error = np.abs(np.asarray(locked).astype(int) - np.asarray(base).astype(int))
            outside_error = int(error[~mask].max(initial=0))
            assert outside_error == 0
            if not spec['openings']:
                assert np.array_equal(locked, base)
            result = {'wall_id': spec['wall_id'], 'openings': len(spec['openings']), **profile,
                'outside_allowed_mask_max_error': outside_error, 'allowed_fraction': float(mask.mean()),
                'blank_wall_exact_material_match': bool(np.array_equal(locked, base)) if not spec['openings'] else None}
            report['panels'].append(result); write_json(output / 'profile.json', report)
            print(json.dumps({'completed': i + 1, 'total': len(panels), 'side': spec['wall_id'],
                              'seconds': profile['seconds'], 'protected_wall_pixel_error': outside_error}), flush=True)
        report.update(complete=True, total_seconds=time.perf_counter() - started,
            sampled_peak_device_mib=max(samples), max_allocated_mib=max(p['peak_torch_allocated_mib'] for p in report['panels']))
        if report['sampled_peak_device_mib'] > 3500:
            raise RuntimeError('Device memory exceeded the local budget')
        write_json(output / 'profile.json', report)
    finally:
        stop.set(); thread.join(timeout=10)


def stage(project, output):
    output = Path(output).resolve()
    target = Path(project).resolve() / 'frontend/tmp/facade-reference-v3'; target.mkdir(parents=True, exist_ok=True)
    for name in ('house.json', 'panels.json', 'settings.json', 'material.json'):
        shutil.copyfile(output / name, target / name)
    ready = (output / 'profile.json').is_file() and json.loads((output / 'profile.json').read_text()).get('complete', False)
    write_json(target / 'state.json', {'ready': ready})
    for folder in (p for p in output.iterdir() if p.is_dir() and (p / 'probe.json').is_file()):
        dest = target / folder.name; dest.mkdir(exist_ok=True)
        for file in folder.iterdir():
            if file.suffix in ('.png', '.json'):
                shutil.copyfile(file, dest / file.name)
    for name in ('viewer.jsx', 'surfaceAppearance.js'):
        shutil.copyfile(Path(__file__).with_name(name), target / name)
    (target / 'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>ArchiGen · Constrained facade V3</title></head><body><div id="root"></div><script type="module" src="./viewer.jsx"></script></body></html>', encoding='utf-8')
    print('Staged http://127.0.0.1:3001/tmp/facade-reference-v3/', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare'); prep.add_argument('--project', required=True); prep.add_argument('--source', required=True)
    gen = commands.add_parser('generate'); gen.add_argument('--model', required=True)
    viewer = commands.add_parser('stage'); viewer.add_argument('--project', required=True)
    for sub in (prep, gen, viewer): sub.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.project, args.source, args.output)
    elif args.command == 'generate': generate(args.model, args.output)
    else: stage(args.project, args.output)


if __name__ == '__main__': main()
