"""Explicit, local-only SDXL experiment. No production imports or implicit downloads."""
import argparse
import base64
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
import numpy as np

from src.models.facade_diffusion import (
    STYLE_PROMPTS, NEGATIVE_PROMPT, cached_models, comparison, gpu_memory,
    prepare_cases, prompt_for, resource_report, sha256,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def download(config, cache):
    from huggingface_hub import hf_hub_download
    from src.models.facade_download import download_weight
    for model in config.values():
        print(f"Downloading pinned files: {model['repo']}", flush=True)
        snapshot = cache / ("models--" + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']
        for filename in model['files']:
            if filename in model['weights']:
                url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{filename}"
                download_weight(url, snapshot / filename, model['weights'][filename])
            else:
                hf_hub_download(model['repo'], filename, revision=model['revision'], cache_dir=cache)
        print(f"Finished: {model['repo']}", flush=True)


def generate(args, config, paths):
    # Prevent any loader from fetching missing components during inference.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from src.models.facade_diffusion_runtime import load_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA torch is required in the isolated diffusion environment')
    torch.set_num_threads(4)
    initial = gpu_memory()
    allocator_mib = min(3500, initial['total_mib']) - initial['used_mib'] - 512
    if allocator_mib < 1024:
        raise RuntimeError('Insufficient GPU headroom; close other GPU workloads first')
    torch.cuda.set_per_process_memory_fraction(allocator_mib / initial['total_mib'], 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    document = json.loads((args.output / 'cases.json').read_text())
    case = next((c for c in document['cases'] if c['id'] == args.case), None)
    if case is None:
        raise ValueError(f'Unknown case {args.case}; prepare controls first')
    control_path = args.output / case['control']
    if sha256(control_path) != case['control_sha256']:
        raise ValueError('Control image changed after case selection; re-prepare explicitly')
    control = Image.open(control_path).convert('RGB')
    if control.size != (1024, 1024):
        raise ValueError('Expected 1024 x 1024 control image')
    target = args.output / f"{args.case}_{list(STYLE_PROMPTS).index(args.style)}_{args.seed}"
    target.mkdir(exist_ok=False)  # Do not overwrite prior candidates or evidence.
    metadata = {
        'status': 'running', 'case': case, 'seed': args.seed, 'style': args.style,
        'style_status': 'pretrained text guidance; not CMP-trained architectural style classes',
        'prompt': prompt_for(args.style), 'negative_prompt': NEGATIVE_PROMPT,
        'resolution': 1024, 'steps': args.steps, 'control_scale': args.control_scale,
        'guidance_scale': 5.0, 'models': config, 'initial_gpu': initial,
        'allocator_limit_mib': allocator_mib, 'sampled_peak_device_mib': initial['used_mib'],
        'offload': args.offload, 'disk_offload_dir': str(args.disk_dir) if args.offload == 'disk' else None,
        'memory_limit_note': 'Allocator cap plus step checks; external GPU allocations are not controlled',
        'quality_review': 'pending; ControlNet does not guarantee opening alignment',
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'diffusers', 'transformers', 'accelerate', 'huggingface-hub')},
    }
    write_json(target / 'run.json', metadata)
    pipe = None
    started = time.monotonic()

    def check_memory():
        gpu = gpu_memory()
        metadata['sampled_peak_device_mib'] = max(metadata['sampled_peak_device_mib'], gpu['used_mib'])
        if gpu['used_mib'] > 3500:
            raise RuntimeError('Device usage exceeded 3500 MiB; stopping this experiment')

    def on_step(pipe, step, timestep, callback_kwargs):
        check_memory()
        if step % 5 == 0:
            print(f'Step {step + 1}/{args.steps}; GPU {metadata["sampled_peak_device_mib"]} MiB peak sampled', flush=True)
        return callback_kwargs

    try:
        # Revision-keyed directory prevents stale offload files after model changes.
        key = '-'.join(config[name]['revision'][:12] for name in ('base', 'controlnet', 'vae'))
        pipe = load_pipeline(paths, disk_dir=args.disk_dir / key if args.offload == 'disk' else None)
        check_memory()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = pipe(
                prompt=metadata['prompt'], negative_prompt=NEGATIVE_PROMPT, image=control,
                width=1024, height=1024, num_inference_steps=args.steps,
                guidance_scale=5.0, controlnet_conditioning_scale=args.control_scale,
                generator=torch.Generator(device='cpu').manual_seed(args.seed),
                callback_on_step_end=on_step,
            ).images[0]
        check_memory()
        if result.size != (1024, 1024) or np.asarray(result).std() < 1:
            raise RuntimeError('Invalid or nearly constant output; no successful result recorded')
        result.save(target / 'facade.png')
        reference_key = 'baseline' if 'baseline' in case else 'target'
        with Image.open(args.output / case[reference_key]) as reference:
            comparison(control, result, reference, target / 'comparison.png',
                       'Previous 512 GAN' if reference_key == 'baseline' else 'CMP reference (not model output)')
        if 'source_fixture' in case:
            if sha256(case['source_fixture']) != case['source_sha256']:
                raise ValueError('Source viewer geometry changed; re-prepare controls')
            fixture = json.loads(Path(case['source_fixture']).read_text())
            fixture['texture_base64'] = 'data:image/png;base64,' + base64.b64encode((target / 'facade.png').read_bytes()).decode()
            # Preserve semantic diagnostic texture, only increase its sampling size.
            sketch = Image.open(io.BytesIO(base64.b64decode(fixture['sketch_base64'].split(',', 1)[1])))
            buffer = io.BytesIO()
            sketch.resize((1024, 1024), Image.Resampling.NEAREST).save(buffer, format='PNG')
            fixture['sketch_base64'] = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
            fixture['style'] = args.style
            fixture['device'] = f'Local SDXL {args.offload} offload / CUDA'
            fixture['model_info'] = {'model': 'sdxl_canny_small_experiment', 'resolution': 1024,
                                     'style_conditioned': True, 'requested_style': args.style,
                                     'effective_style': args.style + ' (pretrained text guidance)',
                                     'supported_styles': [], 'cmp_style_training': False}
            write_json(target / 'viewer_fixture.json', fixture)
        metadata.update(status='generated_pending_visual_review', output_sha256=sha256(target / 'facade.png'))
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata.update(elapsed_seconds=time.monotonic() - started,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
        write_json(target / 'run.json', metadata)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'download', 'generate'])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/phase3')
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--offload', choices=['cpu', 'disk'], default='cpu')
    parser.add_argument('--disk-dir', type=Path, default=Path.home() / '.cache/archigen/sdxl_disk_offload')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'outputs/phase2/viewer_fixture.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/cmp512/manifest.json')
    parser.add_argument('--case', default='viewer')
    parser.add_argument('--style', choices=list(STYLE_PROMPTS), default='neutral')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--control-scale', type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 0 < args.control_scale <= 2:
        parser.error('steps must be 1..100 and control-scale must be >0 and <=2')
    if args.action == 'prepare':
        result = prepare_cases(args.output, args.fixture, args.manifest)
        print(f"Prepared {len(result['cases'])} cases at {args.output}; no generation performed")
        return 0
    config = json.loads((ROOT / 'configs/facade_diffusion.json').read_text())
    paths = cached_models(config, args.cache)
    report = resource_report(args.cache, download=args.action == 'download', cached=paths is not None,
                             disk_dir=args.disk_dir if args.offload == 'disk' else None)
    report['weights_cached'] = paths is not None
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'preflight.json', report)
    print(json.dumps(report, indent=2))
    if not report['ready']:
        return 2
    if args.action == 'download':
        download(config, args.cache)
    elif args.action == 'generate':
        if paths is None:
            raise RuntimeError('Weights are absent; run the explicit download action first')
        generate(args, config, paths)
    return 0


if __name__ == '__main__':
    sys.exit(main())
