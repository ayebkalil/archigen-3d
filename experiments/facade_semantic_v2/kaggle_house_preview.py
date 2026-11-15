"""Generate the actual local house panels on Kaggle; no training or promotion."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import threading
import time
import zipfile

from experiments.facade_sd15_controlnet.data import digest, write_json
from experiments.facade_sd15_controlnet.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case-sha256', required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
    from .runtime import validate_export
    from .conditioning import cmp_labels, control_channels
    from experiments.facade_sd15_controlnet.data import labels_from_bridge
    from src.models.facade_diffusion import gpu_memory
    cases = list(Path('/kaggle/input').rglob('house-input.zip'))
    if len(cases) != 1 or digest(cases[0]) != args.case_sha256:
        raise ValueError('Attach exactly the pinned house-case archive')
    output = Path('/kaggle/working/house-preview'); output.mkdir(exist_ok=True)
    with zipfile.ZipFile(cases[0]) as archive:
        for member in archive.infolist():
            if not (output / member.filename).resolve().is_relative_to(output.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Invalid case archive path')
        archive.extractall(output)
    exports = [p.parent for p in Path('/kaggle/input').rglob('model.json') if p.parent.name == 'export'
               and json.loads(p.read_text()).get('weights_sha256') == '07c7eab625e775b88ca4f9a099f661b7e1dfe9decfbff5bbf648a218e0466d1f']
    if len(exports) != 1:
        raise ValueError('Attach exactly the completed V2 training export')
    root = exports[0]; metadata = validate_export(root); config = metadata['config']
    controlnet = ControlNetModel.from_pretrained(root / 'controlnet', torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(config['base_model'], revision=config['base_revision'],
        variant='fp16', torch_dtype=torch.float16, use_safetensors=True, controlnet=controlnet,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.load_lora_weights(str(root / 'lora'), weight_name='facade_lora.safetensors')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda'); pipe.enable_vae_slicing(); pipe.enable_vae_tiling(); pipe.set_progress_bar_config(disable=True)
    panels = json.loads((output / 'panels.json').read_text())
    report = {'architecture': metadata['architecture'], 'checkpoint_step': metadata['selected_step'],
        'weights_sha256': metadata['weights_sha256'], 'lora_sha256': metadata['lora_sha256'],
        'device': 'Kaggle ' + torch.cuda.get_device_name(0), 'inference_mode': 'full GPU FP16; not a laptop benchmark',
        'inference_steps': config['inference_steps'], 'seed': config['seed'], 'prompt': config['prompt'], 'style': None,
        'case_archive_sha256': args.case_sha256, 'panels': []}
    tracker = RunTracker(output, 'ArchiGen-Facade-V2-House-Multiface',
        {k: v for k, v in report.items() if k != 'panels'}, {'model_training_run_id': metadata['run_id']})
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.is_set():
            samples.append(gpu_memory()['used_mib']); stop.wait(.5)
    thread = threading.Thread(target=monitor, daemon=True); thread.start(); started = time.perf_counter()
    try:
        for i, spec in enumerate(panels):
            folder = output / spec['wall_id']
            if digest(folder / 'sketch.png') != spec['sketch_sha256']:
                raise ValueError('House sketch checksum changed')
            labels = cmp_labels(labels_from_bridge(Image.open(folder / 'sketch.png').convert('RGB')))
            condition = torch.from_numpy(control_channels(labels))[None]
            torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                image = pipe(config['prompt'], image=condition, negative_prompt=config['negative_prompt'],
                    height=512, width=512, num_inference_steps=config['inference_steps'], guidance_scale=config['guidance_scale'],
                    controlnet_conditioning_scale=config['controlnet_conditioning_scale'],
                    generator=torch.Generator(device='cpu').manual_seed(config['seed'])).images[0]
            image.save(folder / 'candidate.png')
            result = {**spec, 'seconds': time.perf_counter() - start, 'seed': config['seed'],
                'peak_torch_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                'weights_sha256': metadata['weights_sha256'], 'lora_sha256': metadata['lora_sha256']}
            write_json(folder / 'candidate.json', result); report['panels'].append(result)
            tracker.metrics({'generation_seconds': result['seconds'], 'peak_allocated_mib': result['peak_torch_allocated_mib']}, i)
            tracker.artifact(folder / 'candidate.png', 'walls/' + spec['wall_id'])
            print(json.dumps({'generated': spec['wall_id'], 'completed': i + 1, 'total': len(panels)}), flush=True)
        report.update(complete=True, total_seconds=time.perf_counter() - started,
                      sampled_peak_device_mib=max(samples), measurement_note='Device-wide VRAM sampled every 0.5 s on Kaggle; not local profiling.')
        write_json(output / 'profile.json', report); tracker.artifact(output / 'profile.json'); tracker.finish()
        with zipfile.ZipFile('/kaggle/working/archigen-house-v2-preview.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for pattern in ('*/candidate.png', '*/candidate.json', 'profile.json', 'tracking-events.jsonl', 'tracking-run.json', 'portable-artifacts/**/*'):
                for file in output.glob(pattern):
                    if file.is_file(): archive.write(file, file.relative_to(output).as_posix())
        print('HOUSE PREVIEW COMPLETE', flush=True)
    except BaseException:
        tracker.finish('FAILED'); raise
    finally:
        stop.set(); thread.join(timeout=10)


if __name__ == '__main__': main()
