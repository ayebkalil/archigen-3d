"""Generate one real house-wall texture offline and record laptop resource use."""
import argparse
import base64
import json
from pathlib import Path
import threading
import time

from .runtime import FacadeControlNetRuntime
from src.models.facade_diffusion import gpu_memory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    output = args.project.resolve() / 'outputs/facade_house_test'
    probe = json.loads((output / 'probe.json').read_text(encoding='utf-8'))
    stopped = threading.Event()
    samples = []

    def monitor():
        while not stopped.is_set():
            try:
                samples.append({'elapsed': time.perf_counter(), **gpu_memory()})
            except Exception:
                pass
            stopped.wait(.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    try:
        runtime = FacadeControlNetRuntime(args.model, allow_download=False)
        runtime._load()
        load_seconds = time.perf_counter() - started
        print(f'Local model loaded in {load_seconds:.1f} seconds; generating 25 diffusion steps.', flush=True)
        runtime.pipeline.set_progress_bar_config(disable=False)
        image = runtime.generate_sketch((output / 'sketch.png').read_bytes(), style=None)
        # Allow one last device-wide observation after CUDA work completes.
        samples.append({'elapsed': time.perf_counter(), **gpu_memory()})
        peak = max(sample['used_mib'] for sample in samples)
        profile = {**runtime.last_profile, 'load_seconds': load_seconds,
                   'sampled_peak_device_mib': peak, 'device_sample_count': len(samples),
                   'sample_interval_seconds': .5, 'target_budget_mib': 3500,
                   'budget_passed_at_samples': peak <= 3500,
                   'measurement_note': 'Sampled device peak is not a guarantee of the instantaneous peak.',
                   'wall_id': probe['wall_id'], 'checkpoint_step': runtime.metadata['selected_step'],
                   'weights_sha256': runtime.metadata['weights_sha256']}
        (output / 'profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
        if peak > 3500:
            raise RuntimeError('Sampled device memory exceeded the target; this test is not accepted.')
        image.save(output / 'candidate.png')
        candidate = {**probe, 'model_info': runtime.describe(), 'device': 'Local RTX 3050 / FP16 / sequential CPU offload',
                     'texture_base64': 'data:image/png;base64,' + base64.b64encode((output / 'candidate.png').read_bytes()).decode(),
                     'sketch_base64': 'data:image/png;base64,' + base64.b64encode((output / 'sketch.png').read_bytes()).decode()}
        (output / 'candidate.json').write_text(json.dumps(candidate), encoding='utf-8')
        print(json.dumps(profile), flush=True)
    finally:
        stopped.set()
        monitor_thread.join(timeout=12)


if __name__ == '__main__':
    main()
