"""Fixed-seed paired evaluation. LPIPS/SSIM/PSNR are not guarantees of geometry."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data import FacadePairs, CMP_TRAINING_LUT, read_labels, digest, write_json


@lru_cache(maxsize=1)
def perceptual_model():
    import lpips
    model = lpips.LPIPS(net='alex').cpu().eval()
    model.requires_grad_(False)
    return model


def image_metrics(generated, target, openings, perceptual=True):
    from skimage.metrics import structural_similarity
    left, right = np.asarray(generated.convert('RGB')), np.asarray(target.convert('RGB'))
    error = ((left.astype(np.float32) - right) / 255) ** 2
    edge_left = cv2.Canny(left, 100, 200).astype(np.float32) / 255
    edge_right = cv2.Canny(right, 100, 200).astype(np.float32) / 255
    region = cv2.dilate(np.asarray(openings), np.ones((5, 5), np.uint8)) > 0
    result = {'psnr': float(-10 * np.log10(max(float(error.mean()), 1e-10))),
              'ssim': float(structural_similarity(left, right, channel_axis=2, data_range=255)),
              'opening_edge_mae': float(np.abs(edge_left - edge_right)[region].mean()) if region.any() else 0.0}
    if perceptual:
        import torch
        tensor = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1).copy())[None] / 127.5 - 1
        with torch.no_grad(): result['lpips'] = float(perceptual_model()(tensor(left), tensor(right)).item())
    return result


def comparison(condition, generated, target, path, baseline=None):
    panels = [('Semantic edges', condition), ('SD1.5 ControlNet candidate', generated), ('Original CMP target', target)]
    if baseline is not None: panels.insert(1, ('Production Pix2Pix raw 256', baseline))
    canvas = Image.new('RGB', (512 * len(panels), 544), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        canvas.paste(image.resize((512, 512)), (index * 512, 32)); draw.text((index * 512 + 8, 8), label, fill='black')
    Path(path).parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.split == 'test' and not args.final_test: parser.error('Use --final-test only after selecting the model on validation data.')
    from .runtime import FacadeControlNetRuntime
    from .tracking import RunTracker
    runtime = FacadeControlNetRuntime(args.model, allow_download=True)
    data = FacadePairs(args.data, args.split)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(output, runtime.config['experiment_name'] + '-Evaluation',
        {'split': args.split, 'seed': runtime.config['seed'], 'inference_steps': runtime.config['inference_steps'],
         'manifest_sha256': digest(Path(args.data) / 'manifest.json')},
        {'model_training_run_id': runtime.metadata['run_id'], 'git_commit': runtime.metadata['git_commit']})
    baseline = None
    if data.manifest.get('baseline'):
        import onnxruntime as ort
        baseline_path = Path(args.data) / data.manifest['baseline']['path']
        if digest(baseline_path) != data.manifest['baseline']['sha256']: raise ValueError('Baseline checksum mismatch')
        baseline = ort.InferenceSession(str(baseline_path), providers=['CPUExecutionProvider'])
    rows = []
    try:
        for index in range(min(args.limit or len(data), len(data))):
            target, condition, openings = data.images(index)
            started = time.perf_counter()
            generated = runtime.generate_control(condition, seed=runtime.config['seed'] + index)
            row = {'id': data.rows[index]['id'], 'diffusion_seconds': time.perf_counter() - started,
                   **{f'candidate_{k}': v for k, v in image_metrics(generated, target, openings).items()}}
            previous = None
            if baseline:
                labels = read_labels(Path(args.data) / data.rows[index]['labels'])
                sketch = Image.fromarray(CMP_TRAINING_LUT[labels]).resize((256, 256), Image.Resampling.NEAREST)
                array = (np.asarray(sketch, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5) - 1
                prediction = baseline.run(None, {baseline.get_inputs()[0].name: array})[0]
                previous = Image.fromarray(np.clip((prediction[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)).resize((512, 512), Image.Resampling.BICUBIC)
                row.update({f'baseline_{k}': v for k, v in image_metrics(previous, target, openings).items()})
            if index < 8:
                path = output / f'comparison_{index:03d}.png'; comparison(condition, generated, target, path, previous); tracker.artifact(path, 'comparisons')
            rows.append(row); write_json(output / 'per-image.json', rows)
        means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'id'}
        report = {'split': args.split, 'n': len(rows), 'means': means,
            'protocol': '512 paired evaluation; baseline raw 256 output resized with bicubic; fixed seed per sorted ID. Opening edge MAE is a proxy, not opening IoU.'}
        if baseline:
            deltas = np.array([row['candidate_lpips'] - row['baseline_lpips'] for row in rows])
            rng = np.random.default_rng(42)
            bootstrap = np.mean(rng.choice(deltas, size=(2000, len(deltas)), replace=True), axis=1)
            report['paired_lpips_delta_ci95'] = np.quantile(bootstrap, [.025, .975]).tolist()
        write_json(output / 'evaluation.json', report)
        tracker.metrics(means, 0); tracker.artifact(output / 'evaluation.json'); tracker.artifact(output / 'per-image.json')
        tracker.finish(); print(json.dumps(report, indent=2))
    except BaseException:
        tracker.finish('FAILED'); raise


if __name__ == '__main__': main()
