"""Matched V1/V2 protocol; padding is excluded, held-out tests remain untouched."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.facade_sd15_controlnet.data import write_json, digest
from experiments.facade_sd15_controlnet.evaluate import image_metrics
from experiments.facade_sd15_controlnet.tracking import RunTracker
from .conditioning import DISPLAY_RGB
from .data import FacadeDataset, selection_indices

PROTOCOL = 'v2-facade-bbox-neutral-background-letterbox-valid-rectangle; identical prompt/seed/steps; validation only'


def pair_metrics(generated, photo, labels, valid, perceptual=True):
    y, x = np.where(valid > 0)
    box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    openings = Image.fromarray((((labels == 2) | (labels == 3)) * 255).astype('uint8'))
    return image_metrics(generated.crop(box), photo.crop(box), openings.crop(box), perceptual=perceptual)


def panel(photo, labels, generated, path, parent=None):
    panels = [('Semantic labels', Image.fromarray(DISPLAY_RGB[labels])), ('V2 semantic + LoRA', generated), ('Target', photo)]
    if parent is not None:
        panels.insert(1, ('V1 parent (matched protocol)', parent))
    canvas = Image.new('RGB', (512 * len(panels), 544), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        canvas.paste(image.resize((512, 512)), (index * 512, 32))
        draw.text((index * 512 + 8, 8), title, fill='black')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def summarize(rows):
    result = {}
    keys = [key for key in rows[0] if key.startswith(('candidate_', 'parent_'))]
    for source in ('CMP', 'IRF'):
        selected = [row for row in rows if row['source'] == source]
        if not selected:
            continue
        result[source] = {'n': len(selected), 'means': {key: float(np.mean([row[key] for row in selected])) for key in keys}}
        if 'parent_lpips' in keys:
            deltas = np.array([row['candidate_lpips'] - row['parent_lpips'] for row in selected])
            rng = np.random.default_rng(42)
            result[source]['lpips_delta_ci95'] = np.quantile(rng.choice(deltas, (2000, len(deltas))).mean(1), [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', required=True); parser.add_argument('--parent', required=True)
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', action='store_true', help='Evaluate fixed 12 CMP + 4 IRF selection set only')
    args = parser.parse_args()
    import torch
    from .runtime import SemanticFacadeRuntime
    from experiments.facade_sd15_controlnet.runtime import FacadeControlNetRuntime
    data = FacadeDataset(args.data, 'val')
    indices = selection_indices(data) if args.pilot else list(range(len(data)))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(output, 'ArchiGen-Facade-Semantic7-V2-Evaluation',
        {'protocol': PROTOCOL, 'manifest_sha256': digest(Path(args.data) / 'manifest.json'), 'n': len(indices)},
        {'quality_status': 'pilot selection set, not independent test' if args.pilot else 'full validation; not held-out test'})
    rows = []
    try:
        # Separate model lifetimes avoid two resident frozen base pipelines on a laptop.
        for name, runtime in [('parent', FacadeControlNetRuntime(args.parent, allow_download=True)),
                              ('candidate', SemanticFacadeRuntime(args.model, allow_download=True))]:
            if name == 'parent':
                candidate_config = json.loads((Path(args.model) / 'model.json').read_text())['config']
                for key in ('prompt', 'negative_prompt', 'inference_steps', 'guidance_scale', 'controlnet_conditioning_scale', 'seed'):
                    runtime.config[key] = candidate_config[key]
            for ordinal, index in enumerate(indices):
                photo, labels, valid, channels = data.images(index)
                control = torch.from_numpy(channels)[None] if name == 'candidate' else Image.fromarray((channels[0] * 255).astype('uint8')).convert('RGB')
                generated = runtime.generate_control(control, seed=42 + index)
                generated.save(output / f'{name}_{ordinal:03d}.png')
                values = pair_metrics(generated, photo, labels, valid)
                if name == 'parent':
                    rows.append({'id': data.rows[index]['id'], 'source': data.rows[index]['source']})
                rows[ordinal].update({f'{name}_{key}': value for key, value in values.items()})
                write_json(output / 'per-image.json', rows)
                if name == 'candidate' and ordinal < 8:
                    path = output / f'comparison_{ordinal:03d}.png'
                    panel(photo, labels, generated, path, Image.open(output / f'parent_{ordinal:03d}.png'))
                    tracker.artifact(path, 'comparisons')
                print(json.dumps({'evaluation': name, 'completed': ordinal + 1, 'n': len(indices)}), flush=True)
            # The list contains runtime objects but not pipelines after this explicit release.
            if runtime.pipeline is not None:
                runtime.pipeline.maybe_free_model_hooks()
                runtime.pipeline = None
            gc.collect(); torch.cuda.empty_cache()
        report = {'protocol': PROTOCOL, 'pilot': args.pilot, 'by_source': summarize(rows),
                  'caveat': 'Opening-edge MAE is an appearance proxy, not semantic opening IoU. V1 published square-resize metrics are not comparable.'}
        write_json(output / 'evaluation.json', report)
        for source, values in report['by_source'].items():
            tracker.metrics({source + '_' + key: value for key, value in values['means'].items()}, 0)
        tracker.artifact(output / 'evaluation.json'); tracker.artifact(output / 'per-image.json'); tracker.finish()
    except BaseException:
        tracker.finish('FAILED'); raise


if __name__ == '__main__': main()
