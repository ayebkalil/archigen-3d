"""Case-study proxies and an unretouched comparison, never model-accuracy scores."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import write_json
from .materials import opening_mask


def assess(output):
    output = Path(output)
    panels = json.loads((output / 'panels.json').read_text())
    rows = []
    for panel in panels:
        size = 512
        blank = dict(panel, openings=[])
        wall = opening_mask(dict(blank, openings=[{'position_along_wall_m': panel['wall_length_m'] / 2,
            'width_m': panel['wall_length_m'] - .3, 'sill_m': .15,
            'height_m': panel['wall_height_m'] - .3}]), size, trim_m=0)
        wall &= ~opening_mask(panel, size, trim_m=.15)
        for mode in ('v2', 'guided', 'locked', 'base'):
            pixels = np.asarray(Image.open(output / panel['wall_id'] / (mode + '.png')).convert('RGB').resize((size, size)))
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            rows.append({'wall_id': panel['wall_id'], 'mode': mode, 'openings': len(panel['openings']),
                'wall_mean_srgb': pixels[wall].mean(axis=0).tolist(),
                'wall_sobel_mean': float(magnitude[wall].mean())})
    report = {'case': 'One bundled house, 15 panels; no held-out facade photos or human judgements',
        'metric_notes': ['Wall-colour dispersion is a texture-space consistency proxy, not realism or material recognition.',
            'Lower blank-wall gradient is not proof of fewer semantic false openings.',
            'Locked/base values are constrained by construction, not evidence of learned-model improvement.'],
        'modes': {}, 'panels': rows}
    for mode in ('v2', 'guided', 'locked', 'base'):
        selected = [row for row in rows if row['mode'] == mode]
        means = np.array([row['wall_mean_srgb'] for row in selected])
        report['modes'][mode] = {'wall_mean_srgb_dispersion': float(np.sqrt(((means - means.mean(0))**2).sum(1).mean())),
            'blank_wall_sobel_mean': float(np.mean([row['wall_sobel_mean'] for row in selected if not row['openings']]))}
    write_json(output / 'appearance-proxies.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor='#f6f2e9')
    for row, side in enumerate(('ground_east', 'ground_south_extension')):
        for col, (mode, title) in enumerate((('v2', 'V2 raw generation'), ('guided', 'Same weights + reference guidance'), ('locked', 'Shared material + hard opening support'))):
            axes[row, col].imshow(Image.open(output / side / (mode + '.png')))
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
    fig.suptitle('Facade consistency study: existing openings (top), solid wall (bottom)', fontsize=15)
    fig.text(.5, .025, 'Unretouched outputs. Reference guidance preserves structure but adds limited detail.\nRight column is compositing, not a retrained generator. No production promotion.', ha='center', fontsize=10)
    fig.subplots_adjust(top=.90, bottom=.10, hspace=.08, wspace=.04)
    fig.savefig(output / 'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(report['modes'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', required=True)
    assess(parser.parse_args().output)
