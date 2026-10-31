"""Match original CMP photos to existing Pix2Pix splits; never invent style labels.

Download base/extended archives from https://cmp.felk.cvut.cz/~tylecr1/facade/
and extract to data/raw/cmp_original before running. Source photos remain untouched.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.facade_palette import CMP_TRAINING_LUT

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def signature(image):
    return np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).flatten() / 255


def prepare(originals, paired, output):
    original_paths = sorted(Path(originals).rglob('*.jpg'))
    if not original_paths: raise ValueError('No original CMP photographs')
    photos, sizes = [], []
    for path in original_paths:
        with Image.open(path) as im:
            photos.append(signature(im.convert('RGB'))); sizes.append(im.size)
    old_paths = [p for split in ('train', 'val', 'test') for p in sorted((Path(paired)/split).glob('*.jpg'))]
    old = []
    for p in old_paths:
        with Image.open(p) as im: old.append(signature(im.crop((0, 0, im.width//2, im.height)).convert('RGB')))
    distances = cdist(np.array(old), np.array(photos), metric='sqeuclidean') / len(photos[0])
    rows, rejected, seen = [], [], {}
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    for i, old_path in enumerate(old_paths):
        ranking = np.argsort(distances[i]); j = int(ranking[0])
        error, runner_up = float(distances[i, j]), float(distances[i, ranking[1]])
        original = original_paths[j]
        # Require close and unambiguous visual identity, not a nearest-neighbour guess.
        if error > .003 or runner_up < max(error * 3, .0001):
            rejected.append({'id': f'{old_path.parent.name}/{old_path.name}', 'mse': error, 'runner_up': runner_up})
            continue
        if original.as_posix() in seen:
            raise ValueError(f'Duplicate source across existing pairs: {original} / {seen[original.as_posix()]} / {old_path}')
        seen[original.as_posix()] = str(old_path)
        mask_path = original.with_suffix('.png')
        with Image.open(mask_path) as im:
            labels = np.asarray(im)
            if labels.ndim != 2 or labels.max() > 12:
                raise ValueError(f'Unexpected CMP label encoding: {mask_path}')
            mask = Image.fromarray(CMP_TRAINING_LUT[labels])
            dest = root/f'{original.stem}_sketch.png'; mask.save(dest)
        rows.append({'id': f'{old_path.parent.name}/{old_path.name}', 'split': old_path.parent.name,
                     'photo': original.as_posix(), 'sketch': dest.as_posix(), 'style_id': 4,
                     'native_size': list(sizes[j]), 'match_mse': error,
                     'photo_sha256': hashlib.sha256(original.read_bytes()).hexdigest()})
    document = {'source': 'https://cmp.felk.cvut.cz/~tylecr1/facade/', 'license': 'CC BY-SA',
                'style_status': 'unlabelled; neutral id 4; four named styles unsupported',
                'matching': '32px RGB identity, MSE <= .003 and >=3x runner-up separation',
                'splits': dict(Counter(r['split'] for r in rows)),
                'native_both_axes_at_least_512': sum(min(r['native_size']) >= 512 for r in rows),
                'rejected': rejected, 'samples': rows}
    (root/'manifest.json').write_text(json.dumps(document, indent=2))
    print(json.dumps({k: v for k, v in document.items() if k not in ('samples', 'rejected')}, indent=2))
    print('Rejected:',len(rejected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--originals', default='data/raw/cmp_original')
    parser.add_argument('--paired', default='data/raw/facades')
    parser.add_argument('--output', default='data/processed/cmp512')
    args = parser.parse_args(); prepare(args.originals, args.paired, args.output)
