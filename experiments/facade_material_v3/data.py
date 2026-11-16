"""Training-only wall references and real annotated opening-free crops."""
import hashlib

import cv2
import numpy as np
from PIL import Image

from experiments.facade_semantic_v2.conditioning import letterbox, control_channels
from experiments.facade_semantic_v2.data import FacadeDataset
from experiments.facade_sd15_controlnet.data import digest, write_json


def wall_patch(photo, labels, rng, minimum=32, maximum=128):
    """Return a truly wall-only square; never erase an opening from a label map."""
    wall = (labels == 1).astype(np.uint8)
    # Explicit zero border avoids treating the edge of an all-wall image as infinity.
    distance = cv2.distanceTransform(np.pad(wall, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    ys, xs = np.where(distance >= minimum / np.sqrt(2) + 1)
    if not len(xs): return None
    at = int(rng.integers(len(xs))); x, y = int(xs[at]), int(ys[at])
    half = min(maximum // 2, int((distance[y, x] - 1) / np.sqrt(2)))
    box = (x-half, y-half, x+half, y+half)
    crop_labels = labels[box[1]:box[3], box[0]:box[2]]
    if crop_labels.size == 0 or not (crop_labels == 1).all():
        raise ValueError('Wall reference intersects a non-wall annotation')
    return photo.crop(box), crop_labels.copy(), box


def material_reference(photo, labels, rng):
    chosen = wall_patch(photo, labels, rng)
    if chosen is None:
        pixels = np.asarray(photo)[labels == 1]
        if not len(pixels): raise ValueError('No annotated wall pixels for material reference')
        tile = Image.new('RGB', (64, 64), tuple(np.median(pixels, axis=0).astype(int)))
        fallback = True
    else:
        tile = chosen[0].resize((128, 128), Image.Resampling.LANCZOS); fallback = False
    pixels = np.tile(np.asarray(tile), (int(np.ceil(labels.shape[0]/tile.height)), int(np.ceil(labels.shape[1]/tile.width)), 1))[:labels.shape[0], :labels.shape[1]].copy()
    return Image.fromarray(pixels), fallback


class MaterialDataset(FacadeDataset):
    def __init__(self, *args, blank_probability=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.blank_probability = blank_probability if self.augment else 0

    def sample(self, index, reference_photo=None, reference_labels=None, force_blank=False, include_reference_info=False):
        row = self.rows[index]
        rng = np.random.default_rng(int(hashlib.sha256(f'v3:{self.epoch}:{row["id"]}'.encode()).hexdigest()[:8], 16))
        blank = False
        if force_blank or (self.blank_probability and rng.random() < self.blank_probability):
            with Image.open(self.sources[row['source']] / row['photo']) as source: raw = source.convert('RGB')
            labels = np.asarray(Image.open(self.root / row['labels']))
            chosen = wall_patch(raw, labels, rng, minimum=64, maximum=256)
            if chosen:
                photo, labels, valid = letterbox(chosen[0], chosen[1]); blank = True
        if not blank:
            photo, labels, valid, _ = super().images(index)
        if (reference_photo is None) != (reference_labels is None):
            raise ValueError('Provide both reference photo and its human annotation')
        external = reference_photo is not None
        reference_photo = reference_photo if external else photo
        reference_labels = reference_labels if external else labels
        from_full = False
        # A legitimate storey crop can contain only windows/doors. The same
        # building's full human-labelled photo may still provide wall material.
        if not external and not (reference_labels == 1).any():
            with Image.open(self.sources[row['source']] / row['photo']) as source:
                reference_photo = source.convert('RGB')
            reference_labels = np.asarray(Image.open(self.root / row['labels']))
            from_full = bool((reference_labels == 1).any())
        available = bool((reference_labels == 1).any())
        if available:
            reference, fallback = material_reference(reference_photo, reference_labels, rng)
        else:
            # Missing reference is explicit conditioning dropout, never a fake
            # wall label or a material extracted from glass/vegetation.
            reference, fallback = Image.new('RGB', photo.size, (0, 0, 0)), False
        if reference.size != photo.size: reference = reference.resize(photo.size)
        # Padding must follow the target layout, even for independent references.
        pixels = np.asarray(reference).copy()
        if available: pixels[labels == 0] = 127
        sample = (photo, labels, valid, control_channels(labels), Image.fromarray(pixels), blank, fallback)
        info = {'available': available, 'from_full_photo': from_full, 'external': external}
        return (*sample, info) if include_reference_info else sample

    def __getitem__(self, index):
        import torch
        photo, labels, valid, semantic, reference, blank, fallback, info = self.sample(index, include_reference_info=True)
        ref = torch.from_numpy(np.asarray(reference, np.float32).transpose(2, 0, 1).copy()) / 255
        return {'photo': torch.from_numpy(np.asarray(photo, np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1,
            'condition': torch.cat([torch.from_numpy(semantic), ref]), 'reference': ref,
            'labels': torch.from_numpy(labels.astype(np.int64)), 'valid': torch.from_numpy(valid),
            'blank_crop': blank, 'reference_median_fallback': fallback,
            'reference_available': info['available'], 'reference_from_full_photo': info['from_full_photo'],
            'id': self.rows[index]['id']}


def audit_material_samples(root, output, crop_probability=.5, blank_probability=.2):
    """Exercise every train/validation sample before any GPU optimizer steps."""
    from pathlib import Path
    import json
    root, output = Path(root), Path(output)
    contract = {'manifest_sha256': digest(root / 'manifest.json'), 'loader_sha256': digest(Path(__file__)),
        'crop_probability': crop_probability, 'blank_probability': blank_probability, 'epoch': 0}
    if output.is_file():
        previous = json.loads(output.read_text())
        if previous.get('contract') == contract: return previous
    report = {'contract': contract, 'splits': {}, 'test_split_accessed': False, 'reference_fallback_samples': []}
    for split in ('train', 'val'):
        data = MaterialDataset(root, split, augment=split == 'train', crop_probability=crop_probability,
            blank_probability=blank_probability)
        counts = {'samples': 0, 'full_photo_references': 0, 'unavailable_references': 0, 'median_references': 0, 'blank_crops': 0}
        for i, row in enumerate(data.rows):
            try:
                photo, labels, valid, semantic, reference, blank, fallback, info = data.sample(i, include_reference_info=True)
            except (ValueError, OSError) as error:
                raise ValueError('Material data audit failed for ' + row['id']) from error
            if photo.size != (512, 512) or reference.size != photo.size or labels.shape != valid.shape or semantic.shape != (7,512,512):
                raise ValueError('Unaligned material sample: ' + row['id'])
            if blank and not (labels[valid > 0] == 1).all(): raise ValueError('Invalid real blank crop: ' + row['id'])
            counts['samples'] += 1; counts['full_photo_references'] += int(info['from_full_photo'])
            counts['unavailable_references'] += int(not info['available']); counts['median_references'] += int(fallback)
            counts['blank_crops'] += int(blank)
            if info['from_full_photo'] or not info['available']:
                report['reference_fallback_samples'].append({'id': row['id'], 'split': split, **info})
            if (i+1)%200==0: print(f'Material data audit {split}: {i+1}/{len(data)}',flush=True)
        report['splits'][split] = counts
    write_json(output, report)
    print('MATERIAL DATA AUDIT PASSED:', json.dumps(report['splits']), flush=True)
    return report
