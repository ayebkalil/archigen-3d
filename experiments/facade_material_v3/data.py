"""Training-only wall references and real annotated opening-free crops."""
import hashlib

import cv2
import numpy as np
from PIL import Image

from experiments.facade_semantic_v2.conditioning import letterbox, control_channels
from experiments.facade_semantic_v2.data import FacadeDataset


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

    def sample(self, index, reference_photo=None, reference_labels=None, force_blank=False):
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
        reference, fallback = material_reference(reference_photo if reference_photo is not None else photo,
            reference_labels if reference_labels is not None else labels, rng)
        if reference.size != photo.size: reference = reference.resize(photo.size)
        # Padding must follow the target layout, even for independent references.
        pixels = np.asarray(reference).copy(); pixels[labels == 0] = 127
        return photo, labels, valid, control_channels(labels), Image.fromarray(pixels), blank, fallback

    def __getitem__(self, index):
        import torch
        photo, labels, valid, semantic, reference, blank, fallback = self.sample(index)
        ref = torch.from_numpy(np.asarray(reference, np.float32).transpose(2, 0, 1).copy()) / 255
        return {'photo': torch.from_numpy(np.asarray(photo, np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1,
            'condition': torch.cat([torch.from_numpy(semantic), ref]), 'reference': ref,
            'labels': torch.from_numpy(labels.astype(np.int64)), 'valid': torch.from_numpy(valid),
            'blank_crop': blank, 'reference_median_fallback': fallback, 'id': self.rows[index]['id']}
