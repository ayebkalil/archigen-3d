"""Portable original-resolution CMP pairs, locked splits, identical conditioning."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

from src.data.facade_palette import CMP_TRAINING_LUT


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_labels(path):
    with Image.open(path) as image:
        labels = np.asarray(image).copy()
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 12:
        raise ValueError(f'Expected original CMP index labels 0..12: {path}')
    return labels.astype(np.uint8)


def condition_from_labels(labels, resolution=512):
    labels = np.asarray(Image.fromarray(labels).resize((resolution, resolution), Image.Resampling.NEAREST)).copy()
    labels[labels == 0] = 1  # The bridge maps both background indices to the same colour.
    edge = np.zeros_like(labels)
    # Semantic boundaries only, never edges taken from the target photograph.
    for value in np.unique(labels):
        edge |= cv2.Canny((labels == value).astype(np.uint8) * 255, 100, 200)
    openings = np.isin(labels, [3, 4]).astype(np.uint8) * 255
    return Image.fromarray(edge).convert('RGB'), Image.fromarray(openings)


def labels_from_bridge(image):
    pixels = np.asarray(image.convert('RGB'), dtype=np.int32)
    colours = CMP_TRAINING_LUT.astype(np.int32)
    distances = ((pixels[:, :, None] - colours[None, None]) ** 2).sum(axis=-1)
    return distances.argmin(axis=-1).astype(np.uint8)


def package_dataset(source_manifest, source_root, output, owner='khalilayeb', baseline=None):
    """Copy existing matched originals. Never reshuffle, relabel styles or recompress photos."""
    source_manifest, source_root, output = Path(source_manifest), Path(source_root), Path(output)
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    rows, seen_ids, seen_hashes = [], set(), set()
    for row in source['samples']:
        if row['id'] in seen_ids or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Duplicate ID or invalid split in source manifest')
        photo = source_root / row['photo']
        labels = photo.with_suffix('.png')
        photo_hash = digest(photo)
        if photo_hash != row['photo_sha256'] or photo_hash in seen_hashes:
            raise ValueError(f'Changed source or duplicate photo across dataset: {photo}')
        seen_ids.add(row['id']); seen_hashes.add(photo_hash)
        label_array = read_labels(labels)
        with Image.open(photo) as image:
            if image.size != tuple(label_array.shape[::-1]):
                raise ValueError(f'Unaligned photograph and labels: {photo}')
            native_size = list(image.size)
        stem = row['id'].replace('/', '__').rsplit('.', 1)[0]
        dest_photo, dest_labels = output / 'photos' / (stem + '.jpg'), output / 'labels' / (stem + '.png')
        dest_photo.parent.mkdir(parents=True, exist_ok=True); dest_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(photo, dest_photo); shutil.copyfile(labels, dest_labels)
        rows.append({'id': row['id'], 'split': row['split'], 'photo': dest_photo.relative_to(output).as_posix(),
                     'labels': dest_labels.relative_to(output).as_posix(), 'photo_sha256': photo_hash,
                     'labels_sha256': digest(dest_labels), 'native_size': native_size})
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val', 'test')}
    document = {'schema_version': 1, 'samples': rows, 'splits': counts,
                'source_manifest_sha256': digest(source_manifest), 'source': source['source'],
                'license': source['license'], 'conditioning': 'cmp_semantic_edges_v1',
                'style_labels': None, 'style_status': 'No four-style labels; prompts are pretrained prior only.'}
    if baseline:
        dest = output / 'baseline' / 'generator.onnx'; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline, dest)
        document['baseline'] = {'path': dest.relative_to(output).as_posix(), 'sha256': digest(dest)}
    write_json(output / 'manifest.json', document)
    write_json(output / 'dataset-metadata.json', {'title': 'ArchiGen CMP Facades Original Locked Splits',
        'id': f'{owner}/archigen-cmp-facades-original-locked-splits', 'licenses': [{'name': 'other'}],
        'description': 'CMP Facade Database, CC BY-SA; see ATTRIBUTION.md and the original dataset documentation.'})
    (output / 'ATTRIBUTION.md').write_text(
        'CMP Facade Database, Radim Tylecek and Radim Sara (2013).\n'
        'https://cmp.felk.cvut.cz/~tylecr1/facade/ — CC BY-SA.\n'
        '606 original photos and indexed masks. Existing 400/100/106 split retained.\n', encoding='utf-8')
    return document


def validate_dataset(root, verify_hashes=True):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    ids, hashes = set(), set()
    for row in manifest['samples']:
        if row['id'] in ids or row['photo_sha256'] in hashes:
            raise ValueError('Duplicate sample or photo across splits')
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split')
        ids.add(row['id']); hashes.add(row['photo_sha256'])
        for key in ('photo', 'labels'):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Dataset path escapes its root')
            if verify_hashes and digest(path) != row[key + '_sha256']:
                raise ValueError(f'Dataset checksum mismatch: {path}')
        read_labels(root / row['labels'])
    return manifest


def locate_kaggle_dataset(input_root='/kaggle/input', working='/kaggle/working/cmp-data'):
    """Handle both expanded Kaggle folders and directory ZIPs uploaded by its CLI."""
    candidates = []
    for path in Path(input_root).rglob('manifest.json'):
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
            if doc.get('conditioning') == 'cmp_semantic_edges_v1' and doc.get('schema_version') == 1:
                candidates.append(path.parent)
        except (ValueError, OSError): pass
    if len(candidates) != 1: raise ValueError(f'Attach exactly one prepared CMP dataset; found {len(candidates)}')
    source = candidates[0]
    document = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    ready = (source / 'photos').is_dir() and (source / 'labels').is_dir()
    if document.get('baseline'): ready = ready and (source / document['baseline']['path']).is_file()
    if ready: return source
    destination = Path(working).resolve(); destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix != '.zip': shutil.copyfile(file, destination / file.name)
    for name in ('photos', 'labels', 'baseline'):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name, dirs_exist_ok=True)
            continue
        archive = source / f'{name}.zip'
        if not archive.exists(): continue
        target = destination / name; target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not (target / member.filename).resolve().is_relative_to(target): raise ValueError('Unsafe dataset archive path')
            zipped.extractall(target)
    return destination


class FacadePairs:
    """Torch tensors are imported lazily so packaging and validation stay lightweight."""
    def __init__(self, root, split, resolution=512, augment=False):
        self.root, self.resolution = Path(root), resolution
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.rows = sorted((r for r in self.manifest['samples'] if r['split'] == split), key=lambda r: r['id'])
        if not self.rows:
            raise ValueError(f'No samples in split {split}')
        self.augment = augment and split == 'train'
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def images(self, index):
        row = self.rows[index]
        photo = Image.open(self.root / row['photo']).convert('RGB').resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        labels = read_labels(self.root / row['labels'])
        condition, openings = condition_from_labels(labels, self.resolution)
        return photo, condition, openings

    def __getitem__(self, index):
        import torch
        photo, condition, openings = self.images(index)
        # Stable for a given sample/epoch, including when resuming mid-epoch.
        flip = int(hashlib.sha256(f'{self.epoch}:{self.rows[index]["id"]}'.encode()).hexdigest()[:8], 16) % 2
        if self.augment and flip:
            photo, condition, openings = (image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in (photo, condition, openings))
        to_tensor = lambda image: torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        return {'photo': to_tensor(photo) * 2 - 1, 'condition': to_tensor(condition),
                'openings': torch.from_numpy(np.array(openings, dtype=np.float32).copy())[None] / 255,
                'id': self.rows[index]['id']}
