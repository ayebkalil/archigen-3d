"""CMP + original IRFs, protected splits, building groups and deployment-shaped pairs."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import digest, write_json, validate_dataset as validate_cmp
from .conditioning import cmp_labels, irf_labels, facade_crop, letterbox, control_channels


def perceptual_hash(image):
    pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), np.float32)
    low = cv2.dct(pixels)[:8, :8].copy()
    bits = low > np.median(low.ravel()[1:])
    bits[0, 0] = False
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), 'big')


def building_key(url, fallback):
    parsed = urlsplit(str(url or ''))
    if parsed.hostname and parsed.path:
        return parsed.hostname.lower() + parsed.path.rstrip('/')
    return 'missing-url:' + fallback.split('_')[0] + '_' + fallback.split('_')[1]


def split_group(group, seed=42):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if value < .8 else 'val' if value < .9 else 'test'


def case_metadata(path):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook['case information']
    iterator = sheet.iter_rows(values_only=True)
    headings = next(iterator)
    result = {}
    for values in iterator:
        row = dict(zip(headings, values))
        if row.get('Sample ID'):
            result[Path(str(row['Sample ID'])).stem] = row
    workbook.close()
    return result


def build_dataset(cmp_root, irf_root, output, phash_threshold=5):
    """Raw source mounts are read-only; only canonical masks/metadata are generated."""
    cmp_root, irf_root, output = Path(cmp_root).resolve(), Path(irf_root).resolve(), Path(output).resolve()
    cmp_manifest = validate_cmp(cmp_root)
    metadata_file = next(irf_root.rglob('sample_list.xlsx'))
    metadata = case_metadata(metadata_file)
    rows, excluded, cmp_hashes, cmp_fingerprints = [], [], set(), []
    source_roots = {'CMP': str(cmp_root), 'IRF': str(irf_root)}

    def add(source, sample_id, split, group, photo, canonical, annotation):
        sparse = (canonical > 0).sum() < canonical.size * .03
        if sparse and source != 'CMP':
            raise ValueError('Annotation has too little facade foreground')
        if sparse:
            excluded.append({'id': 'CMP/' + sample_id, 'reason': 'Sparse mask: retained in locked split, excluded from optimization'})
        mask = output / 'labels' / source / (sample_id.replace('/', '__') + '.png')
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canonical).save(mask, compress_level=3)
        count, _, stats, _ = cv2.connectedComponentsWithStats((canonical == 2).astype(np.uint8))
        windows = sum(stats[i, cv2.CC_STAT_AREA] >= 25 for i in range(1, count))
        with Image.open(photo) as image:
            size = list(image.size)
        rows.append({'id': source + '/' + sample_id, 'source': source, 'split': split, 'group': group,
            'photo': photo.relative_to(Path(source_roots[source])).as_posix(),
            'labels': mask.relative_to(output).as_posix(), 'photo_sha256': digest(photo),
            'labels_sha256': digest(mask), 'annotation_sha256': digest(annotation),
            'native_size': size, 'has_door': bool((canonical == 3).any()), 'window_count': int(windows),
            'training_eligible': not sparse})

    for row in cmp_manifest['samples']:
        photo, annotation = cmp_root / row['photo'], cmp_root / row['labels']
        with Image.open(photo) as image:
            fingerprint = perceptual_hash(image)
        cmp_hashes.add(row['photo_sha256']); cmp_fingerprints.append((row['id'], fingerprint))
        with Image.open(annotation) as labels:
            mapped = cmp_labels(np.asarray(labels))
        add('CMP', row['id'], row['split'], 'CMP/' + row['id'], photo, mapped, annotation)

    fingerprints, seen, group_parent = [], set(cmp_hashes), {}
    def find(group):
        group_parent.setdefault(group, group)
        if group_parent[group] != group:
            group_parent[group] = find(group_parent[group])
        return group_parent[group]
    def union(a, b):
        left, right = sorted((find(a), find(b)))
        group_parent[right] = left

    photos = sorted(p for p in irf_root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg'})
    for index, photo in enumerate(photos):
        sample_id = photo.stem
        try:
            photo_hash = digest(photo)
            if photo_hash in seen:
                excluded.append({'id': sample_id, 'reason': 'exact duplicate'}); continue
            annotation, human_json = photo.with_suffix('.png'), photo.with_suffix('.json')
            if not annotation.is_file() or not human_json.is_file():
                raise ValueError('Missing original human annotation pair')
            with Image.open(photo) as image:
                if min(image.size) < 192 or image.width * image.height < 100000:
                    raise ValueError('Insufficient original image resolution')
                fingerprint = perceptual_hash(image)
                size = image.size
            nearest_id, nearest_distance = min(((identity, (fingerprint ^ value).bit_count()) for identity, value in cmp_fingerprints), key=lambda item: item[1])
            if nearest_distance <= phash_threshold:
                excluded.append({'id': sample_id, 'reason': 'CMP near duplicate', 'cmp_id': nearest_id, 'hamming': nearest_distance}); continue
            with Image.open(annotation) as labels:
                if labels.size != size:
                    raise ValueError('Photo/mask dimensions differ')
                mapped = irf_labels(labels)
            human = json.loads(human_json.read_text(encoding='utf-8'))
            if (human.get('imageWidth'), human.get('imageHeight')) != size or not human.get('shapes'):
                raise ValueError('Human JSON does not match the photograph')
            if sample_id not in metadata:
                raise ValueError('Building provenance missing from sample list')
            case = metadata[sample_id]
            group = building_key(case.get('Case URL'), sample_id)
            find(group)
            for other_hash, other_group in fingerprints:
                if (fingerprint ^ other_hash).bit_count() <= phash_threshold:
                    union(group, other_group)
            fingerprints.append((fingerprint, group)); seen.add(photo_hash)
            add('IRF', sample_id, 'pending', group, photo, mapped, human_json)
        except (ValueError, OSError, KeyError) as error:
            excluded.append({'id': sample_id, 'reason': str(error)})
        if index % 100 == 0:
            print(f'IRFs audit {index}/{len(photos)}', flush=True)
    for row in rows:
        if row['source'] == 'IRF':
            row['group'] = find(row['group'])
            row['split'] = split_group(row['group'])
    counts = {source: {split: sum(row['source'] == source and row['split'] == split for row in rows)
                       for split in ('train', 'val', 'test')} for source in ('CMP', 'IRF')}
    if counts['CMP'] != cmp_manifest['splits']:
        raise ValueError('The original CMP split was changed')
    if min(counts['IRF'].values()) < 10:
        raise ValueError('Insufficient IRFs coverage after audit; do not train silently on a broken dataset')
    manifest = {'schema_version': 2, 'conditioning': 'facade_semantic7_v2', 'samples': rows, 'counts': counts,
        'splits': {split: sum(counts[source][split] for source in counts) for split in ('train', 'val', 'test')},
        'cmp_manifest_sha256': digest(cmp_root / 'manifest.json'), 'irf_metadata_sha256': digest(metadata_file),
        'irf_source': 'https://www.kaggle.com/datasets/liushuyuu/irregular-facades-irfs',
        'license': 'Academic research: IRFs/ArchDaily CC BY-NC-SA 4.0; CMP CC BY-SA; retain original attributions.',
        'style_labels': None, 'phash_threshold': phash_threshold,
        'dedup_note': 'Exact and pHash-near photos screened; building URLs group IRFs. Unrecognized cross-source views of the same building cannot be fully ruled out.'}
    if cmp_manifest.get('baseline'):
        manifest['baseline'] = {**cmp_manifest['baseline'], 'source': 'CMP'}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'sources.json', source_roots)
    write_json(output / 'audit.json', {'counts': counts, 'excluded': excluded,
        'irf_building_groups': len({r['group'] for r in rows if r['source'] == 'IRF'}),
        'native_short_axis_ge_512': sum(min(r['native_size']) >= 512 for r in rows)})
    write_json(output / 'irf-credits.json', {row['id']: metadata[row['id'].split('/', 1)[1]]
        for row in rows if row['source'] == 'IRF'})
    (output / 'ATTRIBUTION.md').write_text(
        'IRFs: Wei, Hu, Zhang and Liu (2024), doi:10.3390/buildings14092602.\n'
        'Material is based on original data provided by ArchDaily. Copyright belongs to ArchDaily and the architects/photographers credited in sample_list.xlsx.\n'
        'Academic research only; CC BY-NC-SA 4.0. Acknowledge ArchDaily in resulting work.\n'
        'CMP: Tylecek and Sara (2013), https://cmp.felk.cvut.cz/~tylecr1/facade/, CC BY-SA.\n', encoding='utf-8')
    print(json.dumps({'counts': counts, 'excluded': len(excluded)}), flush=True)
    return manifest


def validate_dataset(root, verify_hashes=True):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    sources = {key: Path(value).resolve() for key, value in json.loads((root / 'sources.json').read_text()).items()}
    if manifest.get('schema_version') != 2 or manifest.get('conditioning') != 'facade_semantic7_v2':
        raise ValueError('Wrong dataset schema/conditioning')
    groups, ids, hashes = {}, set(), set()
    for row in manifest['samples']:
        if row['id'] in ids or row['photo_sha256'] in hashes or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Duplicate sample or invalid split')
        ids.add(row['id']); hashes.add(row['photo_sha256'])
        key = row['source'] + ':' + row['group']
        if key in groups and groups[key] != row['split']:
            raise ValueError('Building crosses dataset splits')
        groups[key] = row['split']
        photo, labels = (sources[row['source']] / row['photo']).resolve(), (root / row['labels']).resolve()
        if not photo.is_relative_to(sources[row['source']]) or not labels.is_relative_to(root):
            raise ValueError('Dataset path escapes root')
        if verify_hashes and (digest(photo) != row['photo_sha256'] or digest(labels) != row['labels_sha256']):
            raise ValueError('Dataset content changed')
        with Image.open(labels) as image:
            canonical = np.asarray(image)
            if list(image.size) != row['native_size'] or canonical.ndim != 2 or canonical.max() > 5:
                raise ValueError('Invalid canonical mask dimensions/classes')
    counts = {source: {split: sum(row['source'] == source and row['split'] == split for row in manifest['samples'])
                       for split in ('train', 'val', 'test')} for source in sources}
    if counts != manifest['counts']:
        raise ValueError('Dataset counts differ from manifest')
    return manifest


class FacadeDataset:
    def __init__(self, root, split, augment=False, crop_probability=.5, source=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.sources = {key: Path(value) for key, value in json.loads((self.root / 'sources.json').read_text()).items()}
        self.rows = sorted((row for row in self.manifest['samples'] if row['split'] == split
            and (split != 'train' or row.get('training_eligible', True))
            and (not source or row['source'] == source)), key=lambda row: row['id'])
        if not self.rows:
            raise ValueError('Empty dataset split')
        self.augment, self.crop_probability, self.epoch = augment and split == 'train', crop_probability, 0

    def __len__(self):
        return len(self.rows)

    def images(self, index):
        row = self.rows[index]
        with Image.open(self.sources[row['source']] / row['photo']) as file:
            photo = file.convert('RGB')
        labels = np.asarray(Image.open(self.root / row['labels'])).copy()
        rng = np.random.default_rng(int(hashlib.sha256(f'{self.epoch}:{row["id"]}'.encode()).hexdigest()[:8], 16))
        photo, labels = facade_crop(photo, labels, rng if self.augment else None, self.crop_probability)
        # The intended product is the facade surface, not sky/black rectification borders.
        pixels = np.asarray(photo).copy(); pixels[labels == 0] = 127
        photo, labels, valid = letterbox(Image.fromarray(pixels), labels)
        if self.augment and rng.random() < .5:
            photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            labels, valid = np.fliplr(labels).copy(), np.fliplr(valid).copy()
        return photo, labels, valid, control_channels(labels)

    def __getitem__(self, index):
        import torch
        photo, labels, valid, condition = self.images(index)
        return {'photo': torch.from_numpy(np.asarray(photo, np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1,
            'condition': torch.from_numpy(condition), 'labels': torch.from_numpy(labels.astype(np.int64)),
            'valid': torch.from_numpy(valid), 'id': self.rows[index]['id']}

    def training_order(self, epoch, balanced=True):
        import torch
        generator = torch.Generator().manual_seed(42 + epoch)
        if not balanced:
            return torch.randperm(len(self), generator=generator).tolist()
        counts = {source: sum(row['source'] == source for row in self.rows) for source in ('CMP', 'IRF')}
        weights = torch.tensor([1 / counts[row['source']] for row in self.rows], dtype=torch.double)
        return torch.multinomial(weights, len(self), replacement=True, generator=generator).tolist()


def selection_indices(dataset, count_cmp=12, count_irf=4):
    selected = []
    for source, target in [('CMP', count_cmp), ('IRF', count_irf)]:
        buckets = {}
        for index, row in enumerate(dataset.rows):
            if row['source'] != source:
                continue
            bucket = (row['has_door'], min(row['window_count'] // 10, 2))
            buckets.setdefault(bucket, []).append(index)
        for bucket in buckets.values():
            bucket.sort(key=lambda i: hashlib.sha256(dataset.rows[i]['id'].encode()).hexdigest())
        chosen = []
        while len(chosen) < target and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(chosen) < target:
                    chosen.append(buckets[key].pop(0))
        selected.extend(chosen)
    return selected
