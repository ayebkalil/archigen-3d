"""Fetch only pretrained VGG-19 convolution weights (80 MB, not its 494 MB classifier).

Uses the timm mirror of torchvision ImageNet weights, pinned to its resolved commit.
HTTPS range responses, tensor dimensions and byte counts are checked explicitly.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import requests
import torch
from torchvision.models.vgg import make_layers, cfgs


def prepare():
    repo = 'timm/vgg19.tv_in1k'
    info = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=30)
    info.raise_for_status(); revision = info.json()['sha']
    url = f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors'
    def fetch(start, end):
        for attempt in range(4):
            try:
                response = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=60)
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != end-start+1:
                    raise ValueError('Invalid range response')
                return response.content
            except (requests.RequestException, ValueError):
                if attempt == 3: raise
    header_size = struct.unpack('<Q', fetch(0, 7))[0]
    if header_size > 1024**2: raise ValueError('Unexpected safetensors header')
    header = json.loads(fetch(8, 7 + header_size))
    tensors = {k: v for k, v in header.items() if k.startswith('features.')}
    count = max(v['data_offsets'][1] for v in tensors.values())
    if count > 100 * 1024**2: raise ValueError('Unexpected feature weights size')
    offset = 8 + header_size
    chunks = [(s, min(s+1024**2, count)) for s in range(0, count, 1024**2)]
    def chunk(bounds):
        s, e = bounds
        return fetch(offset+s, offset+e-1)
    raw = bytearray()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, part in enumerate(pool.map(chunk, chunks)):
            raw.extend(part)
            if i % 10 == 0: print(f'VGG features: {i+1}/{len(chunks)} MB', flush=True)
    state = {}
    for name, value in tensors.items():
        if value['dtype'] != 'F32': raise ValueError('Unexpected weights dtype')
        start, end = value['data_offsets']
        array = np.frombuffer(raw[start:end], dtype='<f4').copy().reshape(value['shape'])
        state[name.removeprefix('features.')] = torch.from_numpy(array)
    make_layers(cfgs['E']).load_state_dict(state, strict=True)
    dest = Path(torch.hub.get_dir())/'checkpoints/vgg19_features.pth'
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dest)
    dest.with_suffix('.json').write_text(json.dumps({'source': url, 'revision': revision,
        'description': 'Torchvision ImageNet VGG-19 convolutional features, timm mirror',
        'tensor_bytes_sha256': hashlib.sha256(raw).hexdigest(),
        'file_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2))
    print(dest, flush=True)


if __name__ == '__main__': prepare()
