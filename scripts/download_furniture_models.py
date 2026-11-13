"""Fetch official weights, verify TripoSR, and cache auxiliary models locally."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'outputs/runtime/huggingface'))
os.environ.setdefault('U2NET_HOME', str(ROOT / 'outputs/runtime/rembg'))


def main():
    from huggingface_hub import hf_hub_download
    import rembg

    for filename in ('config.yaml', 'model.ckpt'):
        if (ROOT / 'TripoSR' / filename).is_file():
            continue
        print(f'Downloading stabilityai/TripoSR/{filename}', flush=True)
        hf_hub_download('stabilityai/TripoSR', filename, local_dir=ROOT / 'TripoSR')
    checkpoint = ROOT / 'TripoSR/model.ckpt'
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected = '429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee'
    if digest != expected:
        raise RuntimeError(f'TripoSR checkpoint SHA256 mismatch: {digest}')
    print(f'Checkpoint verified: {digest}', flush=True)
    # Encoder weights are already inside the strictly loaded TripoSR checkpoint.
    hf_hub_download('facebook/dino-vitb16', 'config.json')
    rembg.new_session('u2netp', providers=['CPUExecutionProvider'])
    print('All GPU reconstruction model files are ready.', flush=True)


if __name__ == '__main__':
    main()
