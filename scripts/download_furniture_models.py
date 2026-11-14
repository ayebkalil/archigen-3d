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

    hashes = {
        'model.ckpt': '429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee',
        'config.yaml': '74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62',
    }
    for filename, expected in hashes.items():
        path = ROOT / 'TripoSR' / filename
        if not path.is_file():
            print(f'Downloading stabilityai/TripoSR/{filename}', flush=True)
            hf_hub_download('stabilityai/TripoSR', filename,
                            revision='5b521936b01fbe1890f6f9baed0254ab6351c04a', local_dir=ROOT / 'TripoSR')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise RuntimeError(f'TripoSR {filename} SHA256 mismatch: {digest}')
        print(f'Verified {filename}: {digest}', flush=True)
    # Encoder weights are already inside the strictly loaded TripoSR checkpoint.
    hf_hub_download('facebook/dino-vitb16', 'config.json', revision='f205d5d8e640a89a2b8ef0369670dfc37cc07fc2')
    rembg.new_session('u2netp', providers=['CPUExecutionProvider'])
    print('All GPU reconstruction model files are ready.', flush=True)


if __name__ == '__main__':
    main()
