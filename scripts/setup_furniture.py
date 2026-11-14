"""Install pinned public TripoSR source/weights without copying another PC's venv."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.artifacts import sha256

SOURCE = 'https://github.com/VAST-AI-Research/TripoSR.git'
COMMIT = '107cefdc244c39106fa830359024f6a2f1c78871'
REVISION = '5b521936b01fbe1890f6f9baed0254ab6351c04a'
FILES = {
    'model.ckpt': '429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee',
    'config.yaml': '74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62',
}


def download(name, destination):
    url = f'https://huggingface.co/stabilityai/TripoSR/resolve/{REVISION}/{name}'
    temporary = destination.with_suffix(destination.suffix + '.download')
    checksum = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'ArchiGen-model-setup'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                checksum.update(block); output.write(block)
        if checksum.hexdigest() != FILES[name]:
            raise RuntimeError(f'Checksum mismatch for downloaded {name}; existing file was not changed')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['pull', 'check'])
    args = parser.parse_args()
    target = ROOT / 'TripoSR'
    if args.command == 'pull' and not target.exists():
        subprocess.run(['git', 'clone', '--no-checkout', SOURCE, str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', COMMIT], check=True)
    if not (target / 'tsr/system.py').is_file():
        print('MISSING TripoSR source. Run: python scripts/setup_furniture.py pull')
        return 1
    actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'TripoSR source differs from tested revision {COMMIT}; existing checkout was not changed')
    # The quality engine uses its own marching cubes, so this import is optional.
    isosurface = target / 'tsr/models/isosurface.py'
    if args.command == 'pull' and 'except ImportError:' not in isosurface.read_text():
        subprocess.run(['git', '-C', str(target), 'apply', '--unidiff-zero', '--ignore-space-change',
                        str(ROOT / 'patches/triposr-optional-marching-cubes.patch')], check=True)
    failed = False
    for name, expected in FILES.items():
        path = target / name
        good = path.is_file() and sha256(path) == expected
        if not good and args.command == 'pull':
            print(f'Downloading verified {name} (the checkpoint is approximately 1.6 GiB)', flush=True)
            download(name, path)
            good = True
        print(f'{"OK" if good else "MISSING/CORRUPT"} TripoSR/{name}')
        failed |= not good
    if failed:
        print('Run: python scripts/setup_furniture.py pull')
    else:
        print('Weights/source ready. Use the isolated CUDA environment documented in docs/model-setup.md.')
        print('First use also caches the DINO config and rembg background-removal model from their upstream hosts.')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
