"""Restore only selected DVC weights, then verify pinned SHA-256 values."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.artifacts import inspect_artifacts, inventory, sha256, ModelArtifactError


def restore_sidecar(item, root=ROOT):
    """Restore small Git-tracked sidecars atomically, after checking exact bytes."""
    target = Path(root) / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='model-sidecar-', dir=target.parent) as temporary:
        downloaded = Path(temporary) / 'download'
        with urlopen(item['url'], timeout=60) as response, downloaded.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if downloaded.stat().st_size != item['size'] or sha256(downloaded) != item['sha256']:
            raise ModelArtifactError(f'Sidecar checksum mismatch: {item["path"]}')
        downloaded.replace(target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['check', 'pull'])
    parser.add_argument('--profile', choices=sorted(inventory(ROOT)['profiles']), default='runtime')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    manifest = inventory(ROOT)
    before = inspect_artifacts(args.profile, ROOT)
    if args.command == 'pull':
        needed = {item['path'] for item in before if item['status'] != 'ok'}
        selected = [item for item in manifest['artifacts'] if item['path'] in needed]
        targets = [item['dvc'] for item in selected if item.get('dvc')]
        if targets:
            originals = {target: (ROOT / target).read_bytes() for target in targets}
            try:
                # DVC's hydrated import metadata can otherwise require a cache
                # remote after its cache is lost. Recreate a no-download import
                # against the pinned URL, then let DVC fetch from that source.
                for item in selected:
                    if item.get('dvc'):
                        subprocess.run([sys.executable, '-m', 'dvc', 'import-url', '--no-download', '--force',
                                        item['url'], item['path']], cwd=ROOT, check=True)
                subprocess.run([sys.executable, '-m', 'dvc', 'pull', '--force', *targets], cwd=ROOT, check=True)
            except subprocess.CalledProcessError:
                print('Model restore failed. Install DVC (`python -m pip install dvc`) and check access to the versioned release URLs.', file=sys.stderr)
                return 1
            finally:
                # Installation is not a model-version change: keep Git metadata
                # unchanged after DVC fills in its local content hashes.
                for target, content in originals.items():
                    (ROOT / target).write_bytes(content)
        try:
            for item in selected:
                if not item.get('dvc'):
                    restore_sidecar(item)
        except (OSError, ModelArtifactError) as error:
            print(f'Model sidecar restore failed: {error}', file=sys.stderr)
            return 1
    results = inspect_artifacts(args.profile, ROOT)
    if args.json:
        print(json.dumps({'profile': args.profile, 'artifacts': results}, indent=2))
    else:
        for item in results:
            print(f"{item['status'].upper():7} {item['path']}")
        if any(item['status'] != 'ok' for item in results):
            print(f'Restore with: python scripts/models.py pull --profile {args.profile}')
    return int(any(item['status'] != 'ok' for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
