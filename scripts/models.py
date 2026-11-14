"""Restore only selected DVC weights, then verify pinned SHA-256 values."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.artifacts import inspect_artifacts, inventory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['check', 'pull'])
    parser.add_argument('--profile', choices=['runtime'], default='runtime')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    manifest = inventory(ROOT)
    before = inspect_artifacts(args.profile, ROOT)
    if args.command == 'pull':
        needed = {item['path'] for item in before if item['status'] != 'ok'}
        targets = [item['dvc'] for item in manifest['artifacts'] if item['path'] in needed]
        if targets:
            try:
                subprocess.run([sys.executable, '-m', 'dvc', 'pull', '--force', *targets], cwd=ROOT, check=True)
            except subprocess.CalledProcessError:
                print('Model restore failed. Install DVC (`python -m pip install dvc`) and check access to the versioned release URLs.', file=sys.stderr)
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
