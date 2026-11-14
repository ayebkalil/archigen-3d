"""Container-side SQLite backup and non-destructive native library import."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import sqlite3


def backup(database, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ('mlflow-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.db')
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as source:
        with sqlite3.connect(target) as destination:
            source.backup(destination)
    print(f'SQLite backup: {target}', flush=True)


def seed(source, destination):
    source, destination = Path(source), Path(destination)
    marker = destination / '.native-import-complete'
    if marker.exists():
        return
    # Never combine two independently modified catalogs silently.
    existing = list(destination.glob('custom/*.glb'))
    if existing or (destination / 'custom_furniture.json').exists():
        print('Existing Docker library retained; native import skipped.', flush=True)
    else:
        for item in (source / 'custom').glob('*.glb'):
            target = destination / 'custom' / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
        index = source / 'custom_furniture.json'
        if index.exists():
            shutil.copyfile(index, destination / index.name)
    marker.touch()
    for item in [destination, *destination.rglob('*')]:
        os.chown(item, 1000, 1000)
    print('Furniture library initialized.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['backup', 'seed'])
    args = parser.parse_args()
    if args.command == 'backup':
        backup('/mlflow/mlflow.db', '/mlflow/artifacts/.backups')
    else:
        seed('/native-library', '/library')
