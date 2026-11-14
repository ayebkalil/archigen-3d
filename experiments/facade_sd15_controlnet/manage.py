"""Prepare Kaggle inputs, pin notebook source, package results and install locally."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

from .data import digest, package_dataset, validate_dataset, write_json
from .tracking import import_tracking

ROOT = Path(__file__).resolve().parents[2]


def stage_kernel(output, owner='khalilayeb', commit=None, mode='train'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((Path(__file__).with_name('train_facade_controlnet.ipynb')).read_text(encoding='utf-8'))
    commit = commit or subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = [line.replace('PINNED_COMMIT = ""', f'PINNED_COMMIT = "{commit}"')
                .replace('MODE = "train"', f'MODE = "{mode}"') for line in cell['source']]
    write_json(output / 'train_facade_controlnet.ipynb', notebook)
    metadata = json.loads(Path(__file__).with_name('kernel-metadata.json').read_text())
    metadata['id'] = f'{owner}/archigen-facade-sd15-controlnet-mlflow'
    metadata['dataset_sources'] = [f'{owner}/archigen-cmp-facades-original-locked-splits']
    if mode == 'smoke':
        metadata['id'] = f'{owner}/archigen-facade-sd15-controlnet-smoke'
        metadata['title'] = 'ArchiGen Facade SD15 ControlNet Smoke'
    write_json(output / 'kernel-metadata.json', metadata)
    write_json(output / 'source-pin.json', {'commit': commit, 'repo': 'https://github.com/ayebkalil/archigen-3d',
        'branch': 'codex/facade-controlnet-kaggle-mlflow'})
    return metadata


def bundle_run(output, archive):
    output, archive = Path(output), Path(archive)
    if not (output / 'export/model.json').exists(): raise FileNotFoundError('No trained model export. Smoke runs do not produce deployable weights.')
    paths = [output / 'export', output / 'portable-artifacts', output / 'tracking-events.jsonl',
             output / 'tracking-run.json', output / 'requirements-freeze.txt', output / 'config.json', output / 'evaluation', output / 'tracking-history']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        for path in paths:
            for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if file.is_file(): zipped.write(file, file.relative_to(output).as_posix())
    return {'archive': str(archive), 'sha256': digest(archive)}


def install_bundle(archive, destination, tracking_uri=None):
    destination = Path(destination).resolve()
    if destination.exists(): raise FileExistsError('Choose a new experiment folder; existing models are not overwritten.')
    with tempfile.TemporaryDirectory(prefix='archigen-controlnet-') as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unsafe path/link in model archive')
            zipped.extractall(root)
        metadata = json.loads((root / 'export/model.json').read_text(encoding='utf-8'))
        if digest(root / 'export/controlnet/diffusion_pytorch_model.safetensors') != metadata['weights_sha256']:
            raise ValueError('Model checksum mismatch')
        if metadata.get('architecture') != 'sd15_controlnet_cmp_edges_v1': raise ValueError('Wrong model architecture')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, destination)
    imported = []
    if tracking_uri:
        for history in sorted((destination / 'tracking-history').glob('*')):
            if (history / 'tracking-events.jsonl').exists(): imported.append(import_tracking(history, tracking_uri))
        imported.append(import_tracking(destination, tracking_uri))
        if (destination / 'evaluation/tracking-events.jsonl').exists():
            imported.append(import_tracking(destination / 'evaluation', tracking_uri))
    return {'model_directory': str(destination / 'export'), 'mlflow_run_ids': imported,
            'training_run_id': metadata['run_id'], 'quality_status': metadata['quality_status']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('prepare-data')
    pack.add_argument('--manifest', required=True); pack.add_argument('--source-root', required=True)
    pack.add_argument('--output', required=True); pack.add_argument('--owner', default='khalilayeb'); pack.add_argument('--baseline')
    stage = commands.add_parser('stage-kernel'); stage.add_argument('--output', required=True); stage.add_argument('--owner', default='khalilayeb'); stage.add_argument('--commit'); stage.add_argument('--mode', choices=['train', 'smoke'], default='train')
    bundle = commands.add_parser('bundle'); bundle.add_argument('--run', required=True); bundle.add_argument('--archive', required=True)
    install = commands.add_parser('install'); install.add_argument('--archive', required=True); install.add_argument('--destination', required=True)
    install.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    tracking = commands.add_parser('import-tracking'); tracking.add_argument('--bundle', required=True); tracking.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    check = commands.add_parser('check-data'); check.add_argument('--data', required=True)
    args = parser.parse_args()
    if args.command == 'prepare-data':
        report = package_dataset(args.manifest, args.source_root, args.output, args.owner, args.baseline)
        print(json.dumps({'splits': report['splits'], 'manifest_sha256': digest(Path(args.output) / 'manifest.json')}))
    elif args.command == 'stage-kernel': print(json.dumps(stage_kernel(args.output, args.owner, args.commit, args.mode)))
    elif args.command == 'bundle': print(json.dumps(bundle_run(args.run, args.archive)))
    elif args.command == 'install': print(json.dumps(install_bundle(args.archive, args.destination, args.tracking_uri), indent=2))
    elif args.command == 'import-tracking': print(import_tracking(args.bundle, args.tracking_uri))
    else: print(json.dumps(validate_dataset(args.data)['splits']))


if __name__ == '__main__': main()
