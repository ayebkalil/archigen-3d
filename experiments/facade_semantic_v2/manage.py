"""Private Kaggle staging and checksum-validated local V2 installation."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

from experiments.facade_sd15_controlnet.data import write_json
from experiments.facade_sd15_controlnet.manage import bundle_run
from experiments.facade_sd15_controlnet.tracking import import_tracking
from .runtime import validate_export


def stage_kernel(output, commit=None, mode='smoke'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    commit = commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('A full immutable Git SHA is required')
    notebook = json.loads(Path(__file__).with_name('train_facade_semantic.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = [line.replace('PINNED_COMMIT = ""', f'PINNED_COMMIT = "{commit}"')
                .replace('MODE = "smoke"', f'MODE = "{mode}"') for line in cell['source']]
    write_json(output / 'train_facade_semantic.ipynb', notebook)
    metadata = {'id': 'khalilayeb/archigen-facade-semantic7-lora-v2-' + mode,
        'title': 'ArchiGen Facade Semantic7 LoRA V2 ' + mode.title(),
        'code_file': 'train_facade_semantic.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': ['khalilayeb/archigen-cmp-facades-original-locked-splits', 'liushuyuu/irregular-facades-irfs'],
        'kernel_sources': ['khalilayeb/archigen-facade-sd15-controlnet-mlflow'],
        'competition_sources': [], 'model_sources': []}
    write_json(output / 'kernel-metadata.json', metadata)
    write_json(output / 'source-pin.json', {'commit': commit, 'branch': 'codex/facade-semantic-controlnet-v2'})
    return metadata


def install(archive, destination, tracking_uri=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Existing experiments/models are never overwritten')
    with tempfile.TemporaryDirectory(prefix='archigen-semantic-v2-') as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not (root / member.filename).resolve().is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unsafe archive path/link')
            zipped.extractall(root)
        metadata = validate_export(root / 'export')
        if metadata['quality_status'].startswith('SMOKE'):
            raise ValueError('Smoke artifacts are not deployable models')
        shutil.copytree(root, destination)
    ids = []
    if tracking_uri:
        ids.append(import_tracking(destination, tracking_uri))
        if (destination / 'evaluation/tracking-events.jsonl').exists():
            ids.append(import_tracking(destination / 'evaluation', tracking_uri))
    return {'model': str(destination / 'export'), 'mlflow_run_ids': ids}


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage-kernel'); stage.add_argument('--output', required=True)
    stage.add_argument('--commit'); stage.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    pack = commands.add_parser('bundle'); pack.add_argument('--run', required=True); pack.add_argument('--archive', required=True)
    local = commands.add_parser('install'); local.add_argument('--archive', required=True); local.add_argument('--destination', required=True)
    local.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    args = parser.parse_args()
    if args.command == 'stage-kernel':
        print(json.dumps(stage_kernel(args.output, args.commit, args.mode)))
    elif args.command == 'bundle':
        print(json.dumps(bundle_run(args.run, args.archive)))
    else:
        print(json.dumps(install(args.archive, args.destination, args.tracking_uri)))


if __name__ == '__main__': main()
