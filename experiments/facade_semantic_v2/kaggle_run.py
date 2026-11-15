"""Run only on Kaggle: audit raw mounts, train, reload, evaluate and bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.facade_sd15_controlnet.data import locate_kaggle_dataset, write_json, digest
from .data import build_dataset, validate_dataset, FacadeDataset
from .manage import bundle_run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    args = parser.parse_args()
    if not Path('/kaggle/input').exists():
        raise RuntimeError('This entry point requires Kaggle mounts')
    work = Path('/kaggle/working')
    irf_lists = list(Path('/kaggle/input').rglob('sample_list.xlsx'))
    if len(irf_lists) != 1:
        raise ValueError('Attach exactly one original IRFs dataset')
    parents = [p.parent for p in Path('/kaggle/input').rglob('model.json')
               if p.parent.name == 'export' and (p.parent / 'controlnet/config.json').is_file()]
    expected = json.loads(Path(__file__).with_name('config.json').read_text())['init_checkpoint_sha256']
    parents = [p for p in parents if digest(p / 'controlnet/diffusion_pytorch_model.safetensors') == expected]
    if len(parents) != 1:
        raise ValueError('Attach V1 training output with selected step-750 parent weights')
    parent = parents[0]
    data_root, output = work / 'facade-v2-data', work / 'facade-semantic-v2'
    output.mkdir(parents=True, exist_ok=True)
    # Attached V1 run artifacts also contain a manifest; only the raw CMP mount
    # has photos/labels siblings. Do not mistake copied tracking reports for data.
    cmp_mounts = [p.parent for p in Path('/kaggle/input').rglob('manifest.json')
                  if (p.parent / 'photos').is_dir() or (p.parent / 'photos.zip').is_file()]
    if len(cmp_mounts) != 1:
        raise ValueError('Attach exactly one prepared CMP raw dataset mount')
    cmp_root = locate_kaggle_dataset(input_root=cmp_mounts[0])
    manifest = build_dataset(cmp_root, irf_lists[0].parent, data_root)
    validate_dataset(data_root)
    for filename in ('manifest.json', 'audit.json', 'ATTRIBUTION.md', 'irf-credits.json'):
        import shutil
        shutil.copyfile(data_root / filename, output / filename)
    print('Verified original training/validation/test counts:', manifest['counts'], flush=True)
    command = [sys.executable, '-m', 'experiments.facade_semantic_v2.train', '--data', str(data_root), '--parent', str(parent), '--output', str(output)]
    if args.mode == 'smoke':
        command.append('--smoke')
    subprocess.run(command, check=True)
    if args.mode == 'smoke':
        # New process after training fully releases VRAM before testing the portable export.
        code = '''
import json, torch
from pathlib import Path
from experiments.facade_semantic_v2.runtime import SemanticFacadeRuntime
from experiments.facade_semantic_v2.data import FacadeDataset
data = FacadeDataset('/kaggle/working/facade-v2-data', 'val')
runtime = SemanticFacadeRuntime('/kaggle/working/facade-semantic-v2/export', allow_download=True)
image = runtime.generate_control(torch.from_numpy(data.images(0)[3])[None])
image.save('/kaggle/working/facade-semantic-v2/smoke-export-reloaded.png')
Path('/kaggle/working/facade-semantic-v2/smoke-success.json').write_text(json.dumps({'trained_steps': 2, 'export_reload_inference': True, 'profile': runtime.last_profile, 'quality_claim': False}))
print('SMOKE PASSED: semantic7 + LoRA training, checkpoint, export and offloaded reload inference', flush=True)
'''
        subprocess.run([sys.executable, '-c', code], check=True)
    else:
        subprocess.run([sys.executable, '-m', 'experiments.facade_semantic_v2.evaluate',
            '--model', str(output / 'export'), '--parent', str(parent), '--data', str(data_root),
            '--output', str(output / 'evaluation')], check=True)
    report = bundle_run(output, work / 'archigen-facade-semantic-v2-model-and-mlflow.zip')
    write_json(output / 'bundle.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
