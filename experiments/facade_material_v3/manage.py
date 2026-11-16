"""Pin private Kaggle notebooks and safely install experimental exports locally."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

from experiments.facade_sd15_controlnet.data import write_json
from experiments.facade_sd15_controlnet.tracking import import_tracking
from .runtime import validate_export


def stage_kernel(output,commit=None,mode='smoke'):
    if mode not in {'smoke','train'}:raise ValueError('Unknown mode')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commit=commit or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Pin an immutable full Git SHA')
    notebook=json.loads(Path(__file__).with_name('train_facade_material.ipynb').read_text())
    execution_mode='staged' if mode=='train' else 'smoke'
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell['source']=[line.replace('PINNED_COMMIT = ""',f'PINNED_COMMIT = "{commit}"').replace('MODE = "smoke"',f'MODE = "{execution_mode}"') for line in cell['source']]
    write_json(output/'train_facade_material.ipynb',notebook)
    metadata={'id':'khalilayeb/archigen-facade-material10-feedback-v3-'+mode,
        'title':'ArchiGen Facade Material10 Feedback V3 '+mode.title(),
        'code_file':'train_facade_material.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_internet':True,
        'dataset_sources':['khalilayeb/archigen-cmp-facades-original-locked-splits','liushuyuu/irregular-facades-irfs',
            'khalilayeb/archigen-v3-teacher-recovery'],
        'kernel_sources':['khalilayeb/archigen-facade-semantic7-lora-v2-train'],'competition_sources':[],'model_sources':[]}
    write_json(output/'kernel-metadata.json',metadata)
    write_json(output/'source-pin.json',{'commit':commit,'branch':'codex/facade-material-controlnet-v3','mode':mode,'execution_mode':execution_mode})
    return metadata


def install(archive,destination,tracking_uri=None):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Existing model folders are never overwritten')
    with tempfile.TemporaryDirectory(prefix='archigen-material-v3-') as temporary:
        root=Path(temporary).resolve()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not (root/member.filename).resolve().is_relative_to(root) or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError('Unsafe archive path or link')
            zipped.extractall(root)
        metadata=validate_export(root/'export')
        if metadata.get('selected_step',0)==0:
            raise ValueError('No improved trained checkpoint was selected; step-zero initialization is not a new V3 model')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(root,destination)
    ids=[]
    if tracking_uri:
        for folder in (destination/'teacher',destination):
            if (folder/'tracking-events.jsonl').is_file():ids.append(import_tracking(folder,tracking_uri))
    return {'model':str(destination/'export'),'mlflow_run_ids':ids,'selected_step':metadata['selected_step'],
        'production_changed':False,'teacher_needed_for_inference':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    stage=sub.add_parser('stage-kernel');stage.add_argument('--output',required=True);stage.add_argument('--commit');stage.add_argument('--mode',choices=['smoke','train'],default='smoke')
    local=sub.add_parser('install');local.add_argument('--archive',required=True);local.add_argument('--destination',required=True);local.add_argument('--tracking-uri')
    args=parser.parse_args()
    print(json.dumps(stage_kernel(args.output,args.commit,args.mode) if args.command=='stage-kernel' else install(args.archive,args.destination,args.tracking_uri)))
