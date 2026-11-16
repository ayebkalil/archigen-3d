"""Private Kaggle orchestration: teacher gate, diffusion, portable export and reload."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from experiments.facade_sd15_controlnet.data import locate_kaggle_dataset,digest,write_json
from experiments.facade_semantic_v2.data import build_dataset,validate_dataset
from .runtime import validate_export
from .data import audit_material_samples


def run_staged(work):
    subprocess.run([sys.executable,'-m','experiments.facade_material_v3.kaggle_run','--mode','smoke'],check=True)
    smoke=work/'facade-material-v3-smoke'
    check=json.loads((smoke/'reload-check.json').read_text())
    metadata=validate_export(smoke/'export',allow_smoke=True)
    if not check.get('ten_channel_inference') or not check.get('smoke_only') or metadata['config']['max_train_steps']!=2:
        raise RuntimeError('Smoke completion contract failed; full training was not started')
    write_json(work/'v3-preflight-passed.json',{'smoke_passed':True,'weights_sha256':metadata['weights_sha256'],
        'training_run_id':metadata['run_id'],'reload_check':check})
    print('PREFLIGHT PASSED; starting quality-gated full teacher and diffusion training',flush=True)
    subprocess.run([sys.executable,'-m','experiments.facade_material_v3.kaggle_run','--mode','train'],check=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',choices=['smoke','train','staged'],default='smoke')
    args=parser.parse_args()
    inputs=Path('/kaggle/input');work=Path('/kaggle/working')
    if not inputs.exists():raise RuntimeError('This command requires Kaggle mounts')
    if args.mode=='staged':
        run_staged(work);return
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    parents=[p.parent for p in inputs.rglob('model.json') if p.parent.name=='export'
        and json.loads(p.read_text()).get('weights_sha256')==config['parent_weights_sha256']]
    if len(parents)!=1:raise ValueError('Attach exactly the completed V2 training output')
    cmp_mounts=[p.parent for p in inputs.rglob('manifest.json') if (p.parent/'photos').is_dir() or (p.parent/'photos.zip').is_file()]
    irf_lists=list(inputs.rglob('sample_list.xlsx'))
    if len(cmp_mounts)!=1 or len(irf_lists)!=1:raise ValueError('Attach original locked CMP and original human-annotated IRFs')
    root=work/'facade-v3-data';output=work/('facade-material-v3-smoke' if args.mode=='smoke' else 'facade-material-v3');output.mkdir(exist_ok=True)
    cmp=locate_kaggle_dataset(input_root=cmp_mounts[0])
    if not (root/'manifest.json').is_file():build_dataset(cmp,irf_lists[0].parent,root)
    validate_dataset(root)
    material_audit=root/'material-reference-audit.json'
    audit_material_samples(root,material_audit,config['strip_crop_probability'],config['blank_crop_probability'])
    shutil.copyfile(material_audit,output/'material-reference-audit.json')
    for name in ('manifest.json','audit.json','ATTRIBUTION.md','irf-credits.json'):shutil.copyfile(root/name,output/name)
    (output/'requirements-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for module,extra in [('teacher',[]),('train',['--parent',str(parents[0]),'--teacher',str(output/'teacher')])]:
        command=[sys.executable,'-m','experiments.facade_material_v3.'+module,'--data',str(root),'--output',str(output/'teacher' if module=='teacher' else output),*extra]
        if args.mode=='smoke':command.append('--smoke')
        subprocess.run(command,check=True)
    validate_export(output/'export',allow_smoke=args.mode=='smoke')
    # Separate process releases optimizer/base residency before portable offloaded inference.
    reload_code='''
from pathlib import Path
from experiments.facade_material_v3.runtime import MaterialFacadeRuntime
from experiments.facade_material_v3.data import MaterialDataset
from experiments.facade_sd15_controlnet.data import write_json
data=MaterialDataset('/kaggle/working/facade-v3-data','val')
sample=data.sample(0)
runtime=MaterialFacadeRuntime('/kaggle/working/facade-material-v3/export',allow_download=True,allow_smoke=SMOKE)
image=runtime.generate_material(sample[3],sample[4])
image.save('/kaggle/working/facade-material-v3/export-reloaded.png')
write_json(Path('/kaggle/working/facade-material-v3/reload-check.json'),{'ten_channel_inference':True,'teacher_required_for_inference':False,'profile':runtime.last_profile,'smoke_only':SMOKE,'quality_claim':False})
print('EXPORT RELOAD PASSED: material10 + existing LoRA, no teacher needed at inference',flush=True)
'''.replace('/kaggle/working/facade-material-v3',str(output)).replace('SMOKE',str(args.mode=='smoke'))
    subprocess.run([sys.executable,'-c',reload_code],check=True)
    archive=work/('archigen-facade-material-v3-smoke-model-and-mlflow.zip' if args.mode=='smoke' else 'archigen-facade-material-v3-model-and-mlflow.zip')
    paths=['export','teacher','validation','portable-artifacts','tracking-events.jsonl','tracking-run.json',
        'requirements-freeze.txt','config.json','contract.json','teacher.json','reload-check.json','export-reloaded.png',
        'manifest.json','audit.json','material-reference-audit.json','ATTRIBUTION.md','irf-credits.json']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as zipped:
        for name in paths:
            path=output/name
            for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if file.is_file():zipped.write(file,file.relative_to(output).as_posix())
    write_json(output/'bundle.json',{'archive':archive.name,'sha256':digest(archive),'mode':args.mode})
    print('V3 COMPLETE:',json.dumps({'archive':archive.name,'sha256':digest(archive),'mode':args.mode}),flush=True)


if __name__=='__main__':main()
