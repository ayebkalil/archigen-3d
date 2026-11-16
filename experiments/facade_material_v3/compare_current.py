"""Matched validation pilot: configured production Pix2Pix versus available V3 smoke export.

No quality-trained V3 generator exists. The export selected step zero; the
25-step variant changes sampling only. Neither variant is a new trained V3.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import digest, write_json
from experiments.facade_semantic_v2.conditioning import cmp_labels, letterbox, control_channels
from src.data.facade_palette import CMP_TRAINING_LUT


def prepare(project, source, output, count=16):
    project, source, output = Path(project), Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    buckets = {}
    for row in manifest['samples']:
        if row['split'] != 'val': continue
        labels = cmp_labels(np.asarray(Image.open(source / row['labels'])))
        n, _, stats, _ = cv2.connectedComponentsWithStats((labels == 2).astype(np.uint8))
        windows = sum(stats[i, cv2.CC_STAT_AREA] >= 25 for i in range(1, n))
        buckets.setdefault((bool((labels == 3).any()), min(windows // 10, 2)), []).append(row)
    for rows in buckets.values(): rows.sort(key=lambda r: hashlib.sha256(('42:'+r['id']).encode()).hexdigest())
    chosen = []
    while len(chosen) < count and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count: chosen.append(buckets[key].pop(0))
    if len(chosen) != count or count < 1: raise ValueError('Insufficient validation examples or invalid pilot size')
    panels = []
    tile = Image.open(project / 'frontend/public/showcase/tex-1k/white_stucco/diff.jpg').convert('RGB')
    for i, row in enumerate(chosen):
        photo = Image.open(source / row['photo']).convert('RGB')
        indexed = np.asarray(Image.open(source / row['labels'])).copy(); canonical = cmp_labels(indexed)
        yy, xx = np.where(np.isin(canonical, [1,2,3,4]))
        box = (int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1)
        pixels = np.asarray(photo).copy(); pixels[canonical == 0] = 127
        photo, indexed, valid = letterbox(Image.fromarray(pixels).crop(box), indexed[box[1]:box[3],box[0]:box[2]])
        canonical = cmp_labels(indexed)
        folder = output / f'case_{i:02d}'; folder.mkdir(exist_ok=True)
        photo.save(folder / 'target.png'); Image.fromarray(canonical).save(folder / 'labels.png')
        Image.fromarray((valid*255).astype(np.uint8)).save(folder / 'valid.png')
        # Each model receives its native semantic encoding from the same GT mask.
        small = np.asarray(Image.fromarray(indexed).resize((256,256),Image.Resampling.NEAREST))
        Image.fromarray(CMP_TRAINING_LUT[small]).save(folder / 'production-input.png')
        Image.fromarray(CMP_TRAINING_LUT[indexed]).save(folder / 'layout.png')
        ref = np.asarray(tile.resize((512,512),Image.Resampling.LANCZOS)).copy(); ref[canonical == 0] = 127
        Image.fromarray(ref).save(folder / 'reference.png')
        panels.append({'folder':folder.name,'id':row['id'],'split':'val','photo_sha256':digest(source/row['photo']),
            'labels_sha256':digest(source/row['labels']),'crop_box':list(box),'ground_truth_available':True})
    write_json(output / 'cases.json', panels)
    write_json(output / 'protocol.json', {'dataset':'Original CMP, fixed stratified validation pilot', 'n':len(panels),
        'selection':'Seed 42 SHA ordering, round-robin door-presence/window-count strata; selected before outputs',
        'test_used':False,'score_resolutions':[256,512],'primary_score_resolution':256,
        'preprocessing':'Shared facade bbox, neutral GT background, aspect-preserving 512 letterbox; valid rectangle scored',
        'production_condition':'Original CMP 12-class training LUT at native 256; native production sharpening on; styles off',
        'v3_condition':'Same annotation collapsed to semantic7 + explicit independent PBR material reference at 512',
        'reference_is_target_derived':False,'v3_reference_kernels_zero':True,
        'v3_status':'SMOKE export selected step 0; V2 parent initialization, not quality-trained V3',
        'v3_variants':{'v3_default':2,'v3_25steps':25},
        'metrics_note':'Validation previously used for model selection, not an independent test. Single fixed SD seed per image. No FID on this small pilot; no generator IoU using the rejected teacher.',
        'attribution':'CMP Facade Database, Tylecek and Sara (2013), https://cmp.felk.cvut.cz/~tylecr1/facade/, CC BY-SA. Ground-truth images in comparison figures retain this attribution.'})
    print('Prepared',len(panels),'locked validation examples, without reading test images.',flush=True)


def reconstruct(parent, metadata_file, output):
    """Recover the retained initialization only if its exact Kaggle SHA matches."""
    import torch
    from safetensors.torch import load_file,save_file
    parent,output=Path(parent),Path(output);metadata=json.loads(Path(metadata_file).read_text())
    if metadata['selected_step']!=0 or metadata['improved_selection_score']:
        raise ValueError('A trained checkpoint must be downloaded; it cannot be reconstructed from its parent')
    if digest(parent/'controlnet/diffusion_pytorch_model.safetensors')!=metadata['config']['parent_weights_sha256']:
        raise ValueError('Wrong parent weights')
    if digest(parent/'lora/facade_lora.safetensors')!=metadata['lora_sha256']:raise ValueError('Wrong parent LoRA')
    weights=output/'controlnet/diffusion_pytorch_model.safetensors'
    if weights.exists():
        if digest(weights)!=metadata['weights_sha256']:raise FileExistsError('Existing reconstruction differs; choose another directory')
    else:
        weights.parent.mkdir(parents=True,exist_ok=True)
        state=load_file(str(parent/'controlnet/diffusion_pytorch_model.safetensors'))
        key='controlnet_cond_embedding.conv_in.weight';old=state[key]
        state[key]=torch.cat([old,torch.zeros((old.shape[0],3,*old.shape[2:]),dtype=old.dtype)],1)
        save_file(state,str(weights),metadata={'format':'pt'})
        if digest(weights)!=metadata['weights_sha256']:
            raise ValueError('Reconstruction differs from Kaggle bytes; download the actual export before comparing')
    config=json.loads((parent/'controlnet/config.json').read_text());config['conditioning_channels']=10
    write_json(output/'controlnet/config.json',config)
    shutil.copytree(parent/'lora',output/'lora',dirs_exist_ok=True);write_json(output/'model.json',metadata)
    write_json(output.parent/'reconstruction.json',{'selected_step':0,'weights_byte_identical_to_kaggle':True,
        'parent_weights_sha256':metadata['config']['parent_weights_sha256'],'weights_sha256':metadata['weights_sha256'],
        'lora_sha256':metadata['lora_sha256']})
    print('Reconstructed exact Kaggle step-zero export; SHA-256 verified.',flush=True)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def baseline(project, output):
    import torch
    project, output = Path(project), Path(output)
    torch.set_num_threads(2)
    file = project / 'models/saved/generator_perceptual_ema_best.pth'
    registry = json.loads((project / 'models/artifacts.json').read_text())
    expected = next(a['sha256'] for a in registry['artifacts'] if a['path']=='models/saved/generator_perceptual_ema_best.pth')
    if digest(file) != expected: raise ValueError('Production checkpoint checksum mismatch')
    architecture = load_source('primary_facade_architecture',project/'src/models/architecture.py')
    bridge = load_source('primary_facade_bridge',project/'src/bridge/facade_bridge.py')
    state = torch.load(file,map_location='cpu',weights_only=True)
    if 'generator' in state: state=state['generator']
    elif 'model_state_dict' in state:state=state['model_state_dict']
    generator = architecture.UNetGenerator(in_channels=3,out_channels=3,upsample_mode='nearest')
    generator.load_state_dict(state); generator.to('cuda').eval()
    engine = bridge.PyTorchFacadeInference(generator,device='cuda',enhance=True)
    cases = json.loads((output/'cases.json').read_text())
    engine.generate((output/cases[0]['folder']/'production-input.png').read_bytes())
    torch.cuda.reset_peak_memory_stats(); profiles=[]
    for item in cases:
        folder=output/item['folder']; times=[]
        for _ in range(3):
            torch.cuda.synchronize(); start=time.perf_counter()
            result=engine.generate((folder/'production-input.png').read_bytes())
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
        (folder/'production.png').write_bytes(result)
        profiles.append({'id':item['id'],'seconds':float(np.median(times)),'repeats':3})
    write_json(output/'production-profile.json',{'checkpoint':file.name,'sha256':expected,'device':torch.cuda.get_device_name(0),
        'mode':'Native FP32 PyTorch CUDA, production PNG wrapper and default sharpening, no style finish',
        'generator_parameters':sum(p.numel() for p in generator.parameters()),'native_resolution':256,
        'architecture_sha256':digest(project/'src/models/architecture.py'),'bridge_sha256':digest(project/'src/bridge/facade_bridge.py'),
        'checkpoint_bytes':file.stat().st_size,'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'cases':profiles})
    print('Production model completed:',len(profiles),'cases.',flush=True)


def v3(model, output):
    import torch
    from .runtime import MaterialFacadeRuntime
    output=Path(output);runtime=MaterialFacadeRuntime(model,allow_download=False,allow_smoke=True)
    if runtime.metadata['selected_step']!=0:raise ValueError('This protocol documents a step-zero export only')
    from safetensors import safe_open
    with safe_open(str(Path(model)/'controlnet/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as tensors:
        if torch.count_nonzero(tensors.get_tensor('controlnet_cond_embedding.conv_in.weight')[:,7:]):
            raise ValueError('Expected zero reference kernels in retained initialization')
    runtime._load()
    report={'selected_step':0,'quality_status':runtime.metadata['quality_status'],'weights_sha256':runtime.metadata['weights_sha256'],
        'lora_sha256':runtime.metadata['lora_sha256'],'reference_kernels_zero':True,
        'device':torch.cuda.get_device_name(0),'mode':'FP16 sequential CPU offload; no compositor, tinting or retouching',
        'settings':{k:runtime.config[k] for k in ['prompt','negative_prompt','guidance_scale','controlnet_conditioning_scale','seed']},'cases':[]}
    cases=json.loads((output/'cases.json').read_text())
    for i,item in enumerate(cases):
        folder=output/item['folder'];labels=np.asarray(Image.open(folder/'labels.png'))
        control=control_channels(labels);reference=Image.open(folder/'reference.png').convert('RGB')
        for name,steps in [('v3_default',2),('v3_25steps',25)]:
            runtime.config['inference_steps']=steps
            result=runtime.generate_material(control,reference,seed=42+i)
            result.save(folder/(name+'.png'))
            report['cases'].append({'id':item['id'],'variant':name,'inference_steps':steps,**runtime.last_profile})
            write_json(output/'v3-profile.json',report)
        print(json.dumps({'generated':i+1,'total':len(cases),'id':item['id'],'seconds_25steps':runtime.last_profile['seconds']}),flush=True)
    report['complete']=True;write_json(output/'v3-profile.json',report)


def preview(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);cases=json.loads((output/'cases.json').read_text())
    titles=['Same annotated layout','Ground-truth photo','Current local Pix2Pix · 256','Kaggle V3 init · 512 / 25 steps']
    files=['layout.png','target.png','production.png','v3_25steps.png']
    preview_count=min(3,len(cases))
    fig,axes=plt.subplots(preview_count,4,figsize=(14,3*preview_count+1),facecolor='#f6f2e9',squeeze=False)
    for row,item in enumerate(cases[:3]):
        for col,(file,title) in enumerate(zip(files,titles)):
            axes[row,col].imshow(Image.open(output/item['folder']/file),interpolation='nearest' if col==0 else 'antialiased')
            axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(title,fontsize=10)
            if col==0:axes[row,col].text(.02,-.03,item['id'],transform=axes[row,col].transAxes,fontsize=9)
    fig.suptitle('Current project model versus the available Kaggle generator',fontsize=16,y=.985)
    fig.text(.5,.045,'Fixed first three pilot examples. Local images include normal production sharpening. No comparison-specific retouching.\nV3 retained step 0 (V2 initialization). 25 steps change inference only; full V3 training never started.',ha='center',fontsize=9)
    fig.text(.5,.012,'Targets: CMP Facade Database, Tylecek & Sara (2013), CC BY-SA. https://cmp.felk.cvut.cz/~tylecr1/facade/',ha='center',fontsize=8)
    fig.subplots_adjust(top=.93,bottom=.115,left=.02,right=.98,hspace=.06,wspace=.04)
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    # Show the actual two-step smoke configuration separately from the useful
    # 25-step inference diagnostic, without mislabelling either as trained V3.
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),facecolor='#f6f2e9')
    for ax,file,title in zip(axes,['production.png','v3_default.png','v3_25steps.png'],
        ['Current local Pix2Pix','V3 export default: 2 steps (smoke)','Same V3 checkpoint: 25 steps']):
        ax.imshow(Image.open(output/cases[0]['folder']/file));ax.axis('off');ax.set_title(title,fontsize=10)
    fig.text(.5,.015,'Diagnostic only: no full V3 generator fine-tuning occurred. Fixed first pilot case; no manual image correction.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(output/'smoke-settings-comparison.png',dpi=150);plt.close(fig)


def report(output):
    import torch
    torch.set_num_threads(2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.facade_semantic_v2.evaluate import pair_metrics
    output=Path(output);cases=json.loads((output/'cases.json').read_text());rows=[]
    variants=['production','v3_default','v3_25steps']
    for item in cases:
        folder=output/item['folder'];target=Image.open(folder/'target.png').convert('RGB')
        for size in (256,512):
            truth=target.resize((size,size),Image.Resampling.LANCZOS)
            labels=np.asarray(Image.open(folder/'labels.png').resize((size,size),Image.Resampling.NEAREST))
            valid=np.asarray(Image.open(folder/'valid.png').resize((size,size),Image.Resampling.NEAREST))/255
            for name in variants:
                prediction=Image.open(folder/(name+'.png')).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
                metrics=pair_metrics(prediction,truth,labels,valid)
                metrics['foreground_mae']=float(np.abs(np.asarray(prediction,np.float32)-np.asarray(truth,np.float32))[((labels>0)&(valid>0))].mean()/255)
                rows.append({'id':item['id'],'model':name,'score_resolution':size,**metrics})
    summary={}
    for size in (256,512):
        summary[str(size)]={name:{key:float(np.mean([r[key] for r in rows if r['model']==name and r['score_resolution']==size]))
            for key in ['lpips','ssim','psnr','opening_edge_mae','foreground_mae']} for name in variants}
    baseline_profile=json.loads((output/'production-profile.json').read_text());v3_profile=json.loads((output/'v3-profile.json').read_text())
    timings={'production':float(np.median([r['seconds'] for r in baseline_profile['cases']]))}
    for name in variants[1:]:timings[name]=float(np.median([r['seconds'] for r in v3_profile['cases'] if r['variant']==name]))
    paired={};rng=np.random.default_rng(42)
    for name in variants[1:]:
        delta=np.array([next(r['lpips'] for r in rows if r['id']==item['id'] and r['model']==name and r['score_resolution']==256)
            -next(r['lpips'] for r in rows if r['id']==item['id'] and r['model']=='production' and r['score_resolution']==256) for item in cases])
        paired[name]={'delta_v3_minus_current_mean':float(delta.mean()),'v3_lower_lpips_cases':int((delta<0).sum()),
            'paired_bootstrap_mean_delta_95ci':np.quantile(rng.choice(delta,(2000,len(delta))).mean(1),[.025,.975]).tolist()}
    result={'n':len(cases),'dataset':'CMP validation pilot; not held-out test','v3_full_training_completed':False,
        'v3_export_selected_step':0,'means':summary,'median_seconds':timings,'per_image':rows,
        'paired_lpips_diagnostics_256':paired,
        'generated_artifact_hashes':{item['id']:{name:digest(output/item['folder']/(name+'.png'))
            for name in ['target','layout','production','v3_default','v3_25steps']} for item in cases},
        'production':baseline_profile,'v3':v3_profile,'protocol':json.loads((output/'protocol.json').read_text())}
    write_json(output/'comparison-metrics.json',result)
    with (output/'per-image-metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'summary-metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','evaluation_resolution','n','lpips','ssim','psnr','opening_edge_mae','foreground_mae','median_seconds'])
        writer.writeheader()
        for size in (256,512):
            for name in variants:writer.writerow({'model':name,'evaluation_resolution':size,'n':len(cases),**summary[str(size)][name],'median_seconds':timings[name]})
    preview(output)
    # All pilot examples remain inspectable, avoiding a three-example visual ranking.
    fig,axes=plt.subplots(len(cases),3,figsize=(9,3*len(cases)),facecolor='#f6f2e9')
    for i,item in enumerate(cases):
        for j,(file,title) in enumerate(zip(['target.png','production.png','v3_25steps.png'],['Target','Current local','V3 init / 25 steps'])):
            axes[i,j].imshow(Image.open(output/item['folder']/file));axes[i,j].axis('off')
            axes[i,j].set_title(item['id']+' · '+title,fontsize=8)
    fig.tight_layout();fig.savefig(output/'all-pilot-images.png',dpi=100);plt.close(fig)
    print(json.dumps({'n':len(cases),'common_256':summary['256'],'median_seconds':timings},indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project',required=True);p.add_argument('--source',required=True);p.add_argument('--count',type=int,default=16)
    b=sub.add_parser('baseline');b.add_argument('--project',required=True)
    g=sub.add_parser('v3');g.add_argument('--model',required=True)
    r=sub.add_parser('reconstruct');r.add_argument('--parent',required=True);r.add_argument('--metadata',required=True)
    sub.add_parser('report')
    sub.add_parser('preview')
    for command in sub.choices.values():command.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.project,args.source,args.output,args.count)
    elif args.command=='baseline':baseline(args.project,args.output)
    elif args.command=='v3':v3(args.model,args.output)
    elif args.command=='reconstruct':reconstruct(args.parent,args.metadata,args.output)
    elif args.command=='preview':preview(args.output)
    else:report(args.output)


if __name__=='__main__':main()
