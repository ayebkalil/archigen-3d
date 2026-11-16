"""Reproduce the selected Kaggle V3 results and a matched production comparison.

Uses actual saved validation renders, never generated illustrations. The first
three saved facade tasks are fixed by training's predeclared validation order.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import write_json
from experiments.facade_semantic_v2.conditioning import cmp_labels, letterbox
from src.data.facade_palette import CMP_TRAINING_LUT


def prepare(artifacts, source, output):
    artifacts, source, output = Path(artifacts), Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata=json.loads((artifacts/'export/model.json').read_text())
    if metadata['selected_step']<=0 or metadata['completed_optimizer_steps']!=1500:
        raise ValueError('Expected a completed, genuinely trained V3 result')
    selected=artifacts/'validation'/f"step_{metadata['selected_step']:06d}"
    metrics=json.loads((selected/'metrics.json').read_text())
    initial=json.loads((artifacts/'validation/step_000000/metrics.json').read_text())
    original=json.loads((source/'manifest.json').read_text())
    lookup={'CMP/'+r['id']:r for r in original['samples'] if r['split']=='val'}
    tasks=[r for r in metrics['rows'] if r['protocol']=='independent_training_reference']
    cases=[]
    for i,row in enumerate(tasks[:3]):
        if row['source']!='CMP' or row['task']!='facade':raise ValueError('Expected fixed first three CMP facades')
        raw=lookup[row['id']]
        photo=Image.open(source/raw['photo']).convert('RGB')
        indexed=np.asarray(Image.open(source/raw['labels']))
        canonical=cmp_labels(indexed)
        yy,xx=np.where(np.isin(canonical,[1,2,3,4]))
        box=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
        pixels=np.asarray(photo).copy();pixels[canonical==0]=127
        target,indexed,_=letterbox(Image.fromarray(pixels).crop(box),indexed[box[1]:box[3],box[0]:box[2]])
        before=Image.open(artifacts/'validation/step_000000'/f'{i}_independent_training_reference.png').convert('RGB')
        after=Image.open(selected/f'{i}_independent_training_reference.png').convert('RGB')
        if before.size!=(1536,512) or after.size!=before.size:raise ValueError('Unexpected saved render dimensions')
        gt=after.crop((0,0,512,512))
        if not np.array_equal(target,gt) or not np.array_equal(before.crop((0,0,512,512)),gt):
            raise ValueError('Local labels and saved Kaggle target do not match exactly: '+row['id'])
        folder=output/f'case_{i:02d}';folder.mkdir(exist_ok=True)
        gt.save(folder/'target.png')
        Image.fromarray(CMP_TRAINING_LUT[indexed]).save(folder/'layout.png')
        small=np.asarray(Image.fromarray(indexed).resize((256,256),Image.Resampling.NEAREST))
        Image.fromarray(CMP_TRAINING_LUT[small]).save(folder/'production-input.png')
        before.crop((1024,0,1536,512)).save(folder/'initial.png')
        after.crop((1024,0,1536,512)).save(folder/'trained.png')
        after.crop((512,0,1024,512)).save(folder/'reference.png')
        cases.append({'folder':folder.name,'id':row['id'],'reference_id':row['reference_id'],
            'split':'val','target_matches_original_pixel_exact':True})
    write_json(output/'cases.json',cases)
    write_json(output/'summary.json',{'model':metadata,'initial_metrics':initial['metrics'],
        'selected_metrics':metrics['metrics'],'cases':cases,
        'protocol':{'validation_facades':{'CMP':12,'IRF':4},'blank_crops_per_source':2,
            'protocols':['target_wall_reference','independent_training_reference'],
            'test_used_for_selection':False,'independent_test':False,
            'teacher_scores':'same feedback teacher, not independent semantic accuracy',
            'paired_metrics':'Target-derived wall reference assists V3; no paired LPIPS for independent reference material.',
            'image_comparison':'First three predeclared tasks, independent training references, same layouts/SD seeds/25 inference steps. Local production uses original 12-class masks at native 256 and its default sharpening. V3 also receives RGB material reference; this is a disclosed unequal-conditioning qualitative comparison.'}})


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);summary=json.loads((output/'summary.json').read_text());cases=summary['cases']
    step=summary['model']['selected_step']
    headers=['Ground-truth photograph','Current project · Pix2Pix 256','V2 initialization · 512',f'Trained V3 · step {step:,}']
    files=['target.png','production.png','initial.png','trained.png']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(3,4,figsize=(16,12.5),facecolor='#f6f3ec')
    for i,case in enumerate(cases):
        for j,(filename,title) in enumerate(zip(files,headers)):
            axes[i,j].imshow(Image.open(output/case['folder']/filename),interpolation='nearest')
            axes[i,j].axis('off')
            if i==0:axes[i,j].set_title(title,fontsize=12,pad=12,fontweight='bold')
            if j==0:axes[i,j].text(0,-.04,case['id'],transform=axes[i,j].transAxes,fontsize=10)
    fig.suptitle('ArchiGen facade comparison · actual model outputs',fontsize=21,y=.985,fontweight='bold')
    fig.text(.5,.027,'Fixed first three CMP validation tasks; identical annotated layout. V2/V3: same seed and 25 inference steps.\n'
        'V3 receives an independent training-building material reference. Current project: native 256, default sharpening.\n'
        'No manual corrections. These validation images were used for model selection; this is not an independent test.',ha='center',fontsize=10)
    fig.text(.5,.006,'CMP Facade Database · Tylecek & Sara (2013) · CC BY-SA · https://cmp.felk.cvut.cz/~tylecr1/facade/',ha='center',fontsize=9)
    fig.subplots_adjust(top=.93,bottom=.095,left=.02,right=.98,wspace=.025,hspace=.06)
    fig.savefig(output/'comparison.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,12.5),facecolor='#f6f3ec')
    for i,case in enumerate(cases):
        for j,(filename,title) in enumerate(zip(['reference.png','initial.png','trained.png'],
                ['Requested material reference','V2 initialization','Trained V3'])):
            axes[i,j].imshow(Image.open(output/case['folder']/filename),interpolation='nearest');axes[i,j].axis('off')
            if i==0:axes[i,j].set_title(title,fontsize=12,pad=10)
    fig.suptitle('Independent material-reference check · same saved examples',fontsize=16,y=.98)
    fig.text(.5,.022,'References come from distinct training buildings. Learned reference response is weak in these examples;\n'
        'a better selection score does not prove control of wall finish or consistency across house sides.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.94,bottom=.075,left=.02,right=.98,hspace=.035,wspace=.025)
    fig.savefig(output/'material-reference-check.png',dpi=130);plt.close(fig)
    # All reported aggregate diagnostics, with explicit direction and protocol.
    rows=[]
    for key,initial in summary['initial_metrics'].items():
        if key not in summary['selected_metrics']:continue
        selected=summary['selected_metrics'][key]
        rows.append({'metric':key,'initial':initial,f'selected_step_{step}':selected,
            'relative_change_percent':100*(selected-initial)/initial if initial else None})
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--artifacts',required=True)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--project',required=True);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();prepare(args.artifacts,args.source,args.output)
    if not args.prepare_only:
        from .compare_current import baseline
        baseline(args.project,args.output);figures(args.output)
    print('Prepared matched actual model outputs:',args.output)
