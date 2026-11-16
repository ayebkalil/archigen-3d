"""Disclosed target-reference and independent-training-reference validation protocols."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.facade_sd15_controlnet.data import write_json
from experiments.facade_semantic_v2.evaluate import pair_metrics
from .teacher import segmentation_scores


def semantic_image_scores(teacher, image, labels, valid, device):
    import torch
    import torch.nn.functional as F
    pixels=torch.from_numpy(np.asarray(image,np.float32).transpose(2,0,1).copy())[None].to(device)/127.5-1
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        predicted=teacher(F.interpolate(pixels,(256,256),mode='bilinear',align_corners=False)).argmax(1)[0].cpu().numpy()
    target=np.asarray(Image.fromarray(labels).resize((256,256),Image.Resampling.NEAREST))
    mask=np.asarray(Image.fromarray(valid.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))>0
    confusion=np.bincount((target[mask]*6+predicted[mask]).ravel(),minlength=36).reshape(6,6)
    return segmentation_scores(confusion)


def reference_colour_error(image, reference, labels, valid, available=True):
    wall=(labels==1)&(valid>0)
    if not available or not wall.any():return None
    return float(np.abs(np.asarray(image,np.float32)[wall].mean(0)-np.asarray(reference,np.float32)[wall].mean(0)).mean()/255)


def validation(controlnet,unet,vae,encoder,tokenizer,scheduler,data,train_reference_data,teacher,config,output,step,tracker,device,indices):
    import torch
    from diffusers import StableDiffusionControlNetPipeline,DPMSolverMultistepScheduler
    controlnet.eval();unet.eval();teacher.eval()
    pipe=StableDiffusionControlNetPipeline(vae=vae,unet=unet,controlnet=controlnet,text_encoder=encoder,
        tokenizer=tokenizer,scheduler=DPMSolverMultistepScheduler.from_config(scheduler.config),
        safety_checker=None,feature_extractor=None,requires_safety_checker=False).to(device)
    pipe.enable_vae_slicing();pipe.enable_vae_tiling();pipe.set_progress_bar_config(disable=True)
    rows=[];directory=Path(output)/'validation'/f'step_{step:06d}';directory.mkdir(parents=True,exist_ok=True)
    try:
        tasks=[(i,False) for i in indices]
        blank_counts={'CMP':0,'IRF':0}
        for i,row in enumerate(data.rows):
            if blank_counts[row['source']] >= (1 if len(indices)==1 else 2):continue
            if data.sample(i,force_blank=True)[5]:
                tasks.append((i,True));blank_counts[row['source']]+=1
        if len(indices)>1 and sum(blank_counts.values())<2:raise ValueError('Insufficient real blank-wall validation coverage')
        for order,(index,blank_task) in enumerate(tasks):
            # Reference is exclusively from the training split and a distinct building.
            other=(index*97+13)%len(train_reference_data)
            for _ in range(len(train_reference_data)):
                reference_sample = train_reference_data.sample(other, include_reference_info=True)
                if (train_reference_data.rows[other]['group'] != data.rows[index]['group']
                        and reference_sample[7]['available']): break
                other=(other+1)%len(train_reference_data)
            else: raise ValueError('No independent training building has an annotated wall reference')
            ref_photo,ref_labels,*_=reference_sample
            initial_image=None
            for protocol in ('target_wall_reference','independent_training_reference'):
                photo,labels,valid,semantic,reference,_,fallback,reference_info=data.sample(index,
                    reference_photo=ref_photo if protocol.startswith('independent') else None,
                    reference_labels=ref_labels if protocol.startswith('independent') else None,force_blank=blank_task,
                    include_reference_info=True)
                condition=torch.cat([torch.from_numpy(semantic),torch.from_numpy(np.asarray(reference,np.float32).transpose(2,0,1).copy())/255])[None]
                if step==0 and initial_image is not None:
                    image=initial_image
                else:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        image=pipe(config['prompt'],image=condition,negative_prompt=config['negative_prompt'],
                            height=512,width=512,num_inference_steps=config['inference_steps'],guidance_scale=config['guidance_scale'],
                            controlnet_conditioning_scale=config['controlnet_conditioning_scale'],
                            generator=torch.Generator(device='cpu').manual_seed(config['seed']+index)).images[0]
                    if step==0:initial_image=image
                semantic_scores=semantic_image_scores(teacher,image,labels,valid,device)
                row={'id':data.rows[index]['id'],'source':data.rows[index]['source'],'protocol':protocol,
                    'task':'real_blank_wall_crop' if blank_task else 'facade',
                    'reference_id':train_reference_data.rows[other]['id'] if protocol.startswith('independent') else data.rows[index]['id'],
                    'reference_median_fallback':fallback,
                    'reference_available':reference_info['available'], 'reference_from_full_photo':reference_info['from_full_photo'],
                    **{'teacher_'+k:v for k,v in semantic_scores.items() if isinstance(v,(int,float))},
                    'reference_colour_mae':reference_colour_error(image,reference,labels,valid,reference_info['available'])}
                if protocol=='target_wall_reference':row.update(pair_metrics(image,photo,labels,valid))
                rows.append(row)
                if order<4 or blank_task:
                    # Ground truth, explicit material reference, generated output; never silently retouched.
                    canvas=Image.new('RGB',(1536,512));canvas.paste(photo,(0,0));canvas.paste(reference,(512,0));canvas.paste(image,(1024,0))
                    file=directory/f'{order}_{protocol}.png';canvas.save(file);tracker.artifact(file,f'validation/step_{step:06d}')
        metrics={}
        for protocol in ('target_wall_reference','independent_training_reference'):
            for source in ('CMP','IRF'):
                selected=[row for row in rows if row['protocol']==protocol and row['source']==source]
                if not selected:continue
                for key in ('lpips','ssim','opening_edge_mae','teacher_opening_iou','teacher_wall_false_opening_rate','reference_colour_mae'):
                    values=[row[key] for row in selected if row.get(key) is not None]
                    if values:metrics[f'val_{protocol}_{source}_{key}']=float(np.mean(values))
        # Equal weighting by source and protocol; teacher reward is not an independent semantic benchmark.
        geometry=[1-row['teacher_opening_iou']+3*row['teacher_wall_false_opening_rate'] for row in rows]
        per_source=[np.mean([geometry[i] for i,row in enumerate(rows) if row['source']==source]) for source in ('CMP','IRF') if any(r['source']==source for r in rows)]
        assisted=[np.mean([r['lpips'] for r in rows if r['protocol']=='target_wall_reference' and r['source']==source]) for source in ('CMP','IRF') if any(r['source']==source and 'lpips'in r for r in rows)]
        metrics['val_selection_score']=float(np.mean(per_source)+.25*np.mean(assisted))
        metrics['val_reference_unavailable_rows']=sum(not row['reference_available'] for row in rows)
        metrics['val_reference_full_photo_rows']=sum(row['reference_from_full_photo'] for row in rows)
        blanks=[r for r in rows if r['task']=='real_blank_wall_crop']
        if blanks:
            metrics['val_blank_wall_teacher_false_opening_rate']=float(np.mean([r['teacher_wall_false_opening_rate'] for r in blanks]))
        path=directory/'metrics.json';write_json(path,{'protocol_notes':'Target-reference metrics use target-derived wall crops. Independent references come from training only; no paired LPIPS for mismatched target material. Teacher metrics use the same model as training feedback; not independent accuracy.', 'rows':rows,'metrics':metrics})
        tracker.artifact(path,f'validation/step_{step:06d}');tracker.metrics(metrics,step)
        print(json.dumps({'validation_step':step,**metrics}),flush=True)
        return metrics
    finally:
        controlnet.train();unet.train();del pipe;torch.cuda.empty_cache()
