"""Kaggle-only material ControlNet + existing LoRA, frozen validated semantic reward."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

from experiments.facade_sd15_controlnet.data import digest,write_json
from experiments.facade_sd15_controlnet.train import git_revision,predict_noise
from experiments.facade_sd15_controlnet.tracking import RunTracker
from experiments.facade_semantic_v2.architecture import add_facade_lora,restore_lora,save_lora
from experiments.facade_semantic_v2.data import validate_dataset,selection_indices
from experiments.facade_semantic_v2.losses import facade_loss
from experiments.facade_semantic_v2.runtime import validate_export as validate_parent
from .architecture import upgrade_material_conditioning,make_teacher
from .data import MaterialDataset
from .losses import predicted_clean,semantic_feedback,material_statistics
from .validation import validation
from .teacher import teacher_accepted


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.checkpoint import checkpoint
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL,ControlNetModel,DDPMScheduler,UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTokenizer,CLIPTextModel
    from safetensors.torch import load_file
    if not torch.cuda.is_available():raise RuntimeError('Train on Kaggle CUDA; laptop inference is separate')
    config=json.loads(Path(args.config).read_text())
    if args.smoke:config.update(max_train_steps=2,warmup_steps=0,validation_steps=2,checkpoint_steps=2,inference_steps=2)
    if config['batch_size']!=1:raise ValueError('Physical batch 1 is required for bounded decoded-feedback memory')
    parent=Path(args.parent);parent_meta=validate_parent(parent)
    if parent_meta['weights_sha256']!=config['parent_weights_sha256'] or parent_meta['lora_sha256']!=config['parent_lora_sha256']:
        raise ValueError('Wrong V2 parent checkpoint or LoRA')
    manifest=validate_dataset(args.data)
    teacher_root=Path(args.teacher);teacher_meta=json.loads((teacher_root/'teacher.json').read_text())
    if teacher_meta['weights_sha256']!=digest(teacher_root/'teacher.safetensors') or teacher_meta['manifest_sha256']!=digest(Path(args.data)/'manifest.json'):
        raise ValueError('Teacher checksum or training/validation split mismatch')
    if not args.smoke and (not teacher_meta.get('feedback_accepted') or teacher_meta['smoke_only']
            or not teacher_accepted(teacher_meta['validation'],config)):
        raise ValueError('A validated, quality-gated teacher is required')
    accelerator=Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'],mixed_precision='fp16')
    if accelerator.num_processes!=1:raise ValueError('Single GPU process required')
    set_seed(config['seed']);device=accelerator.device;dtype=torch.float16
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    train_data=MaterialDataset(args.data,'train',augment=True,crop_probability=config['strip_crop_probability'],blank_probability=config['blank_crop_probability'])
    val_data=MaterialDataset(args.data,'val');ref_data=MaterialDataset(args.data,'train',augment=False)
    indices=selection_indices(val_data)
    if args.smoke:indices=indices[:1]
    contract={'config':config,'manifest_sha256':digest(Path(args.data)/'manifest.json'),'teacher_sha256':teacher_meta['weights_sha256'],
        'validation_ids':[val_data.rows[i]['id'] for i in indices]}
    state={'step':0,'epoch':0,'next_batch':0,'best_score':float('inf'),'best_step':None,'initial_score':None}
    if args.resume:
        state=json.loads((Path(args.resume)/'training-state.json').read_text())
        if state['contract']!=contract:raise ValueError('Resume training/data/teacher contract changed')
    common=dict(revision=config['base_revision'],variant='fp16',torch_dtype=dtype,use_safetensors=True)
    tokenizer=CLIPTokenizer.from_pretrained(config['base_model'],subfolder='tokenizer',revision=config['base_revision'])
    encoder=CLIPTextModel.from_pretrained(config['base_model'],subfolder='text_encoder',**common).to(device)
    vae=AutoencoderKL.from_pretrained(config['base_model'],subfolder='vae',**common).to(device)
    unet=UNet2DConditionModel.from_pretrained(config['base_model'],subfolder='unet',**common).to(device)
    scheduler_noise=DDPMScheduler.from_pretrained(config['base_model'],subfolder='scheduler',revision=config['base_revision'])
    control=upgrade_material_conditioning(ControlNetModel.from_pretrained(parent/'controlnet',torch_dtype=torch.float32))
    lora=add_facade_lora(unet,config['unet_lora_rank']);restore_lora(unet,parent/'lora')
    teacher=make_teacher(config['teacher_width']);teacher.load_state_dict(load_file(str(teacher_root/'teacher.safetensors')));teacher.to(device)
    for frozen in (teacher,vae,encoder):frozen.requires_grad_(False);frozen.eval()
    unet.train();control.train();unet.enable_gradient_checkpointing();control.enable_gradient_checkpointing()
    import bitsandbytes as bnb
    parameters=list(control.parameters())+lora
    optimizer=bnb.optim.AdamW8bit([{'params':list(control.parameters()),'lr':config['learning_rate']},
        {'params':lora,'lr':config['lora_learning_rate']}],weight_decay=.01,eps=1e-8)
    lr_scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=config['warmup_steps'],num_training_steps=config['max_train_steps'])
    control,optimizer,lr_scheduler=accelerator.prepare(control,optimizer,lr_scheduler)
    ids=tokenizer([config['prompt']],max_length=tokenizer.model_max_length,padding='max_length',truncation=True,return_tensors='pt').input_ids.to(device)
    with torch.no_grad():embeddings=encoder(ids)[0]
    del encoder;torch.cuda.empty_cache()
    tracker=RunTracker(output,config['experiment_name'],{**config,'manifest_sha256':contract['manifest_sha256'],
        'teacher_sha256':contract['teacher_sha256'],'training_samples':len(train_data),'dataset_counts':manifest['counts']},
        {'git_commit':git_revision(),'git_branch':'codex/facade-material-controlnet-v3','mode':'smoke' if args.smoke else 'train',
         'gpu':torch.cuda.get_device_name(0),'teacher_run_id':teacher_meta['run_id'],'test_used':False},tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    if args.resume:
        accelerator.load_state(str(args.resume));restore_lora(unet,Path(args.resume)/'lora')
        tracker.client.set_tag(tracker.run_id,'resumed_from_run_id',state['run_id'])
    for name,value in [('config.json',config),('contract.json',contract),('teacher.json',teacher_meta)]:
        write_json(output/name,value);tracker.artifact(output/name)
    for name in ('manifest.json','audit.json','ATTRIBUTION.md','irf-credits.json'):tracker.artifact(Path(args.data)/name)
    material_audit=Path(args.data)/'material-reference-audit.json'
    if material_audit.is_file():tracker.artifact(material_audit)
    def validate(step):
        encoder=CLIPTextModel.from_pretrained(config['base_model'],subfolder='text_encoder',**common).to(device)
        try:return validation(accelerator.unwrap_model(control),unet,vae,encoder,tokenizer,scheduler_noise,val_data,ref_data,teacher,config,output,step,tracker,device,indices)
        finally:del encoder;gc.collect();torch.cuda.empty_cache()
    def select(scores,step):
        if scores['val_selection_score']<state['best_score']:
            state.update(best_score=scores['val_selection_score'],best_step=step)
            accelerator.unwrap_model(control).save_pretrained(output/'best-controlnet',safe_serialization=True)
            save_lora(unet,output/'best-lora')
    step,epoch,skip=state['step'],state['epoch'],state['next_batch'];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    counts={'microbatches':0,'feedback_microbatches':0,'blank_crops':0,'reference_fallbacks':0,
        'reference_unavailable':0,'reference_full_photo':0}
    try:
        if not args.resume:
            print('FULL DIFFUSION TRAINING STARTED' if not args.smoke else 'SMOKE DIFFUSION STARTED',flush=True)
            scores=validate(0);state['initial_score']=scores['val_selection_score'];select(scores,0)
        sums={};n=0
        while step<config['max_train_steps']:
            train_data.epoch=epoch
            loader=accelerator.prepare(DataLoader(train_data,batch_size=1,sampler=train_data.training_order(epoch),num_workers=0,
                generator=torch.Generator().manual_seed(config['seed']+epoch)))
            batches=accelerator.skip_first_batches(loader,skip) if skip else loader
            for index,batch in enumerate(batches,start=skip):
                with accelerator.accumulate(control):
                    with torch.no_grad():latent=vae.encode(batch['photo'].to(device,dtype=dtype)).latent_dist.sample()*vae.config.scaling_factor
                    noise=torch.randn_like(latent)
                    t=torch.randint(0,scheduler_noise.config.num_train_timesteps,(len(latent),),device=device).long()
                    # Smoke forces one low-noise feedback pass to exercise the full gradient path.
                    if args.smoke:t.fill_(100)
                    noisy=scheduler_noise.add_noise(latent,noise,t)
                    labels,valid=batch['labels'].to(device),batch['valid'].to(device)
                    with accelerator.autocast():
                        prediction=predict_noise(control,unet,noisy,t,embeddings,batch['condition'].to(device,dtype=dtype))
                        target=noise if scheduler_noise.config.prediction_type=='epsilon' else scheduler_noise.get_velocity(latent,noise,t)
                        denoise=facade_loss(prediction,target,labels,valid,t,scheduler_noise,config)
                        loss=denoise;parts={'noise_loss':float(denoise.detach())}
                        if int(t[0])<=config['feedback_max_timestep']:
                            clean=predicted_clean(noisy,prediction,t,scheduler_noise).clamp(-4,4).to(dtype)/vae.config.scaling_factor
                            decoded=checkpoint(lambda z:vae.decode(z).sample,clean,use_reentrant=False).clamp(-1,1)
                            small=F.interpolate(decoded,(256,256),mode='bilinear',align_corners=False)
                            logits=teacher(small)
                            semantic,false_opening=semantic_feedback(logits,labels,valid)
                            material_valid=valid*batch['reference_available'].to(device)[:,None,None]
                            material=material_statistics(decoded,batch['reference'].to(device),labels,material_valid)
                            loss=loss+config['semantic_loss_weight']*semantic+config['false_opening_loss_weight']*false_opening+config['material_loss_weight']*material
                            counts['feedback_microbatches']+=1
                            parts.update(semantic_loss=float(semantic.detach()),false_opening_loss=float(false_opening.detach()),material_loss=float(material.detach()))
                    accelerator.backward(loss);counts['microbatches']+=1;counts['blank_crops']+=int(batch['blank_crop'].sum());counts['reference_fallbacks']+=int(batch['reference_median_fallback'].sum())
                    counts['reference_unavailable']+=int((~batch['reference_available']).sum())
                    counts['reference_full_photo']+=int(batch['reference_from_full_photo'].sum())
                    parts['train_loss']=float(loss.detach())
                    for k,v in parts.items():sums[k]=sums.get(k,0.)+v
                    n+=1
                    if accelerator.sync_gradients:accelerator.clip_grad_norm_(parameters,config['max_grad_norm'])
                    optimizer.step();lr_scheduler.step();optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:continue
                if accelerator.optimizer_step_was_skipped:sums={};n=0;continue
                step+=1
                metrics={k:v/n for k,v in sums.items()};sums={};n=0
                metrics.update(learning_rate=lr_scheduler.get_last_lr()[0],peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    elapsed_seconds=time.perf_counter()-started,**counts)
                metrics['reference_kernel_l2']=float(accelerator.unwrap_model(control).controlnet_cond_embedding.conv_in.weight[:,7:].detach().float().norm())
                tracker.metrics(metrics,step);print(json.dumps({'step':step,**metrics}),flush=True)
                if step%config['validation_steps']==0 or step==config['max_train_steps']:select(validate(step),step)
                if step%config['checkpoint_steps']==0 or step==config['max_train_steps']:
                    checkpoint_root=output/'checkpoints';checkpoint_root.mkdir(exist_ok=True)
                    old_directories=sorted(checkpoint_root.glob('step_*'))
                    for obsolete in old_directories[:max(0,len(old_directories)-config['keep_checkpoints']+1)]:
                        if not obsolete.resolve().is_relative_to(checkpoint_root.resolve()):raise ValueError('Invalid checkpoint path')
                        shutil.rmtree(obsolete)
                    directory=checkpoint_root/f'step_{step:06d}';accelerator.save_state(str(directory));save_lora(unet,directory/'lora')
                    write_json(directory/'training-state.json',{**state,'step':step,'epoch':epoch,'next_batch':index+1,'contract':contract,'run_id':tracker.run_id})
                if step>=config['max_train_steps']:break
            epoch+=1;skip=0
        exported=ControlNetModel.from_pretrained(output/'best-controlnet',torch_dtype=torch.float16)
        exported.save_pretrained(output/'export/controlnet',safe_serialization=True)
        shutil.copytree(output/'best-lora',output/'export/lora',dirs_exist_ok=True)
        metadata={'schema_version':3,'architecture':config['architecture'],'config':config,'git_commit':git_revision(),
            'run_id':tracker.run_id,'manifest_sha256':contract['manifest_sha256'],'teacher_sha256':contract['teacher_sha256'],
            'teacher_validation':teacher_meta['validation'],'selected_step':state['best_step'],'initial_score':state['initial_score'],'selected_score':state['best_score'],
            'improved_selection_score':state['best_step']!=0,'selection':'Source-macro teacher opening consistency plus 0.25 target-reference-assisted LPIPS; teacher is not independent. Test untouched.',
            'quality_status':('SMOKE ONLY; not quality trained' if args.smoke else
                'NO IMPROVEMENT; retained V2 initialization, not a trained V3 upgrade' if state['best_step']==0 else
                'experimental trained V3; held-out and local house evaluation required'),
            'completed_optimizer_steps':step,
            'weights_sha256':digest(output/'export/controlnet/diffusion_pytorch_model.safetensors'),
            'lora_sha256':digest(output/'export/lora/facade_lora.safetensors'),'training_counts':counts}
        write_json(output/'export/model.json',metadata);tracker.artifact(output/'export/model.json','model')
        if not args.smoke and state['best_step']==0:
            # Keep a genuine trained candidate for diagnosis instead of silently
            # distributing the untouched parent as the outcome of full training.
            latest=output/'trained-candidate'
            accelerator.unwrap_model(control).to(dtype=torch.float16).save_pretrained(latest/'controlnet',safe_serialization=True)
            save_lora(unet,latest/'lora')
            candidate={**metadata,'selected_step':step,'improved_selection_score':False,
                'quality_status':'EXPERIMENTAL NONSELECTED trained candidate; failed to improve validation selection',
                'weights_sha256':digest(latest/'controlnet/diffusion_pytorch_model.safetensors'),
                'lora_sha256':digest(latest/'lora/facade_lora.safetensors')}
            write_json(latest/'model.json',candidate)
            tracker.artifact(latest/'model.json','nonselected-candidate')
        print('DIFFUSION TRAINING FINISHED: '+json.dumps({'completed_optimizer_steps':step,
            'selected_step':state['best_step'],'improved_selection_score':state['best_step']!=0}),flush=True)
        tracker.finish()
    except BaseException:
        tracker.finish('FAILED');raise
    finally:accelerator.end_training()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ('data','parent','teacher','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume')
    train(parser.parse_args())
