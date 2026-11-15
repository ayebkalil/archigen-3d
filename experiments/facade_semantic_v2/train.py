"""Single-GPU semantic ControlNet + rank-8 UNet LoRA; reproducible AMP checkpoints."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.facade_sd15_controlnet.data import digest, write_json
from experiments.facade_sd15_controlnet.train import git_revision, predict_noise
from experiments.facade_sd15_controlnet.tracking import RunTracker
from .architecture import upgrade_conditioning, add_facade_lora, save_lora, restore_lora
from .data import FacadeDataset, validate_dataset, selection_indices
from .losses import facade_loss


def validation(controlnet, unet, vae, encoder, tokenizer, scheduler, data, config, output, step, tracker, device, indices):
    import numpy as np
    import torch
    from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
    from .evaluate import pair_metrics, panel
    controlnet.eval(); unet.eval()
    pipeline = StableDiffusionControlNetPipeline(vae=vae, text_encoder=encoder, tokenizer=tokenizer,
        unet=unet, controlnet=controlnet, scheduler=DPMSolverMultistepScheduler.from_config(scheduler.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
    pipeline.enable_vae_slicing(); pipeline.enable_vae_tiling(); pipeline.set_progress_bar_config(disable=True)
    rows = []
    for index in indices:
        photo, labels, valid, channels = data.images(index)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            generated = pipeline(config['prompt'], image=torch.from_numpy(channels)[None],
                negative_prompt=config['negative_prompt'], height=512, width=512,
                num_inference_steps=config['inference_steps'], guidance_scale=config['guidance_scale'],
                controlnet_conditioning_scale=config['controlnet_conditioning_scale'],
                generator=torch.Generator(device=device).manual_seed(config['seed'] + index)).images[0]
        rows.append({'id': data.rows[index]['id'], 'source': data.rows[index]['source'],
                     **pair_metrics(generated, photo, labels, valid)})
        if len(rows) <= 4:
            path = output / 'validation' / f'step_{step:06d}_{len(rows)}.png'
            panel(photo, labels, generated, path); tracker.artifact(path, f'validation/step_{step:06d}')
    metrics = {}
    for source in sorted({row['source'] for row in rows}):
        selected = [row for row in rows if row['source'] == source]
        for key in ('lpips', 'ssim', 'psnr', 'opening_edge_mae'):
            metrics[f'val_{source}_{key}'] = float(np.mean([row[key] for row in selected]))
    metrics['val_macro_lpips'] = float(np.mean([value for key, value in metrics.items() if key.endswith('_lpips')]))
    path = output / 'validation' / f'step_{step:06d}.json'; write_json(path, rows)
    tracker.artifact(path, 'validation'); tracker.metrics(metrics, step)
    controlnet.train(); unet.train()
    del pipeline; gc.collect(); torch.cuda.empty_cache()
    return metrics


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config.update(max_train_steps=2, warmup_steps=0, inference_steps=2, validation_steps=2, checkpoint_steps=2)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires Kaggle CUDA')
    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'], mixed_precision='fp16')
    if accelerator.num_processes != 1:
        raise ValueError('Single GPU required; do not launch with torchrun')
    set_seed(config['seed'])
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    data_root, parent = Path(args.data).resolve(), Path(args.parent).resolve()
    manifest = validate_dataset(data_root)
    if digest(parent / 'controlnet/diffusion_pytorch_model.safetensors') != config['init_checkpoint_sha256']:
        raise ValueError('Parent checkpoint differs from the selected V1 step 750')
    train_data = FacadeDataset(data_root, 'train', augment=True, crop_probability=config['strip_crop_probability'])
    val_data = FacadeDataset(data_root, 'val')
    indices = selection_indices(val_data)
    if args.smoke:
        indices = indices[:1]
    contract = {'config': config, 'manifest_sha256': digest(data_root / 'manifest.json'),
                'validation_ids': [val_data.rows[index]['id'] for index in indices]}
    state = {'step': 0, 'epoch': 0, 'next_batch': 0, 'best_lpips': math.inf, 'best_step': None}
    if args.resume:
        state = json.loads((Path(args.resume) / 'training-state.json').read_text())
        if state['contract'] != contract:
            raise ValueError('Resume config/data/validation contract changed')
    dtype = torch.float16
    common = dict(revision=config['base_revision'], variant='fp16', torch_dtype=dtype, use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(config['base_model'], subfolder='tokenizer', revision=config['base_revision'])
    encoder = CLIPTextModel.from_pretrained(config['base_model'], subfolder='text_encoder', **common).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(config['base_model'], subfolder='vae', **common).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(config['base_model'], subfolder='unet', **common).to(accelerator.device)
    scheduler_noise = DDPMScheduler.from_pretrained(config['base_model'], subfolder='scheduler', revision=config['base_revision'])
    controlnet = upgrade_conditioning(ControlNetModel.from_pretrained(parent / 'controlnet', torch_dtype=torch.float32))
    lora_parameters = add_facade_lora(unet, config['unet_lora_rank'])
    for model in (vae, encoder):
        model.requires_grad_(False); model.eval()
    unet.train(); controlnet.train()
    if config['gradient_checkpointing']:
        unet.enable_gradient_checkpointing(); controlnet.enable_gradient_checkpointing()
    if config['adam_8bit']:
        import bitsandbytes as bnb
        optimizer_type = bnb.optim.AdamW8bit
    else:
        optimizer_type = torch.optim.AdamW
    parameters = list(controlnet.parameters()) + lora_parameters
    optimizer = optimizer_type([{'params': list(controlnet.parameters()), 'lr': config['learning_rate']},
        {'params': lora_parameters, 'lr': config['lora_learning_rate']}], betas=(.9, .999), weight_decay=.01, eps=1e-8)
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_train_steps'])
    # UNet stays outside Accelerate's model registry: checkpoints contain adapters,
    # not another 1.7 GB frozen base. Single-process gradients share this optimizer.
    controlnet, optimizer, scheduler = accelerator.prepare(controlnet, optimizer, scheduler)
    ids = tokenizer([config['prompt']], max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(accelerator.device)
    with torch.no_grad():
        embeddings = encoder(ids)[0]
    del encoder; torch.cuda.empty_cache()
    tracker = RunTracker(output, config['experiment_name'], {**config, 'manifest_sha256': contract['manifest_sha256'],
        'dataset_counts': manifest['splits'], 'lora_parameters': sum(p.numel() for p in lora_parameters)},
        {'git_commit': git_revision(), 'git_branch': 'codex/facade-semantic-controlnet-v2',
         'mode': 'smoke' if args.smoke else 'train', 'gpu': torch.cuda.get_device_name(0)},
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    if args.resume:
        accelerator.load_state(str(args.resume)); restore_lora(unet, Path(args.resume) / 'lora')
        tracker.client.set_tag(tracker.run_id, 'resumed_from_run_id', state['run_id'])
    write_json(output / 'config.json', config); write_json(output / 'contract.json', contract)
    for path in (output / 'config.json', output / 'contract.json', data_root / 'manifest.json', data_root / 'audit.json', data_root / 'ATTRIBUTION.md'):
        tracker.artifact(path)
    (output / 'requirements-freeze.txt').write_text(subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout)
    tracker.artifact(output / 'requirements-freeze.txt')
    step, epoch, skip = state['step'], state['epoch'], state['next_batch']
    started, loss_sum, microbatches = time.perf_counter(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    try:
        while step < config['max_train_steps']:
            train_data.epoch = epoch
            loader = accelerator.prepare(DataLoader(train_data, batch_size=config['batch_size'],
                sampler=train_data.training_order(epoch, config['source_balance']), num_workers=0,
                generator=torch.Generator().manual_seed(config['seed'] + epoch)))
            batches = accelerator.skip_first_batches(loader, skip) if skip else loader
            for index, batch in enumerate(batches, start=skip):
                with accelerator.accumulate(controlnet):
                    with torch.no_grad():
                        latent = vae.encode(batch['photo'].to(accelerator.device, dtype=dtype)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latent)
                    t = torch.randint(0, scheduler_noise.config.num_train_timesteps, (len(latent),), device=latent.device).long()
                    noisy = scheduler_noise.add_noise(latent, noise, t)
                    with accelerator.autocast():
                        prediction = predict_noise(controlnet, unet, noisy, t, embeddings.expand(len(latent), -1, -1), batch['condition'].to(latent.device, dtype=dtype))
                        target = noise if scheduler_noise.config.prediction_type == 'epsilon' else scheduler_noise.get_velocity(latent, noise, t)
                        loss = facade_loss(prediction, target, batch['labels'].to(latent.device), batch['valid'].to(latent.device), t, scheduler_noise, config)
                    accelerator.backward(loss)
                    loss_sum += loss.item(); microbatches += 1
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, config['max_grad_norm'])
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                mean_loss = loss_sum / microbatches; loss_sum, microbatches = 0., 0
                if accelerator.optimizer_step_was_skipped:
                    continue
                step += 1
                metrics = {'train_loss': mean_loss, 'learning_rate': scheduler.get_last_lr()[0],
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2, 'elapsed_seconds': time.perf_counter() - started}
                tracker.metrics(metrics, step); print(json.dumps({'step': step, **metrics}), flush=True)
                model = accelerator.unwrap_model(controlnet)
                if step % config['validation_steps'] == 0 or step == config['max_train_steps']:
                    encoder = CLIPTextModel.from_pretrained(config['base_model'], subfolder='text_encoder', **common).to(accelerator.device)
                    scores = validation(model, unet, vae, encoder, tokenizer, scheduler_noise, val_data, config, output, step, tracker, accelerator.device, indices)
                    del encoder; torch.cuda.empty_cache()
                    if scores['val_macro_lpips'] < state['best_lpips']:
                        state.update(best_lpips=scores['val_macro_lpips'], best_step=step)
                        model.save_pretrained(output / 'best-controlnet', safe_serialization=True)
                        save_lora(unet, output / 'best-lora')
                if step % config['checkpoint_steps'] == 0 or step == config['max_train_steps']:
                    directory = output / 'checkpoints' / f'step_{step:06d}'
                    # Prune previous optimizer state first to keep below Kaggle's output quota.
                    directories = sorted((output / 'checkpoints').glob('step_*'))
                    prune_count = max(0, len(directories) - config['keep_checkpoints'] + 1)
                    for obsolete in directories[:prune_count]:
                        if not obsolete.resolve().is_relative_to((output / 'checkpoints').resolve()):
                            raise ValueError('Invalid checkpoint path')
                        shutil.rmtree(obsolete)
                    accelerator.save_state(str(directory)); save_lora(unet, directory / 'lora')
                    write_json(directory / 'training-state.json', {**state, 'step': step, 'epoch': epoch,
                        'next_batch': index + 1, 'contract': contract, 'run_id': tracker.run_id})
                if step >= config['max_train_steps']:
                    break
            epoch += 1; skip = 0
        exported = ControlNetModel.from_pretrained(output / 'best-controlnet', torch_dtype=torch.float16)
        exported.save_pretrained(output / 'export/controlnet', safe_serialization=True)
        shutil.copytree(output / 'best-lora', output / 'export/lora', dirs_exist_ok=True)
        write_json(output / 'export/model.json', {'schema_version': 2, 'architecture': config['architecture'],
            'config': config, 'git_commit': git_revision(), 'run_id': tracker.run_id,
            'manifest_sha256': contract['manifest_sha256'], 'selected_step': state['best_step'],
            'selection': 'lowest source-macro LPIPS on fixed stratified validation subset; tests untouched',
            'quality_status': 'SMOKE ONLY; not quality trained' if args.smoke else 'experimental; matched parent evaluation and local wall test required',
            'weights_sha256': digest(output / 'export/controlnet/diffusion_pytorch_model.safetensors'),
            'lora_sha256': digest(output / 'export/lora/facade_lora.safetensors')})
        tracker.artifact(output / 'export/model.json', 'model'); tracker.finish()
    except BaseException:
        tracker.finish('FAILED'); raise
    finally:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True); parser.add_argument('--parent', required=True)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output', default='/kaggle/working/facade-semantic-v2')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--resume')
    train(parser.parse_args())


if __name__ == '__main__': main()
