"""Single-GPU ControlNet fine-tuning; frozen SD1.5 prior, resumable AMP training."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .data import FacadePairs, digest, validate_dataset, write_json
from .tracking import RunTracker


def opening_weighted_loss(prediction, target, openings, weight=1.0):
    import torch.nn.functional as F
    # Weight the latent locations covering doors/windows and a small surrounding band.
    focus = F.interpolate(F.max_pool2d(openings.float(), 7, 1, 3), size=prediction.shape[-2:], mode='nearest')
    weights = 1 + weight * focus
    return ((prediction.float() - target.float()).square() * weights).sum() / (weights.sum() * prediction.shape[1])


def git_revision():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def predict_noise(controlnet, unet, noisy, timesteps, hidden, condition):
    down, middle = controlnet(noisy, timesteps, encoder_hidden_states=hidden,
        controlnet_cond=condition, return_dict=False)
    return unet(noisy, timesteps, encoder_hidden_states=hidden,
        down_block_additional_residuals=[value.to(noisy.dtype) for value in down],
        mid_block_additional_residual=middle.to(noisy.dtype)).sample


def validation(controlnet, unet, vae, text_encoder, tokenizer, scheduler, dataset, config, output, step, tracker, device):
    import numpy as np
    import torch
    from diffusers import StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
    from .evaluate import image_metrics, comparison
    controlnet.eval(); unet.eval()
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=DPMSolverMultistepScheduler.from_config(scheduler.config), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_vae_slicing(); pipeline.enable_vae_tiling()
    scores = []
    count = min(config['validation_count'], len(dataset))
    for index in range(count):
        photo, condition, openings = dataset.images(index)
        generator = torch.Generator(device=device).manual_seed(config['seed'] + index)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            generated = pipeline(config['prompt'], image=condition, negative_prompt=config['negative_prompt'],
                generator=generator, num_inference_steps=config['inference_steps'], guidance_scale=config['guidance_scale'],
                controlnet_conditioning_scale=config['controlnet_conditioning_scale'], height=512, width=512).images[0]
        scores.append(image_metrics(generated, photo, openings))
        path = output / 'validation' / f'step_{step:06d}_{index}.png'
        comparison(condition, generated, photo, path)
        tracker.artifact(path, f'validation/step_{step:06d}')
    means = {f'val_{key}': float(np.mean([row[key] for row in scores])) for key in scores[0]}
    tracker.metrics(means, step)
    controlnet.train(); unet.train()
    del pipeline
    gc.collect(); torch.cuda.empty_cache()
    return means


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke:
        config.update(max_train_steps=2, warmup_steps=0, validation_count=1, inference_steps=2, validation_steps=0)
    if not torch.cuda.is_available(): raise RuntimeError('Training requires a CUDA GPU. Use --check-data for an offline check.')
    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'], mixed_precision=config['mixed_precision'])
    if accelerator.num_processes != 1: raise ValueError('Run on one Kaggle GPU; two T4 VRAM capacities do not combine.')
    set_seed(config['seed'])
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data).resolve()
    manifest = validate_dataset(data_root)
    manifest_hash = digest(data_root / 'manifest.json')
    contract = {'config': config, 'manifest_sha256': manifest_hash}
    if args.resume:
        state = json.loads((Path(args.resume) / 'training-state.json').read_text())
        if state['contract'] != contract: raise ValueError('Resume configuration or dataset differs; start a new experiment.')
    else: state = {'step': 0, 'epoch': 0, 'next_batch': 0, 'best_lpips': math.inf, 'best_step': None}

    dtype = torch.float16
    common = {'revision': config['base_revision'], 'variant': 'fp16', 'torch_dtype': dtype, 'use_safetensors': True}
    tokenizer = CLIPTokenizer.from_pretrained(config['base_model'], subfolder='tokenizer', revision=config['base_revision'])
    text_encoder = CLIPTextModel.from_pretrained(config['base_model'], subfolder='text_encoder', **common)
    vae = AutoencoderKL.from_pretrained(config['base_model'], subfolder='vae', **common)
    unet = UNet2DConditionModel.from_pretrained(config['base_model'], subfolder='unet', **common)
    noise_scheduler = DDPMScheduler.from_pretrained(config['base_model'], subfolder='scheduler', revision=config['base_revision'])
    controlnet = ControlNetModel.from_pretrained(config['controlnet_model'], revision=config['controlnet_revision'],
        variant='fp16', use_safetensors=True, torch_dtype=torch.float32)
    for model in (text_encoder, vae, unet): model.requires_grad_(False); model.to(accelerator.device)
    vae.eval(); text_encoder.eval(); unet.train(); controlnet.train()
    if config['gradient_checkpointing']:
        controlnet.enable_gradient_checkpointing(); unet.enable_gradient_checkpointing()
    if config['adam_8bit']:
        import bitsandbytes as bnb
        optimizer_type = bnb.optim.AdamW8bit
    else: optimizer_type = torch.optim.AdamW
    optimizer = optimizer_type(controlnet.parameters(), lr=config['learning_rate'], betas=(.9, .999), weight_decay=.01, eps=1e-8)
    scheduler = get_scheduler('cosine', optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['max_train_steps'])
    controlnet, optimizer, scheduler = accelerator.prepare(controlnet, optimizer, scheduler)
    train_data = FacadePairs(data_root, 'train', augment=True)
    val_data = FacadePairs(data_root, 'val')
    ids = tokenizer([config['prompt']], max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(accelerator.device)
    with torch.no_grad(): embeddings = text_encoder(ids)[0]
    del text_encoder
    # Validation reloads this small frozen encoder only when needed, saving GPU memory during training.
    tracker = RunTracker(output, config['experiment_name'], {**config, 'manifest_sha256': manifest_hash,
        'train_count': manifest['splits']['train'], 'validation_selection': 'first sorted val IDs; test untouched'},
        {'mlflow.runName': f'controlnet-{git_revision()[:8]}', 'git_commit': git_revision(),
         'git_branch': 'codex/facade-controlnet-kaggle-mlflow', 'mode': 'smoke' if args.smoke else 'train',
         'gpu': torch.cuda.get_device_name(0)}, tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    if args.resume:
        accelerator.load_state(str(args.resume))
        tracker.client.set_tag(tracker.run_id, 'resumed_from_run_id', state['run_id'])
    write_json(output / 'config.json', config)
    tracker.artifact(output / 'config.json')
    tracker.artifact(data_root / 'manifest.json')
    environment = subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    (output / 'requirements-freeze.txt').write_text(environment)
    tracker.artifact(output / 'requirements-freeze.txt')
    step, epoch, skip = state['step'], state['epoch'], state['next_batch']
    start = time.perf_counter()
    accumulated_loss, microbatches = 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    try:
        while step < config['max_train_steps']:
            train_data.epoch = epoch
            order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(config['seed'] + epoch)).tolist()
            loader = accelerator.prepare(DataLoader(train_data, batch_size=config['batch_size'], sampler=order,
                num_workers=0, generator=torch.Generator().manual_seed(config['seed'] + epoch)))
            batches = accelerator.skip_first_batches(loader, skip) if skip else loader
            for index, batch in enumerate(batches, start=skip):
                with accelerator.accumulate(controlnet):
                    with torch.no_grad():
                        latents = vae.encode(batch['photo'].to(accelerator.device, dtype=dtype)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                    hidden = embeddings.expand(latents.shape[0], -1, -1)
                    with accelerator.autocast():
                        predicted = predict_noise(controlnet, unet, noisy, timesteps, hidden,
                            batch['condition'].to(accelerator.device, dtype=dtype))
                        target = noise if noise_scheduler.config.prediction_type == 'epsilon' else noise_scheduler.get_velocity(latents, noise, timesteps)
                        loss = opening_weighted_loss(predicted, target, batch['openings'].to(latents.device), config['opening_loss_weight'])
                    accelerator.backward(loss)
                    accumulated_loss += loss.item(); microbatches += 1
                    if accelerator.sync_gradients: accelerator.clip_grad_norm_(controlnet.parameters(), config['max_grad_norm'])
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients: continue
                mean_loss = accumulated_loss / microbatches
                accumulated_loss, microbatches = 0.0, 0
                if accelerator.optimizer_step_was_skipped: continue
                step += 1
                tracker.metrics({'train_loss': mean_loss, 'learning_rate': scheduler.get_last_lr()[0],
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'elapsed_seconds': time.perf_counter() - start}, step)
                print(json.dumps({'step': step, 'loss': loss.item(), 'elapsed_seconds': round(time.perf_counter() - start, 1)}), flush=True)
                model = accelerator.unwrap_model(controlnet)
                should_validate = not args.smoke and (step % config['validation_steps'] == 0 or step == config['max_train_steps'])
                if should_validate:
                    encoder = CLIPTextModel.from_pretrained(config['base_model'], subfolder='text_encoder', **common).to(accelerator.device)
                    metrics = validation(model, unet, vae, encoder, tokenizer, noise_scheduler, val_data, config, output, step, tracker, accelerator.device)
                    del encoder; torch.cuda.empty_cache()
                    if metrics['val_lpips'] < state['best_lpips']:
                        state.update(best_lpips=metrics['val_lpips'], best_step=step)
                        model.save_pretrained(output / 'best-controlnet', safe_serialization=True)
                if not args.smoke and (step % config['checkpoint_steps'] == 0 or step == config['max_train_steps']):
                    directory = output / 'checkpoints' / f'step_{step:06d}'
                    accelerator.save_state(str(directory))
                    write_json(directory / 'training-state.json', {**state, 'step': step, 'epoch': epoch,
                        'next_batch': index + 1, 'contract': contract, 'run_id': tracker.run_id})
                    directories = sorted((output / 'checkpoints').glob('step_*'))
                    for obsolete in directories[:-config['keep_checkpoints']]:
                        if not obsolete.resolve().is_relative_to((output / 'checkpoints').resolve()): raise ValueError('Invalid checkpoint path')
                        shutil.rmtree(obsolete)
                if step >= config['max_train_steps']: break
            epoch += 1; skip = 0
        if not args.smoke:
            # Export FP16 CPU weights; the frozen base is named/pinned, not duplicated.
            exported = ControlNetModel.from_pretrained(output / 'best-controlnet', torch_dtype=torch.float16)
            exported.save_pretrained(output / 'export' / 'controlnet', safe_serialization=True)
            write_json(output / 'export' / 'model.json', {'schema_version': 1, 'architecture': 'sd15_controlnet_cmp_edges_v1',
                'config': config, 'git_commit': git_revision(), 'run_id': tracker.run_id,
                'manifest_sha256': manifest_hash, 'selected_step': state['best_step'],
                'selection': 'lowest LPIPS on fixed small validation subset; not held-out test performance',
                'quality_status': 'experimental; local profiling and broader evaluation required',
                'weights_sha256': digest(output / 'export/controlnet/diffusion_pytorch_model.safetensors')})
            tracker.artifact(output / 'export' / 'model.json', 'model')
        tracker.finish()
    except BaseException:
        tracker.finish('FAILED')
        raise
    finally:
        accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output', default='/kaggle/working/facade-controlnet')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if args.check_data: print(json.dumps(validate_dataset(args.data)['splits'])); return
    train(args)


if __name__ == '__main__': main()
