"""Isolated, resumable 512 facade experiment. Never writes production checkpoints.

python -m src.models.facade_experiment --manifest data/processed/cmp512/manifest.json
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

from src.models.architecture import FacadeGenerator512, MultiScaleFacadeDiscriminator, NEUTRAL_STYLE
from src.models.losses import VGG19PerceptualLoss, OpeningSharpnessLoss, multiscale_feature_matching


class ExperimentDataset(Dataset):
    def __init__(self, manifest, split, augment=False):
        document = json.loads(Path(manifest).read_text())
        self.rows = [r for r in document['samples'] if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} rows in {manifest}')
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['photo']) as im:
            photo = im.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        with Image.open(row['sketch']) as im:
            sketch = im.convert('RGB').resize((512, 512), Image.Resampling.NEAREST)
        if self.augment and random.random() < .5:
            photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            sketch = sketch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {'photo': to_tensor(photo) * 2 - 1, 'sketch': to_tensor(sketch) * 2 - 1,
                'style': int(row.get('style_id', NEUTRAL_STYLE)), 'id': row['id']}


class MemoryBudget:
    """Conservative allocator cap + measured total-device guard; fails closed."""
    def __init__(self, limit_mib=3500):
        if not torch.cuda.is_available():
            raise RuntimeError('This training experiment requires CUDA.')
        self.limit = limit_mib
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        # Leave room for context, kernels and other driver allocations.
        torch.cuda.set_per_process_memory_fraction(min(.80, (limit_mib - 450) / total))
        torch.cuda.reset_peak_memory_stats()
        self.peak_device_mib = 0

    def check(self):
        torch.cuda.synchronize()
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        used = int(raw.strip().splitlines()[0])
        self.peak_device_mib = max(self.peak_device_mib, used)
        reserved = torch.cuda.max_memory_reserved() / 2**20
        if used > self.limit or reserved > self.limit:
            raise RuntimeError(f'GPU budget exceeded: device={used}MiB, peak reserved={reserved:.0f}MiB')
        return {'device_mib': used, 'peak_device_sampled_mib': self.peak_device_mib,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': reserved}


def load_generator(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('architecture') != 'facade_residual_film_512_v1':
        raise ValueError('Not a versioned 512 facade checkpoint')
    model = FacadeGenerator512(**state['model_config'])
    model.load_state_dict(state['generator'])
    return model.to(device).eval(), state


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    budget = MemoryBudget(args.memory_mib)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = ExperimentDataset(args.manifest, 'train', augment=True)
    val = ExperimentDataset(args.manifest, 'val')
    loader = DataLoader(data, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    config = {'width': args.width, 'blocks': args.blocks, 'gradient_checkpointing': True}
    generator = FacadeGenerator512(**config)
    generator.coarse.load_state_dict(torch.load(args.baseline, map_location='cpu', weights_only=True))
    generator = generator.cuda()
    discriminator = MultiScaleFacadeDiscriminator(width=16).cuda()
    vgg = VGG19PerceptualLoss().cuda()
    edge = OpeningSharpnessLoss().cuda()
    opt_g = torch.optim.Adam([p for p in generator.parameters() if p.requires_grad], lr=args.lr, betas=(.5, .999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.5, .999))
    scaler = torch.amp.GradScaler('cuda')
    start_epoch, best, step = 0, float('inf'), 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state['model_config'] != config:
            raise ValueError('Resume configuration differs')
        generator.load_state_dict(state['generator']); discriminator.load_state_dict(state['discriminator'])
        opt_g.load_state_dict(state['optimizer_g']); opt_d.load_state_dict(state['optimizer_d'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best, step = state['epoch'], state['best_val'], state['step']
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    start = time.monotonic()
    supported = sorted({int(r.get('style_id', NEUTRAL_STYLE)) for r in data.rows} - {NEUTRAL_STYLE})
    metadata = {'architecture': 'facade_residual_film_512_v1', 'resolution': 512, 'model_config': config,
                'supported_style_ids': supported, 'neutral_style_id': NEUTRAL_STYLE,
                'baseline_sha256': hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
                'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'loss_weights': {'l1': 5., 'vgg': 1., 'feature_matching': 5., 'opening_edges': 2., 'gan': 1.},
                'physical_batch': 2, 'accumulation': 4, 'seed': args.seed}
    (out/'config.json').write_text(json.dumps({**metadata, 'arguments': vars(args)}, indent=2))
    for epoch in range(start_epoch, args.epochs):
        generator.train(); discriminator.train()
        opt_g.zero_grad(set_to_none=True); opt_d.zero_grad(set_to_none=True)
        count = min(len(loader), args.max_batches) if args.max_batches else len(loader)
        totals = []
        for i, batch in enumerate(loader):
            if i >= count: break
            sketch, photo, style = batch['sketch'].cuda(), batch['photo'].cuda(), batch['style'].cuda()
            # Average the last partial accumulation group correctly.
            divisor = min(4, count - (i // 4) * 4)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                fake = generator(sketch, style)
            discriminator.requires_grad_(True).train()
            with torch.autocast('cuda', dtype=torch.float16):
                real_d = discriminator(sketch, photo, style)
                fake_d = discriminator(sketch, fake, style)
                loss_d = sum((F.relu(1 - r[0]).mean() + F.relu(1 + f[0]).mean()) * .5
                             for r, f in zip(real_d, fake_d)) / 2
            scaler.scale(loss_d / divisor).backward()
            del real_d, fake_d, fake
            discriminator.requires_grad_(False).eval()
            with torch.autocast('cuda', dtype=torch.float16):
                fake = generator(sketch, style)
                fake_d = discriminator(sketch, fake, style)
                with torch.no_grad(): real_d = discriminator(sketch, photo, style)
                gan = -sum(r[0].mean() for r in fake_d) / 2
                fm = multiscale_feature_matching(real_d, fake_d)
                # Sequential VGG samples reduce peak memory without changing physical batch.
                perceptual = sum(vgg(fake[j:j+1], photo[j:j+1]) for j in range(len(fake))) / len(fake)
                sharpness = edge(fake, photo, sketch)
                l1 = F.l1_loss(fake.float(), photo.float())
                loss_g = gan + 5 * fm + perceptual + 2 * sharpness + 5 * l1
            if not torch.isfinite(loss_g + loss_d): raise FloatingPointError('Nonfinite GAN loss')
            scaler.scale(loss_g / divisor).backward()
            if (i + 1) % 4 == 0 or i + 1 == count:
                for optimizer in (opt_g, opt_d):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 10.)
                    scaler.step(optimizer)
                scaler.update()
                opt_g.zero_grad(set_to_none=True); opt_d.zero_grad(set_to_none=True)
                step += 1
            totals.append([loss_g.item(), loss_d.item(), l1.item(), perceptual.item(), sharpness.item()])
            del fake, fake_d, real_d, loss_g, loss_d, perceptual, fm, gan, sharpness, l1
            if i % 10 == 0:
                print(json.dumps({'epoch': epoch+1, 'batch': i+1, 'steps': step, **budget.check()}), flush=True)
        generator.eval()
        values = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for batch in DataLoader(val, batch_size=1):
                x, y, s = batch['sketch'].cuda(), batch['photo'].cuda(), batch['style'].cuda()
                pred = generator(x, s)
                values.append(vgg(pred, y).item())
        metric = float(np.mean(values))
        improved = metric < best
        best = min(best, metric)
        record = {'epoch': epoch+1, 'steps': step, 'train_losses_g_d_l1_vgg_edges': np.mean(totals, 0).tolist(),
                  'val_vgg': metric, 'best_val_vgg': best, 'elapsed_s': time.monotonic()-start, **budget.check()}
        print(json.dumps(record), flush=True)
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        state = {**metadata, 'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                 'optimizer_g': opt_g.state_dict(), 'optimizer_d': opt_d.state_dict(), 'scaler': scaler.state_dict(),
                 'epoch': epoch+1, 'step': step, 'best_val': best, 'python_rng': random.getstate(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        temporary = out/'last.tmp'
        torch.save(state, temporary); temporary.replace(out/'last.pth')
        if improved:
            torch.save({**metadata, 'generator': generator.state_dict(), 'epoch': epoch+1, 'val_vgg': metric}, out/'best.pth')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', default='data/processed/cmp512/manifest.json')
    parser.add_argument('--baseline', default='models/saved/generator_perceptual_best.pth')
    parser.add_argument('--output', default='outputs/phase2/training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--memory-mib', type=int, default=3500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume')
    train(parser.parse_args())
