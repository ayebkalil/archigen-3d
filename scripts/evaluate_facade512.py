"""Paired held-out evaluation, raw texture comparisons, and real viewer fixture."""
import argparse
import base64
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
from torchvision.models import inception_v3, Inception_V3_Weights

from src.models.facade_experiment import ExperimentDataset, load_generator, MemoryBudget
from src.models.architecture import UNetGenerator
from src.models.losses import OpeningSharpnessLoss
from src.utils.metrics import compute_ssim, compute_psnr
from src.bridge.facade_bridge import FacadeBridge, PyTorchFacadeInference
from src.bridge.facade_experimental import ExperimentalFacadeInference


def pil(tensor):
    return Image.fromarray(((tensor.detach().float().cpu().clamp(-1, 1)[0].permute(1, 2, 0).numpy()+1)*127.5).astype(np.uint8))


def grid(images, labels, dest):
    canvas = Image.new('RGB', (512*len(images), 548), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image.resize((512, 512), Image.Resampling.BICUBIC), (i*512, 36))
        draw.text((i*512+12, 10), label, fill='black')
    canvas.save(dest)


def evaluate(args):
    torch.set_num_threads(4)
    budget = MemoryBudget()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, metadata = load_generator(args.checkpoint, 'cuda')
    production = UNetGenerator()
    production.load_state_dict(torch.load(args.production_baseline, map_location='cpu', weights_only=True))
    production = production.cuda().eval()
    metric = lpips.LPIPS(net='squeeze', verbose=False).cuda().eval()
    sharpness = OpeningSharpnessLoss().cuda()
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).cuda().eval()
    inception.fc = torch.nn.Identity()
    features = {'real': [], 'baseline': [], 'coarse_path': [], 'experimental': []}
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    def extract(tensor):
        normalized = (F.interpolate((tensor.float()+1)*.5, (299,299), mode='bilinear', align_corners=False)-mean)/std
        return inception(normalized).cpu().numpy()
    data = ExperimentDataset(args.manifest, args.split)
    records = []
    for index, batch in enumerate(DataLoader(data, batch_size=1)):
        x, y, s = batch['sketch'].cuda(), batch['photo'].cuda(), batch['style'].cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            low = F.interpolate(x, (256, 256), mode='nearest')
            baseline = F.interpolate(production(low), (512, 512), mode='bilinear', align_corners=False)
            coarse_prior = F.interpolate(model.coarse(low), (512, 512), mode='bilinear', align_corners=False)
            enhanced = model(x, s)
        with torch.inference_mode():
            measures = {}
            features['real'].append(extract(y))
            for name, prediction in [('baseline', baseline.float()), ('coarse_path', coarse_prior.float()), ('experimental', enhanced.float())]:
                features[name].append(extract(prediction))
                measures[name] = {'lpips': metric(prediction, y).mean().item(),
                                  'ssim': compute_ssim(prediction, y), 'psnr': compute_psnr(prediction, y),
                                  'opening_edge_mae': sharpness(prediction, y, x).item()}
        records.append({'id': batch['id'][0], **measures})
        if index < 8:
            grid([pil(x), pil(baseline), pil(enhanced), pil(y)],
                 ['Semantic input', 'Baseline 256 (displayed at 512)', 'Experimental 512', 'Original CMP target'], out/f'{args.split}_{index:03}.png')
        if index % 20 == 0: print(f'{args.split}: {index+1}/{len(data)}', flush=True)
    means = {model_name: {key: float(np.mean([r[model_name][key] for r in records]))
                         for key in records[0][model_name]} for model_name in ('baseline', 'coarse_path', 'experimental')}
    # Unbiased full-set polynomial MMD on torchvision Inception pool3 features.
    # Report extractor explicitly; this is not interchangeable with TF-FID KID.
    real_features = np.concatenate(features['real']).astype(np.float64)
    def kid(fake_features):
        fake_features = np.concatenate(fake_features).astype(np.float64)
        xx = (real_features @ real_features.T / 2048 + 1)**3
        yy = (fake_features @ fake_features.T / 2048 + 1)**3
        xy = (real_features @ fake_features.T / 2048 + 1)**3
        n = len(real_features)
        return float(((xx.sum()-np.trace(xx))+(yy.sum()-np.trace(yy)))/(n*(n-1))-2*xy.mean())
    for name in means:
        means[name]['kid_torchvision_x1000'] = kid(features[name])*1000
    differences = np.array([r['experimental']['lpips']-r['baseline']['lpips'] for r in records])
    rng = np.random.default_rng(42)
    boot = rng.choice(differences, (2000, len(differences)), replace=True).mean(1)
    summary = {'split': args.split, 'n': len(records), 'checkpoint_epoch': metadata['epoch'],
               'production_baseline': args.production_baseline,
               'production_baseline_sha256': hashlib.sha256(Path(args.production_baseline).read_bytes()).hexdigest(),
               'coarse_prior_sha256': metadata['baseline_sha256'],
               'comparison': 'Both models use identical original CMP semantic input; baseline output bilinear-upsampled to 512',
               'metrics': means, 'lpips_delta_95ci': np.quantile(boot, [.025, .975]).tolist(),
               'kid_extractor': 'torchvision Inception_V3_Weights.DEFAULT, transform_input=False, pool3, ImageNet normalization; full-set unbiased MMD',
               'supported_style_ids': metadata['supported_style_ids'], **budget.check()}
    (out/f'{args.split}_metrics.json').write_text(json.dumps({'summary': summary, 'per_image': records}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # Same physical wall and openings as Phase 1 browser verification.
    wall = {'id': 'wall_0', 'height_m': 2.8, 'is_front': True, 'is_exterior': True,
            'polygon_meters': [[0,0],[8,0],[8,.18],[0,.18]], 'openings': [
                {'id':'door_0','type':'door','position_meters':[5.8,0],'width_m':1.1,'height_m':2.1},
                {'id':'window_0','type':'window','position_meters':[1.6,0],'width_m':1.4,'height_m':1.4},
                {'id':'window_1','type':'window','position_meters':[3.7,0],'width_m':1.2,'height_m':1.3}]}
    bridge = FacadeBridge.__new__(FacadeBridge)
    bridge.enhance_textures = False
    engine = ExperimentalFacadeInference.__new__(ExperimentalFacadeInference)
    engine.generator, engine.metadata, engine.device, engine.session = model, metadata, 'cuda', None
    engine.supported_styles = metadata['supported_style_ids']
    bridge.inference_engine = engine
    result = bridge.render_wall_facade(wall, num_floors=2, is_front=True)
    result['photo_image'].save(out/'facade_512.png'); result['sketch_image'].save(out/'facade_sketch_512.png')
    fixture = {k: v for k, v in result.items() if k not in ('photo_image','sketch_image','photo_bytes','sketch_bytes')}
    fixture.update({'texture_base64':'data:image/png;base64,'+base64.b64encode(result['photo_bytes']).decode(),
                    'sketch_base64':'data:image/png;base64,'+base64.b64encode(result['sketch_bytes']).decode(),
                    'num_floors':2, 'device':'Experimental 512 PyTorch CUDA'})
    (out/'viewer_fixture.json').write_text(json.dumps(fixture))
    bridge.inference_engine = PyTorchFacadeInference(production, 'cuda', enhance=True)
    legacy = bridge.render_wall_facade(wall, num_floors=2, is_front=True)
    legacy['photo_image'].save(out/'baseline_256.png')
    grid([legacy['photo_image'], result['photo_image']], ['Production baseline 256', 'Experimental 512 - CMP neutral'], out/'texture_comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', default='models/saved/facade512_experiment/generator.pth')
    parser.add_argument('--manifest', default='data/processed/cmp512/manifest.json')
    parser.add_argument('--production-baseline', default='models/saved/generator_perceptual_ema_best.pth')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', default='outputs/phase2')
    evaluate(parser.parse_args())
