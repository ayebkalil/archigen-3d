"""
Authoritative Facade Evaluation Suite for ArchiGen 3D.
Uses FacadesDataset(split='test') directly to guarantee consistent cropping.
Evaluates all 3 generator checkpoints on the CMP Facades test set (N=106 images):
1. Baseline pix2pix (TransposeConv, 100 epochs)
2. Enhanced pix2pix (Nearest+Conv, 150 epochs, SSIM/L1 focus)
3. Perceptual pix2pix (SpectralNorm + LPIPS, 100 epochs)

Computes:
- L1 Loss
- SSIM
- PSNR
- LPIPS (SqueezeNet)
- KID (x10^3) with unbiased cubic estimator (100 subsets of size 50)
- Saves output to models/saved/facade_authoritative_metrics.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FacadesDataset
from src.models.architecture import UNetGenerator
from src.utils.metrics import compute_ssim, compute_psnr, compute_lpips
from src.utils.seed import seed_everything

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.eval()
        self.inception = inception
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x_01):
        x_299 = self.preprocess(x_01)
        x_299 = self.inception.Conv2d_1a_3x3(x_299)
        x_299 = self.inception.Conv2d_2a_3x3(x_299)
        x_299 = self.inception.Conv2d_2b_3x3(x_299)
        x_299 = self.inception.maxpool1(x_299)
        x_299 = self.inception.Conv2d_3b_1x1(x_299)
        x_299 = self.inception.Conv2d_4a_3x3(x_299)
        x_299 = self.inception.maxpool2(x_299)
        x_299 = self.inception.Mixed_5b(x_299)
        x_299 = self.inception.Mixed_5c(x_299)
        x_299 = self.inception.Mixed_5d(x_299)
        x_299 = self.inception.Mixed_6a(x_299)
        x_299 = self.inception.Mixed_6b(x_299)
        x_299 = self.inception.Mixed_6c(x_299)
        x_299 = self.inception.Mixed_6d(x_299)
        x_299 = self.inception.Mixed_6e(x_299)
        x_299 = self.inception.Mixed_7a(x_299)
        x_299 = self.inception.Mixed_7b(x_299)
        x_299 = self.inception.Mixed_7c(x_299)
        x_299 = self.inception.avgpool(x_299)
        x_299 = torch.flatten(x_299, 1)
        return x_299

def compute_kid(real_feats, fake_feats, num_subsets=100, subset_size=50):
    n_r = len(real_feats)
    n_f = len(fake_feats)
    sub_size = min(subset_size, n_r, n_f)
    mmd2_list = []

    def poly_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        return (gamma * np.dot(X, Y.T) + coef0) ** degree

    for _ in range(num_subsets):
        r_sub = real_feats[np.random.choice(n_r, sub_size, replace=False)]
        f_sub = fake_feats[np.random.choice(n_f, sub_size, replace=False)]

        K_rr = poly_kernel(r_sub, r_sub)
        K_ff = poly_kernel(f_sub, f_sub)
        K_rf = poly_kernel(r_sub, f_sub)

        m = sub_size
        term_rr = (K_rr.sum() - np.trace(K_rr)) / (m * (m - 1))
        term_ff = (K_ff.sum() - np.trace(K_ff)) / (m * (m - 1))
        term_rf = 2.0 * K_rf.sum() / (m * m)

        mmd2 = term_rr + term_ff - term_rf
        mmd2_list.append(mmd2)

    return float(np.mean(mmd2_list)), float(np.std(mmd2_list))

def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Authoritative Facade Benchmark on: {device}")

    dataset = FacadesDataset(root_dir="data/raw/facades", split="test", img_size=256, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"[*] Loaded {len(dataset)} test images from FacadesDataset.")

    models_info = [
        ("Baseline (100ep)", "models/saved/generator_best.pth", "transpose"),
        ("Enhanced (150ep)", "models/saved/generator_enhanced_best.pth", "nearest"),
        ("Perceptual (100ep)", "models/saved/generator_perceptual_best.pth", "nearest")
    ]

    inception_extractor = InceptionFeatureExtractor().to(device)

    # 1. Extract real features
    print("[*] Extracting real photo Inception features...")
    real_feats_list = []
    with torch.no_grad():
        for batch in loader:
            # photo in [-1, 1]
            photo = batch["photo"].to(device)
            photo_01 = ((photo + 1.0) / 2.0).clamp(0, 1)
            feats = inception_extractor(photo_01).cpu().numpy()
            real_feats_list.append(feats)
    real_feats = np.concatenate(real_feats_list, axis=0)

    results = {}
    criterion_l1 = nn.L1Loss()

    for label, ckpt_p, up_mode in models_info:
        p = Path(ckpt_p)
        if not p.exists():
            print(f"[!] Checkpoint not found: {ckpt_p}")
            continue

        print(f"\n[*] Evaluating: {label} ({ckpt_p})...")
        gen = UNetGenerator(in_channels=3, out_channels=3, num_filters=64, upsample_mode=up_mode).to(device)
        ckpt = torch.load(p, map_location=device)
        state_dict = ckpt["generator_state_dict"] if "generator_state_dict" in ckpt else ckpt
        gen.load_state_dict(state_dict)
        gen.eval()

        l1_sum = 0.0
        ssim_sum = 0.0
        psnr_sum = 0.0
        lpips_sum = 0.0
        n_batches = 0
        fake_feats_list = []

        with torch.no_grad():
            for batch in loader:
                sketch = batch["sketch"].to(device) # [-1, 1]
                real_photo = batch["photo"].to(device) # [-1, 1]

                fake_photo = gen(sketch) # [-1, 1]

                l1_sum += criterion_l1(fake_photo, real_photo).item()
                ssim_sum += compute_ssim(fake_photo, real_photo)
                psnr_sum += compute_psnr(fake_photo, real_photo)
                lpips_sum += compute_lpips(fake_photo, real_photo, net="squeeze")

                fake_01 = ((fake_photo + 1.0) / 2.0).clamp(0, 1)
                feats = inception_extractor(fake_01).cpu().numpy()
                fake_feats_list.append(feats)
                n_batches += 1

        fake_feats = np.concatenate(fake_feats_list, axis=0)
        kid_m, kid_s = compute_kid(real_feats, fake_feats)

        mean_l1 = l1_sum / n_batches
        mean_ssim = ssim_sum / n_batches
        mean_psnr = psnr_sum / n_batches
        mean_lpips = lpips_sum / n_batches

        results[label] = {
            "checkpoint": ckpt_p,
            "upsample_mode": up_mode,
            "l1_loss": round(mean_l1, 4),
            "ssim": round(mean_ssim, 4),
            "psnr_db": round(mean_psnr, 2),
            "lpips": round(mean_lpips, 4),
            "kid_x1000": round(kid_m * 1000, 3),
            "kid_std_x1000": round(kid_s * 1000, 3)
        }

    out_file = Path("models/saved/facade_authoritative_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print("AUTHORITATIVE FACADE EVALUATION SUMMARY (N=106 CMP Facades Test Images)")
    print("=" * 85)
    print(f"{'Model':<22} | {'L1 Loss':<8} | {'SSIM':<8} | {'PSNR (dB)':<9} | {'LPIPS':<8} | {'KID (x10^3)':<16}")
    print("-" * 85)
    for label, d in results.items():
        print(f"{label:<22} | {d['l1_loss']:<8.4f} | {d['ssim']:<8.4f} | {d['psnr_db']:<9.2f} | {d['lpips']:<8.4f} | {d['kid_x1000']:6.3f} ± {d['kid_std_x1000']:5.3f}")
    print("=" * 85)
    print(f"Authoritative metrics saved to: {out_file}\n")

if __name__ == "__main__":
    main()
