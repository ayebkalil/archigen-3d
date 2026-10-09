"""
Fréchet Inception Distance (FID) and Image Quality Evaluation Suite.
Evaluates:
1. Baseline pix2pix (100ep, L1 weight 100, TransposeConv)
2. Enhanced pix2pix (150ep, L1 weight 100, Upsample+Conv, Augment)
3. Perceptual pix2pix (100ep, VGG Perceptual + L1 weight 10, Spectral Norm)
Against Ground-Truth Real Facades on the CMP Test Split (106 images).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.linalg
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from src.models.architecture import UNetGenerator
from src.utils.metrics import compute_ssim, compute_psnr
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

    def forward(self, x):
        # x is in [0, 1] range of shape (B, 3, H, W)
        x_pre = self.preprocess(x)
        with torch.no_grad():
            feat = self.inception(x_pre)
            # Inception returns 1000 classes or logits; pool3 feature can be tapped or logits used for distribution
        return feat

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_all_metrics():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FID BENCHMARK] Device: {device}")

    test_dir = Path("data/raw/facades/test")
    img_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    print(f"[DATA] Found {len(img_files)} test facade pairs.")

    # Inception Model
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
    inception.eval()
    # Replace final classification head with identity to get 2048-dim pool3 embeddings
    inception.fc = nn.Identity()

    # Models to compare
    model_configs = {
        "Baseline pix2pix": {
            "path": "models/saved/generator_best.pth",
            "upsample": "transpose"
        },
        "Enhanced pix2pix": {
            "path": "models/saved/generator_enhanced_best.pth",
            "upsample": "nearest"
        },
        "Perceptual pix2pix (Active)": {
            "path": "models/saved/generator_perceptual_best.pth",
            "upsample": "nearest"
        }
    }

    loaded_models = {}
    for name, cfg in model_configs.items():
        if Path(cfg["path"]).exists():
            gen = UNetGenerator(in_channels=3, out_channels=3, upsample_mode=cfg["upsample"]).to(device)
            gen.load_state_dict(torch.load(cfg["path"], map_location=device))
            gen.eval()
            loaded_models[name] = gen
        else:
            print(f"[WARNING] Checkpoint not found: {cfg['path']}")

    incept_transform = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_features = []
    model_features = {name: [] for name in loaded_models}
    model_l1_errors = {name: [] for name in loaded_models}
    model_ssim_scores = {name: [] for name in loaded_models}

    # Normalize [-1, 1] to [0, 1]
    def to_01(t):
        return (t + 1.0) / 2.0

    print("[EVALUATION] Extracting features across test dataset...")
    for f in img_files:
        pair = Image.open(f).convert("RGB")
        w, h = pair.size
        sketch = pair.crop((0, 0, w // 2, h)).resize((256, 256), Image.Resampling.BICUBIC)
        real = pair.crop((w // 2, 0, w, h)).resize((256, 256), Image.Resampling.BICUBIC)

        t_sketch = transforms.ToTensor()(sketch).unsqueeze(0).to(device) * 2.0 - 1.0 # [-1, 1]
        t_real = transforms.ToTensor()(real).unsqueeze(0).to(device)                # [0, 1]

        with torch.no_grad():
            real_299 = incept_transform(t_real)
            f_real = inception(real_299).cpu().numpy().squeeze(0)
            real_features.append(f_real)

            for name, gen in loaded_models.items():
                pred = gen(t_sketch)
                pred_01 = to_01(pred).clamp(0, 1)

                pred_299 = incept_transform(pred_01)
                f_pred = inception(pred_299).cpu().numpy().squeeze(0)
                model_features[name].append(f_pred)

                l1 = torch.nn.functional.l1_loss(pred_01, t_real).item()
                ssim = compute_ssim(pred_01, t_real)
                model_l1_errors[name].append(l1)
                model_ssim_scores[name].append(ssim)

    # Real Statistics
    real_feats = np.array(real_features)
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)

    print("\n" + "=" * 75)
    print("      FACADE GENERATIVE BENCHMARK: CMP TEST SET (106 IMAGES)")
    print("=" * 75)
    header = f"{'Model Variant':<28} | {'FID (Lower=Better)':<18} | {'SSIM (Higher)':<13} | {'L1 Error':<10}"
    print(header)
    print("-" * 75)

    for name in loaded_models:
        m_feats = np.array(model_features[name])
        mu_m = np.mean(m_feats, axis=0)
        sigma_m = np.cov(m_feats, rowvar=False)

        fid = calculate_frechet_distance(mu_real, sigma_real, mu_m, sigma_m)
        avg_ssim = float(np.mean(model_ssim_scores[name]))
        avg_l1 = float(np.mean(model_l1_errors[name]))

        print(f"{name:<28} | {fid:<18.2f} | {avg_ssim:<13.4f} | {avg_l1:<10.4f}")

    print("=" * 75 + "\n")

if __name__ == "__main__":
    compute_all_metrics()
