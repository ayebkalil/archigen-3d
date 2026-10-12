"""
Kernel Inception Distance (KID) Computation Suite for CMP Facades Test Set.
KID is an unbiased estimator of Maximum Mean Discrepancy (MMD) with a polynomial kernel:
  k(x, y) = (1/d * x^T y + 1)^3
Unlike FID, KID has an unbiased cubic estimator that remains statistically reliable
on small sample regimes (N = 106 images).
Reference: Binkowski et al., "Demystifying MMD GANs", ICLR 2018.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.models.architecture import UNetGenerator
from src.utils.seed import seed_everything

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.fc = nn.Identity()  # Return 2048-dim pool3 embeddings
        inception.eval()
        self.inception = inception
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x_pre = self.preprocess(x)
        with torch.no_grad():
            feat = self.inception(x_pre)
        return feat

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * np.dot(X, Y.T) + coef0) ** degree

def compute_unbiased_mmd2(K_XX, K_YY, K_XY):
    m = K_XX.shape[0]
    n = K_YY.shape[0]
    # Zero diagonals for unbiased estimate
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd2 = (np.sum(K_XX) / (m * (m - 1)) +
            np.sum(K_YY) / (n * (n - 1)) -
            2 * np.sum(K_XY) / (m * n))
    return float(mmd2)

def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray, num_subsets: int = 100, subset_size: int = 50):
    n_real = len(real_feats)
    n_fake = len(fake_feats)
    actual_sub = min(subset_size, n_real, n_fake)

    mmd2_list = []
    for _ in range(num_subsets):
        idx_r = np.random.choice(n_real, actual_sub, replace=False)
        idx_f = np.random.choice(n_fake, actual_sub, replace=False)
        X = real_feats[idx_r]
        Y = fake_feats[idx_f]

        K_XX = polynomial_kernel(X, X)
        K_YY = polynomial_kernel(Y, Y)
        K_XY = polynomial_kernel(X, Y)

        mmd2_list.append(compute_unbiased_mmd2(K_XX, K_YY, K_XY))

    kid_mean = float(np.mean(mmd2_list))
    kid_std = float(np.std(mmd2_list))
    return kid_mean, kid_std

def evaluate_facade_kid():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[KID EVALUATION] Extracting 2048-dim Inception-V3 embeddings on device: {device}")

    extractor = InceptionFeatureExtractor().to(device)

    test_dir = Path("data/raw/facades/test")
    test_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    print(f"[DATA] Found {len(test_files)} CMP test image pairs.")

    real_imgs, sketch_imgs = [], []
    for f in test_files:
        img = Image.open(f).convert("RGB")
        w, h = img.size
        half_w = w // 2
        real = img.crop((0, 0, half_w, h)).resize((256, 256), Image.Resampling.BICUBIC)
        sketch = img.crop((half_w, 0, w, h)).resize((256, 256), Image.Resampling.BICUBIC)
        real_imgs.append(transforms.ToTensor()(real))
        sketch_imgs.append(transforms.ToTensor()(sketch))

    real_tensor = torch.stack(real_imgs)
    sketch_tensor = torch.stack(sketch_imgs)

    # 1. Extract Real Features
    print("  Extracting real image features...")
    real_feats = []
    for i in range(0, len(real_tensor), 16):
        batch = real_tensor[i:i+16].to(device)
        feats = extractor(batch).cpu().numpy()
        real_feats.append(feats)
    real_feats = np.concatenate(real_feats, axis=0)

    # Models to evaluate
    models_to_test = [
        ("Baseline (100ep, TransposeConv)", "models/saved/generator_best.pth", "transpose"),
        ("Enhanced (150ep, Nearest+Conv)", "models/saved/generator_enhanced_best.pth", "nearest"),
        ("Perceptual (100ep, LPIPS+SN)", "models/saved/generator_perceptual_best.pth", "nearest")
    ]

    results = {}
    for name, path_str, up_mode in models_to_test:
        p = Path(path_str)
        if not p.exists():
            print(f"  [SKIP] Weights not found: {path_str}")
            continue

        print(f"  Evaluating: {name}...")
        gen = UNetGenerator(in_channels=3, out_channels=3, num_filters=64, upsample_mode=up_mode).to(device)
        ckpt = torch.load(p, map_location=device)
        state_dict = ckpt["generator_state_dict"] if "generator_state_dict" in ckpt else ckpt
        gen.load_state_dict(state_dict)
        gen.eval()

        fake_feats = []
        with torch.no_grad():
            for i in range(0, len(sketch_tensor), 16):
                batch_sketch = sketch_tensor[i:i+16].to(device)
                batch_sketch_norm = batch_sketch * 2.0 - 1.0
                fake_norm = gen(batch_sketch_norm)
                fake_01 = (fake_norm + 1.0) / 2.0
                feats = extractor(fake_01).cpu().numpy()
                fake_feats.append(feats)

        fake_feats = np.concatenate(fake_feats, axis=0)
        kid_m, kid_s = compute_kid(real_feats, fake_feats)
        results[name] = {"kid_mean": kid_m, "kid_std": kid_s}
        print(f"    -> KID (x1000): {kid_m * 1000:.3f} +/- {kid_s * 1000:.3f}")

    print("\n" + "=" * 70)
    print("      CMP FACADES TEST SET (106 IMAGES) - KID RESULTS")
    print("=" * 70)
    for name, res in results.items():
        print(f"{name:<35} | KID: {res['kid_mean']*1000:7.3f} +/- {res['kid_std']*1000:6.3f}")
    print("=" * 70 + "\n")

    return results

if __name__ == "__main__":
    evaluate_facade_kid()
