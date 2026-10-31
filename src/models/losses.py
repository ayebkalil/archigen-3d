"""
Advanced Loss Functions for ArchiGen 3D Facade Generation.
Implements:
1. Feature Matching Loss (Pix2PixHD style from Discriminator intermediate layers)
2. Perceptual LPIPS Loss (using pretrained deep features to restore sharp edges and textures)
"""

import torch
import torch.nn as nn
import lpips
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class FeatureMatchingLoss(nn.Module):
    """
    Discriminator Feature Matching Loss (from Pix2PixHD, Wang et al. 2018).
    Directly matches multi-scale intermediate layer representations between real and synthesized facades.
    Forces the generator to produce the same high-frequency edge and corner activations
    as ground-truth images across multiple receptive field scales (128px, 64px, 32px, 31px).
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, real_features: list, fake_features: list) -> torch.Tensor:
        loss = 0.0
        for r_feat, f_feat in zip(real_features, fake_features):
            loss += self.criterion(f_feat, r_feat.detach())
        return loss / max(len(real_features), 1)


class PerceptualLoss(nn.Module):
    """
    Deep Perceptual Loss based on LPIPS (Learned Perceptual Image Patch Similarity).
    Penalizes blurred averages and rewards sharp high-frequency architectural textures
    (window mullions, sills, brick texture, balconies).
    Expects inputs normalized in [-1, 1].
    """
    def __init__(self, net: str = "squeeze"):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net=net, verbose=False)
        # Freeze weights
        for param in self.lpips_model.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips_model(pred, target).mean()


class VGG19PerceptualLoss(nn.Module):
    """ImageNet VGG-19 relu{1,2,3,4,5}_2; target detached, prediction differentiable."""
    indices = (3, 8, 13, 22, 31)

    def __init__(self, pretrained=True, checkpoint_features=True):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        from torchvision.models.vgg import make_layers, cfgs
        from pathlib import Path
        feature_cache = Path(torch.hub.get_dir()) / 'checkpoints/vgg19_features.pth'
        if pretrained and feature_cache.is_file():
            features = make_layers(cfgs['E'])
            features.load_state_dict(torch.load(feature_cache, map_location='cpu', weights_only=True))
        elif pretrained:
            features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        else:
            features = make_layers(cfgs['E'])
        starts = (0, 4, 9, 14, 23)
        self.slices = nn.ModuleList([features[start:end + 1] for start, end in zip(starts, self.indices)])
        # In-place ReLU is unsafe at checkpoint boundaries.
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.requires_grad_(False).eval()
        self.checkpoint_features = checkpoint_features
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = ((pred + 1) * .5 - self.mean) / self.std
        target = ((target.detach() + 1) * .5 - self.mean) / self.std
        loss = pred.new_zeros((), dtype=torch.float32)
        for block, weight in zip(self.slices, (1/32, 1/16, 1/8, 1/4, 1.0)):
            pred = checkpoint(block, pred, use_reentrant=False) if pred.requires_grad and self.checkpoint_features else block(pred)
            with torch.no_grad():
                target = block(target)
            loss = loss + weight * F.l1_loss(pred.float(), target.float())
        return loss


class OpeningSharpnessLoss(nn.Module):
    """Photograph edge matching near semantic doors/windows, not invented mask edges."""
    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ], dtype=torch.float32)[:, None] / 4)

    def forward(self, pred, target, sketch):
        rgb = (sketch.detach().float() + 1) * 127.5
        window = rgb.new_tensor([0, 128, 255])[None, :, None, None]
        door = rgb.new_tensor([0, 207, 255])[None, :, None, None]
        mask = (((rgb - window).abs().amax(1, keepdim=True) < 24) |
                ((rgb - door).abs().amax(1, keepdim=True) < 24)).float()
        mask = F.max_pool2d(mask, 9, stride=1, padding=4)
        # Luminance edge residual normalized by actual masked area; empty masks give zero.
        lum = pred.new_tensor([.299, .587, .114], dtype=torch.float32)[None, :, None, None]
        p = F.conv2d((pred.float() * lum).sum(1, keepdim=True), self.kernels.float(), padding=1)
        t = F.conv2d((target.detach().float() * lum).sum(1, keepdim=True), self.kernels.float(), padding=1)
        return ((p - t).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1)


def multiscale_feature_matching(real, fake):
    return torch.stack([F.l1_loss(f.float(), r.detach().float())
                        for (_, rfs), (_, ffs) in zip(real, fake)
                        for r, f in zip(rfs, ffs)]).mean()
