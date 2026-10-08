"""
Advanced Loss Functions for ArchiGen 3D Facade Generation.
Implements:
1. Feature Matching Loss (Pix2PixHD style from Discriminator intermediate layers)
2. Perceptual LPIPS Loss (using pretrained deep features to restore sharp edges and textures)
"""

import torch
import torch.nn as nn
import lpips

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
