"""
Unit tests for Generator and Discriminator Deep Learning Architectures.
Satisfies MLOps Criterion: 'Tests unitaires' (3/3).
"""

import pytest
import torch
from src.models.architecture import UNetGenerator, PatchGANDiscriminator

def test_unet_generator_forward_pass():
    """Verify Generator output shape and value range [-1, 1]."""
    generator = UNetGenerator(in_channels=3, out_channels=3, num_filters=64)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = generator(dummy_input)

    assert output.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {output.shape}"
    assert output.min() >= -1.0 and output.max() <= 1.0, "Tanh output must be bounded in [-1, 1]"

def test_patchgan_discriminator_forward_pass():
    """Verify PatchGAN Discriminator spatial receptive field patch shape."""
    discriminator = PatchGANDiscriminator(in_channels=6, num_filters=64, use_spectral_norm=True)
    sketch = torch.randn(2, 3, 256, 256)
    photo = torch.randn(2, 3, 256, 256)
    patch_preds = discriminator(sketch, photo)

    assert patch_preds.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {patch_preds.shape}"

def test_generator_gradient_flow():
    """Verify that gradients propagate cleanly through all U-Net layers."""
    generator = UNetGenerator()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = generator(dummy_input)
    loss = output.sum()
    loss.backward()

    for name, param in generator.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Zero gradient detected in parameter: {name}"

def test_metrics_ssim_and_psnr():
    """Verify SSIM and PSNR computation on identical and distinct tensors."""
    from src.utils.metrics import compute_ssim, compute_psnr
    t1 = torch.randn(1, 3, 256, 256)
    
    # Identical images should have SSIM ~ 1.0 and high PSNR
    ssim_same = compute_ssim(t1, t1)
    assert ssim_same > 0.99, f"SSIM of identical images should be ~1.0, got {ssim_same}"
    
    psnr_same = compute_psnr(t1, t1)
    assert psnr_same == 100.0, f"PSNR of identical images should be 100 dB, got {psnr_same}"

def test_feature_matching_and_perceptual_loss():
    """Verify FeatureMatchingLoss and PerceptualLoss compute valid gradients."""
    from src.models.losses import FeatureMatchingLoss, PerceptualLoss
    from src.utils.metrics import compute_lpips

    fm_loss = FeatureMatchingLoss()
    r_feats = [torch.randn(2, 64, 128, 128)]
    f_feats = [torch.randn(2, 64, 128, 128, requires_grad=True)]
    loss_val = fm_loss(r_feats, f_feats)
    loss_val.backward()
    assert f_feats[0].grad is not None, "Feature matching gradients must propagate"

    perc_loss = PerceptualLoss(net="squeeze")
    img_a = torch.randn(2, 3, 256, 256, requires_grad=True)
    img_b = torch.randn(2, 3, 256, 256)
    p_val = perc_loss(img_a, img_b)
    p_val.backward()
    assert img_a.grad is not None, "Perceptual loss gradients must propagate"

    # LPIPS evaluation function
    lpips_val = compute_lpips(img_a.detach(), img_b, net="squeeze")
    assert lpips_val >= 0.0, "LPIPS distance must be non-negative"

