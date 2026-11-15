"""Mask-aware epsilon/v-prediction loss with Min-SNR timestep weighting."""


def facade_loss(prediction, target, labels, valid, timesteps, scheduler, config):
    import torch
    import torch.nn.functional as F
    openings = ((labels == 2) | (labels == 3)).float()[:, None]
    foreground = (labels != 0).float()[:, None]
    weights = config['background_loss_weight'] + foreground
    weights += config['opening_loss_weight'] * F.max_pool2d(openings, 7, 1, 3)
    weights *= valid[:, None]
    weights = F.interpolate(weights, size=prediction.shape[-2:], mode='area')
    numerator = ((prediction.float() - target.float()).square() * weights).sum(dim=(1, 2, 3))
    denominator = (weights.sum(dim=(1, 2, 3)) * prediction.shape[1]).clamp_min(1e-6)
    alphas = scheduler.alphas_cumprod.to(prediction.device)[timesteps].float()
    snr = alphas / (1 - alphas).clamp_min(1e-8)
    gamma = torch.minimum(snr, torch.full_like(snr, config['min_snr_gamma']))
    balance = gamma / (snr + 1 if scheduler.config.prediction_type == 'v_prediction' else snr.clamp_min(1e-8))
    return ((numerator / denominator) * balance).mean()
