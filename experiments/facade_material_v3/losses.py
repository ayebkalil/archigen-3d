"""Differentiable low-noise image feedback, separate from the diffusion objective."""


def predicted_clean(noisy, prediction, timesteps, scheduler):
    import torch
    a = scheduler.alphas_cumprod.to(noisy.device)[timesteps].float().view(-1, 1, 1, 1)
    if scheduler.config.prediction_type == 'epsilon':
        return (noisy.float() - (1-a).sqrt() * prediction.float()) / a.sqrt().clamp_min(1e-6)
    if scheduler.config.prediction_type == 'v_prediction':
        return a.sqrt() * noisy.float() - (1-a).sqrt() * prediction.float()
    raise ValueError('Unsupported diffusion prediction type')


def semantic_feedback(logits, labels, valid):
    import torch
    import torch.nn.functional as F
    target = F.interpolate(labels[:, None].float(), logits.shape[-2:], mode='nearest')[:, 0].long()
    weight = F.interpolate(valid[:, None].float(), logits.shape[-2:], mode='nearest')[:, 0]
    ce = F.cross_entropy(logits.float(), target, reduction='none',
        weight=logits.new_tensor([.2, 1, 3, 4, 1, 1]).float())
    ce = (ce * weight).sum() / weight.sum().clamp_min(1)
    opening = ((target == 2) | (target == 3)).float()[:, None]
    near = F.max_pool2d(opening, 5, 1, 2)[:, 0] > 0
    wall = (target == 1) & ~near & (weight > 0)
    probability = logits.float().softmax(1)[:, 2:4].sum(1)
    false_opening = (probability * wall).sum() / wall.sum().clamp_min(1)
    return ce, false_opening


def material_statistics(image, reference, labels, valid):
    import torch
    import torch.nn.functional as F
    mask = ((labels == 1).float() * valid)[:, None]
    mask = F.interpolate(mask, image.shape[-2:], mode='nearest')
    ref = F.interpolate(reference.float(), image.shape[-2:], mode='bilinear', align_corners=False)
    image = (image.float() + 1) / 2
    count = mask.sum((2,3)).clamp_min(1)
    def stats(x):
        mean = (x*mask).sum((2,3))/count
        var = ((x-mean[:,:,None,None]).square()*mask).sum((2,3))/count
        return mean, (var + 1e-6).sqrt()
    a, sa = stats(image); b, sb = stats(ref)
    return (a-b).abs().mean() + .25*(sa-sb).abs().mean()
