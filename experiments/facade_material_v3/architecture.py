"""Extend the selected V2 ControlNet without changing its initial response."""


def upgrade_material_conditioning(model):
    import torch
    from torch import nn
    old = model.controlnet_cond_embedding.conv_in
    if old.in_channels != 7:
        raise ValueError('Expected the selected seven-channel V2 ControlNet')
    new = nn.Conv2d(10, old.out_channels, old.kernel_size, old.stride, old.padding,
        bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :7].copy_(old.weight)
        if old.bias is not None: new.bias.copy_(old.bias)
    model.controlnet_cond_embedding.conv_in = new
    model.register_to_config(conditioning_channels=10)
    return model


def make_teacher(width=24):
    """Small six-class GroupNorm UNet, independent of the diffusion control input."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
    class FacadeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([block(3, width), block(width, width*2), block(width*2, width*4), block(width*4, width*8)])
            self.decoders = nn.ModuleList([block(width*12, width*4), block(width*6, width*2), block(width*3, width)])
            self.head = nn.Conv2d(width, 6, 1)
        def forward(self, x):
            skips = []
            for i, layer in enumerate(self.encoders):
                if i: x = F.avg_pool2d(x, 2)
                x = layer(x); skips.append(x)
            for layer, skip in zip(self.decoders, reversed(skips[:-1])):
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = layer(torch.cat([x, skip], 1))
            return self.head(x)
    return FacadeTeacher()
