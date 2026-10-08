"""
Deep Learning Model Architectures for ArchiGen 3D (Pix2Pix Conditional GAN).
Features:
1. UNetGenerator (256x256) with skip-connections to preserve high-frequency structural lines.
2. PatchGANDiscriminator (70x70) with Spectral Normalization for Minimax stabilization.
3. Gaussian weight initialization (mean=0.0, std=0.02) per Isola et al. (2017).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def weights_init_normal(m):
    """Initializes weights with Gaussian distribution (mean=0.0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """Downsampling block: Convolution -> InstanceNorm -> LeakyReLU."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Upsampling block:
    Replaces ConvTranspose2d with Upsample(nearest) + Conv2d to eliminate checkerboard artifacts.
    (Reference: Odena et al., 'Deconvolution and Checkerboard Artifacts', Distill 2016).
    """
    def __init__(self, in_size, out_size, dropout=0.0, upsample_mode="nearest"):
        super().__init__()
        if upsample_mode == "transpose":
            up_layer = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            # Nearest-neighbor or Bilinear interpolation followed by standard convolution
            up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
            )

        layers = [
            up_layer,
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net 256 Generator with 8 downsampling and 7 upsampling stages.
    Skip connections transfer fine details directly from encoder to decoder.
    Configurable upsample_mode: 'nearest' (prevents checkerboard artifacts) or 'transpose'.
    """
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, upsample_mode="nearest"):
        super().__init__()
        self.upsample_mode = upsample_mode

        # Encoder (Downsampling)
        self.down1 = UNetDown(in_channels, num_filters, normalize=False)        # 256 -> 128
        self.down2 = UNetDown(num_filters, num_filters * 2)                     # 128 -> 64
        self.down3 = UNetDown(num_filters * 2, num_filters * 4)                 # 64 -> 32
        self.down4 = UNetDown(num_filters * 4, num_filters * 8)                 # 32 -> 16
        self.down5 = UNetDown(num_filters * 8, num_filters * 8)                 # 16 -> 8
        self.down6 = UNetDown(num_filters * 8, num_filters * 8)                 # 8 -> 4
        self.down7 = UNetDown(num_filters * 8, num_filters * 8)                 # 4 -> 2
        self.down8 = UNetDown(num_filters * 8, num_filters * 8, normalize=False)# 2 -> 1 (Bottleneck)

        # Decoder (Upsampling with Skip Connections without checkerboard artifacts)
        self.up1 = UNetUp(num_filters * 8, num_filters * 8, dropout=0.5, upsample_mode=upsample_mode) # 1 -> 2
        self.up2 = UNetUp(num_filters * 16, num_filters * 8, dropout=0.5, upsample_mode=upsample_mode)# 2 -> 4
        self.up3 = UNetUp(num_filters * 16, num_filters * 8, dropout=0.5, upsample_mode=upsample_mode)# 4 -> 8
        self.up4 = UNetUp(num_filters * 16, num_filters * 8, upsample_mode=upsample_mode)             # 8 -> 16
        self.up5 = UNetUp(num_filters * 16, num_filters * 4, upsample_mode=upsample_mode)             # 16 -> 32
        self.up6 = UNetUp(num_filters * 8, num_filters * 2, upsample_mode=upsample_mode)              # 32 -> 64
        self.up7 = UNetUp(num_filters * 4, num_filters, upsample_mode=upsample_mode)                  # 64 -> 128

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(num_filters * 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Normalizes output to [-1, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 70x70 Discriminator with optional Spectral Normalization.
    Penalizes high-frequency structural errors at local patches instead of full image.
    Takes concatenated input: [sketch (3) + photo (3)] = 6 channels.
    """
    def __init__(self, in_channels=6, num_filters=64, use_spectral_norm=True):
        super().__init__()

        def disc_block(in_f, out_f, normalization=True):
            conv = nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers = [conv]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, num_filters, normalization=False), # 256 -> 128
            *disc_block(num_filters, num_filters * 2),                  # 128 -> 64
            *disc_block(num_filters * 2, num_filters * 4),              # 64 -> 32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, padding=1, bias=False),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, padding=1)     # 1-channel Patch prediction
        )

    def forward(self, img_a, img_b):
        # Concatenate condition (sketch) and image (real or fake photo) along channel axis
        img_input = torch.cat((img_a, img_b), 1)
        return self.model(img_input)
