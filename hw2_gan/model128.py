"""Native RGB128 DCGAN demo; RGB64 model/state definitions remain untouched.

No resizing or interpolation occurs in these networks. Generator stages double
spatial size from4 to128; discriminator stages reduce128 to one real logit.
"""

import torch
from torch import nn


class Generator128(nn.Module):
    image_size = 128

    def __init__(self, latent_dim=100, base_channels=64):
        super().__init__()
        if latent_dim < 1 or base_channels < 1:
            raise ValueError("Latent dimension and base channels must be positive.")
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        c = base_channels
        layers = []
        channels = [latent_dim, 16*c, 8*c, 4*c, 2*c, c]
        for index, (before, after) in enumerate(zip(channels[:-1], channels[1:])):
            stride, padding = (1, 0) if index == 0 else (2, 1)
            layers.extend([nn.ConvTranspose2d(before, after, 4, stride, padding, bias=False),
                           nn.BatchNorm2d(after), nn.ReLU(True)])
        layers.extend([nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False), nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, latent):
        if latent.ndim not in (2, 4) or latent.shape[1] != self.latent_dim \
                or (latent.ndim == 4 and tuple(latent.shape[2:]) != (1, 1)):
            raise ValueError("Expected latent shape (N,latent_dim) or (N,latent_dim,1,1).")
        return self.layers(latent.reshape(-1, self.latent_dim, 1, 1))

    def stage_shapes(self, batch_size=1):
        c = self.base_channels
        return [(batch_size, self.latent_dim, 1, 1), (batch_size, 16*c, 4, 4),
                (batch_size, 8*c, 8, 8), (batch_size, 4*c, 16, 16),
                (batch_size, 2*c, 32, 32), (batch_size, c, 64, 64),
                (batch_size, 3, 128, 128)]


class Discriminator128(nn.Module):
    image_size = 128

    def __init__(self, base_channels=64, *, spectral_norm=False, batch_norm=None):
        super().__init__()
        if base_channels < 1:
            raise ValueError("Base channels must be positive.")
        self.base_channels = base_channels
        self.spectral_norm = bool(spectral_norm)
        self.batch_norm = not self.spectral_norm if batch_norm is None else bool(batch_norm)
        if self.spectral_norm and self.batch_norm:
            raise ValueError("The SN arm uses no D BatchNorm.")
        c = base_channels

        def convolution(before, after, stride, padding):
            layer = nn.Conv2d(before, after, 4, stride, padding, bias=False)
            return nn.utils.spectral_norm(layer) if self.spectral_norm else layer

        layers = [convolution(3, c, 2, 1), nn.LeakyReLU(0.2, True)]
        for before, after in [(c, 2*c), (2*c, 4*c), (4*c, 8*c), (8*c, 16*c)]:
            layers.append(convolution(before, after, 2, 1))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(after))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(convolution(16*c, 1, 1, 0))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
            raise ValueError("Discriminator128 requires native (N,3,128,128) images.")
        return self.layers(images).flatten()

    def stage_shapes(self, batch_size=1):
        c = self.base_channels
        return [(batch_size, 3, 128, 128), (batch_size, c, 64, 64),
                (batch_size, 2*c, 32, 32), (batch_size, 4*c, 16, 16),
                (batch_size, 8*c, 8, 8), (batch_size, 16*c, 4, 4),
                (batch_size, 1, 1, 1)]


def parameter_counts(latent_dim=100, g_base_channels=64, d_base_channels=64, *, d_batch_norm=True):
    """Exact trainable parameter counts for these bias-free convolution stages."""
    g = 256*latent_dim*g_base_channels+2720*g_base_channels**2+110*g_base_channels
    d = 2720*d_base_channels**2+(364 if d_batch_norm else 304)*d_base_channels
    return {"generator": g, "discriminator": d}
