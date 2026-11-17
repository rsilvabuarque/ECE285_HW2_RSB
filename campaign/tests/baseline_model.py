"""Convolutional generator and discriminator for RGB 64x64 images."""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        c = base_channels
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 8*c, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8*c), nn.ReLU(True),
            nn.ConvTranspose2d(8*c, 4*c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*c), nn.ReLU(True),
            nn.ConvTranspose2d(4*c, 2*c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*c), nn.ReLU(True),
            nn.ConvTranspose2d(2*c, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(True),
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.layers(z.reshape(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.layers = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, 2*c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*c, 4*c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4*c, 8*c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8*c, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images):
        # Logits are passed to BCEWithLogitsLoss; sigmoid is fused into that loss.
        return self.layers(images).flatten()


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)


def adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer, rng=None):
    """One D step, then one non-saturating G step, as in the lecture.

    D's parameters are frozen during the G step, but D remains in training
    mode: its BatchNorm running buffers update on that forward, as in the
    displayed lecture sequence. They are distinct from trainable parameters.
    """
    criterion = nn.BCEWithLogitsLoss()
    size = real.shape[0]
    noise = torch.randn(size, generator.latent_dim, device=real.device, generator=rng)
    fake = generator(noise)
    d_optimizer.zero_grad(set_to_none=True)
    real_logits = discriminator(real)
    fake_logits = discriminator(fake.detach())
    d_real = criterion(real_logits, torch.ones_like(real_logits))
    d_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    d_loss = d_real + d_fake
    if not torch.isfinite(d_loss):
        raise FloatingPointError("Nonfinite discriminator loss.")
    d_loss.backward()
    d_optimizer.step()

    g_optimizer.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        generated_logits = discriminator(fake)
        g_loss = criterion(generated_logits, torch.ones_like(generated_logits))
        if not torch.isfinite(g_loss):
            raise FloatingPointError("Nonfinite generator loss.")
        g_loss.backward()
        g_optimizer.step()
    finally:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
    return {"g_loss": g_loss.item(), "d_loss": d_loss.item(),
            "d_real_loss": d_real.item(), "d_fake_loss": d_fake.item(),
            "d_real_probability": real_logits.detach().sigmoid().mean().item(),
            "d_fake_probability": fake_logits.detach().sigmoid().mean().item(),
            "d_fake_after_d_probability": generated_logits.detach().sigmoid().mean().item()}
