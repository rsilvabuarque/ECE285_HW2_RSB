"""RGB64 DCGAN and explicitly controlled adversarial updates.

Historical defaults retain the same state-dict keys, tensor shapes, random-noise
draw, optimizer ordering, BCE sum, and train-mode discriminator BN forwards.
"""

from contextlib import contextmanager
from copy import deepcopy
import math

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
    def __init__(self, base_channels=64, *, spectral_norm=False, batch_norm=None):
        super().__init__()
        self.base_channels = base_channels
        self.spectral_norm = bool(spectral_norm)
        self.batch_norm = not self.spectral_norm if batch_norm is None else bool(batch_norm)
        if self.spectral_norm and self.batch_norm:
            raise ValueError("The SN arm uses no D BatchNorm; set batch_norm=False.")
        c = base_channels

        def convolution(in_channels, out_channels, stride, padding):
            layer = nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias=False)
            return nn.utils.spectral_norm(layer) if self.spectral_norm else layer

        # Keep these positions unchanged when BN is enabled for legacy loading.
        layers = [convolution(3, c, 2, 1), nn.LeakyReLU(0.2, True)]
        for in_channels, out_channels in [(c, 2*c), (2*c, 4*c), (4*c, 8*c)]:
            layers.append(convolution(in_channels, out_channels, 2, 1))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(convolution(8*c, 1, 1, 0))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        # BCEWithLogitsLoss and R1 both receive logits, never sigmoid outputs.
        return self.layers(images).flatten()


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        # Legacy spectral_norm stores the optimized tensor as weight_orig.
        nn.init.normal_(getattr(module, "weight_orig", module.weight), 0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1, 0.02)
        nn.init.zeros_(module.bias)


@contextmanager
def module_mode(module, training):
    """Temporarily set a mode and restore each submodule's prior mode exactly."""
    previous = [(child, child.training) for child in module.modules()]
    module.train(training)
    try:
        yield module
    finally:
        for child, flag in previous:
            child.training = flag


@contextmanager
def frozen_parameters(module):
    """Freeze D weights while retaining derivatives with respect to fake images."""
    previous = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    for parameter, _ in previous:
        parameter.requires_grad_(False)
    try:
        yield module
    finally:
        for parameter, flag in previous:
            parameter.requires_grad_(flag)


def r1_penalty(real_logits, real_images):
    """Mean squared input-logit gradient, before multiplication by gamma/2.

    Images have the existing dimensionless [-1,1] RGB coordinates. The gradient
    is summed over C,H,W, then averaged over N. This sum-of-logits shortcut is
    valid only for a discriminator that does not couple minibatch examples.
    In particular, use fixed running statistics for any BatchNorm layers.
    """
    if real_logits.shape != (real_images.shape[0],):
        raise ValueError("R1 requires exactly one discriminator logit per image.")
    gradient, = torch.autograd.grad(real_logits.sum(), real_images, create_graph=True)
    return gradient.square().flatten(1).sum(1).mean()


def parameter_gradient_norm(module):
    """Euclidean norm of all trainable-parameter gradients before optimizer.step."""
    squared = [parameter.grad.detach().square().sum()
               for parameter in module.parameters() if parameter.grad is not None]
    if not squared:
        return 0.0
    return torch.stack(squared).sum().sqrt().item()


def make_ema_generator(generator):
    """Initialize the inference-only shadow exactly from the current raw model."""
    shadow = deepcopy(generator).eval()
    shadow.requires_grad_(False)
    return shadow


@torch.no_grad()
def update_ema(ema_generator, generator, decay):
    """One EMA step after a successful G optimizer step.

    Floating parameters AND floating BN buffers use decay*shadow+(1-decay)*raw.
    Integer buffers (e.g. num_batches_tracked) are copied without interpolation.
    The shadow never runs in train mode; its running BN statistics are averages
    of raw running statistics, not separately recalibrated population moments.
    """
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("EMA decay must be finite and in [0,1).")
    source = generator.state_dict()
    destination = ema_generator.state_dict()
    if source.keys() != destination.keys():
        raise ValueError("EMA and raw model state keys differ.")
    for name, target in destination.items():
        current = source[name].detach()
        if target.shape != current.shape or target.dtype != current.dtype:
            raise ValueError(f"EMA state mismatch for {name}.")
        if torch.is_floating_point(target) or torch.is_complex(target):
            target.mul_(decay).add_(current, alpha=1-decay)
        else:
            target.copy_(current)
    ema_generator.eval()


def horizontal_flip_batch(images, rng=None):
    """Independent p=1/2 real-image horizontal flips, preserving [-1,1].

    Supply a dedicated generator on images.device and checkpoint its state so
    augmentation does not consume the GAN noise stream or global RNG.
    """
    mask = torch.rand(images.shape[0], 1, 1, 1, device=images.device, generator=rng) < 0.5
    return torch.where(mask, images.flip(-1), images)


def adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer, rng=None,
                     *, update_discriminator=True, r1_gamma=0.0, r1_interval=1,
                     apply_r1=True, r1_batchnorm="eval", d_eval_for_g=False,
                     ema_generator=None, ema_decay=0.0, horizontal_flip=False,
                     augmentation_rng=None):
    """An optional D optimizer step, then one non-saturating G optimizer step.

    For periodic R1, caller counts D optimizer steps, applies on every k-th step,
    and supplies r1_interval=k. Multiplication by k preserves its mean coefficient
    gamma/2 per D update; Adam hyperparameters are left unchanged and that choice
    must be recorded. Skipping D skips its real/fake forwards completely: D metrics
    are None on such batches. Its default train-mode G forward still updates BN.

    R1+BN policy is explicit: BCE uses the historical train-mode forwards; R1 uses
    an additional eval-mode real forward with fixed current running BN statistics.
    Thus it penalizes the inference-mode logit function, not a batch-coupled BN
    training function. Set r1_batchnorm='error' to require a no-BN discriminator.
    No no_grad context is used on the D forward that supplies G gradients.
    """
    if not math.isfinite(r1_gamma) or r1_gamma < 0 or r1_interval < 1:
        raise ValueError("R1 requires finite gamma>=0 and interval>=1.")
    if r1_batchnorm not in ("eval", "error"):
        raise ValueError("r1_batchnorm must be 'eval' or 'error'.")
    if ema_generator is not None and (not math.isfinite(ema_decay) or not 0 <= ema_decay < 1):
        raise ValueError("EMA decay must be finite and in [0,1).")
    criterion = nn.BCEWithLogitsLoss()
    size = real.shape[0]
    noise = torch.randn(size, generator.latent_dim, device=real.device, generator=rng)
    fake = generator(noise)
    if horizontal_flip:
        real = horizontal_flip_batch(real, augmentation_rng)
    regularize = bool(update_discriminator and r1_gamma and apply_r1)
    has_bn = any(isinstance(child, nn.modules.batchnorm._BatchNorm)
                 for child in discriminator.modules())
    if regularize and has_bn and r1_batchnorm == "error":
        raise ValueError("Training BatchNorm couples samples; use noBN or explicit eval R1.")
    metrics = {key: None for key in ("d_loss", "d_real_loss", "d_fake_loss",
                                   "d_real_probability", "d_fake_probability")}
    metrics.update(d_updated=int(bool(update_discriminator)), r1_applied=int(regularize),
                   r1_penalty=0.0, r1_loss=0.0)
    if update_discriminator:
        d_optimizer.zero_grad(set_to_none=True)
        # Avoid adding input gradients/graphs in historical gamma=0 runs.
        real_for_d = real.detach().requires_grad_(regularize and not has_bn) if regularize else real
        real_logits = discriminator(real_for_d)
        fake_logits = discriminator(fake.detach())
        d_real = criterion(real_logits, torch.ones_like(real_logits))
        d_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
        d_loss = d_real + d_fake
        if regularize:
            if has_bn:
                real_for_r1 = real.detach().requires_grad_(True)
                with module_mode(discriminator, False):
                    penalty = r1_penalty(discriminator(real_for_r1), real_for_r1)
            else:
                penalty = r1_penalty(real_logits, real_for_d)
            contribution = (0.5*r1_gamma*r1_interval)*penalty
            d_loss = d_loss + contribution
            metrics.update(r1_penalty=penalty.detach().item(), r1_loss=contribution.detach().item())
        if not torch.isfinite(d_loss):
            raise FloatingPointError("Nonfinite discriminator loss.")
        d_loss.backward()
        d_optimizer.step()
        metrics.update(d_loss=d_loss.item(), d_real_loss=d_real.item(), d_fake_loss=d_fake.item(),
                       d_real_probability=real_logits.detach().sigmoid().mean().item(),
                       d_fake_probability=fake_logits.detach().sigmoid().mean().item())

    g_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(discriminator):
        with module_mode(discriminator, False if d_eval_for_g else discriminator.training):
            generated_logits = discriminator(fake)
            g_loss = criterion(generated_logits, torch.ones_like(generated_logits))
            if not torch.isfinite(g_loss):
                raise FloatingPointError("Nonfinite generator loss.")
            g_loss.backward()
        g_gradient_norm = parameter_gradient_norm(generator)
        if not math.isfinite(g_gradient_norm):
            raise FloatingPointError("Nonfinite generator gradient norm.")
        g_optimizer.step()
    if ema_generator is not None:
        update_ema(ema_generator, generator, ema_decay)
    metrics.update(g_loss=g_loss.item(), g_gradient_norm=g_gradient_norm,
                   d_fake_after_d_probability=generated_logits.detach().sigmoid().mean().item())
    return metrics
