"""Tiny deterministic CPU checks; no dataset, GPU, training campaign, or Inception."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types

import torch
from torch import nn

from hw2_gan import model, utils

spec = importlib.util.spec_from_file_location("baseline_model", Path(__file__).with_name("baseline_model.py"))
baseline_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_model)


def equal_state(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def make_pair(width=2, **d_options):
    generator = model.Generator(8, width)
    discriminator = model.Discriminator(width, **d_options)
    generator.apply(model.initialize_weights)
    discriminator.apply(model.initialize_weights)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return generator, discriminator, g_optimizer, d_optimizer


def test_shapes():
    with torch.inference_mode():
        for width in (64, 96):
            generator = model.Generator(8, width).eval()
            discriminator = model.Discriminator(64).eval()
            fake = generator(torch.randn(1, 8))
            assert fake.shape == (1, 3, 64, 64)
            assert fake.min() >= -1 and fake.max() <= 1
            assert discriminator(fake).shape == (1,)
    print("PASS actual G64/D64 and G96/D64 shapes", flush=True)


def test_exact_baseline():
    torch.manual_seed(4)
    generator, discriminator, g_optimizer, d_optimizer = make_pair()
    old_g = baseline_model.Generator(8, 2)
    old_d = baseline_model.Discriminator(2)
    assert {key: value.shape for key, value in generator.state_dict().items()} == {
        key: value.shape for key, value in old_g.state_dict().items()}
    assert {key: value.shape for key, value in discriminator.state_dict().items()} == {
        key: value.shape for key, value in old_d.state_dict().items()}
    old_g.load_state_dict(generator.state_dict()); old_d.load_state_dict(discriminator.state_dict())
    old_g_optimizer = torch.optim.Adam(old_g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    old_d_optimizer = torch.optim.Adam(old_d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    old_rng = torch.Generator().manual_seed(12)
    new_rng = torch.Generator().manual_seed(12)
    real = torch.randn(2, 3, 64, 64).tanh()
    for _ in range(2):
        expected = baseline_model.adversarial_step(old_g, old_d, real, old_g_optimizer,
                                                 old_d_optimizer, old_rng)
        observed = model.adversarial_step(generator, discriminator, real, g_optimizer,
                                          d_optimizer, new_rng)
        assert all(observed[key] == expected[key] for key in expected)
        equal_state(generator.state_dict(), old_g.state_dict())
        equal_state(discriminator.state_dict(), old_d.state_dict())
    assert all(buffer.item() == 6 for name, buffer in discriminator.named_buffers()
               if name.endswith("num_batches_tracked"))
    assert torch.equal(new_rng.get_state(), old_rng.get_state())
    print("PASS exact legacy keys, parameters, BN buffers, metrics, noise stream: two Adam steps", flush=True)


def test_gradient_isolation():
    generator, discriminator, g_optimizer, d_optimizer = make_pair()
    fake = generator(torch.randn(2, 8))
    d_optimizer.zero_grad(set_to_none=True)
    nn.functional.binary_cross_entropy_with_logits(discriminator(fake.detach()), torch.zeros(2)).backward()
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in discriminator.parameters())
    d_optimizer.zero_grad(set_to_none=True)
    before = deepcopy(discriminator.state_dict())
    with model.frozen_parameters(discriminator), model.module_mode(discriminator, False):
        nn.functional.binary_cross_entropy_with_logits(discriminator(fake), torch.ones(2)).backward()
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in generator.parameters())
    assert model.parameter_gradient_norm(generator) > 0
    equal_state(before, discriminator.state_dict())
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    # Restore preexisting per-parameter freezes and mixed child modes, too.
    first = next(discriminator.parameters()); first.requires_grad_(False)
    discriminator.layers[3].eval()
    modes = [child.training for child in discriminator.modules()]
    with model.frozen_parameters(discriminator), model.module_mode(discriminator, False):
        pass
    assert not first.requires_grad
    assert modes == [child.training for child in discriminator.modules()]
    print("PASS detached D graph; G gradients through frozen D; weights and eval BN unchanged", flush=True)


def test_r1():
    # Independent analytic check of unnormalized C,H,W sum and batch mean.
    real = torch.randn(3, 2, 2, 2, requires_grad=True)
    weight = torch.arange(1.0, 9.0).reshape(1, 2, 2, 2).requires_grad_(True)
    penalty = model.r1_penalty((real*weight).flatten(1).sum(1), real)
    assert torch.equal(penalty, weight.square().sum())
    penalty.backward()
    assert torch.allclose(weight.grad, 2*weight.detach(), rtol=1e-6, atol=1e-7)
    for batch_norm in (False, True):
        pair = make_pair(batch_norm=batch_norm)
        generator, discriminator, g_optimizer, d_optimizer = pair
        metrics = model.adversarial_step(*pair[:2], torch.randn(2, 3, 64, 64).tanh(),
                                         g_optimizer, d_optimizer,
                                         torch.Generator().manual_seed(21),
                                         r1_gamma=1, r1_interval=3)
        assert metrics["r1_penalty"] > 0
        assert math.isclose(metrics["r1_loss"], 1.5*metrics["r1_penalty"], rel_tol=1e-6)
        assert all(math.isfinite(value) for value in metrics.values() if value is not None)
        assert all(torch.isfinite(parameter).all() for parameter in discriminator.parameters())
        counters = [buffer.item() for name, buffer in discriminator.named_buffers()
                    if name.endswith("num_batches_tracked")]
        assert all(value == 3 for value in counters), "R1 eval pass must not change BN counters."
        if batch_norm:
            try:
                model.adversarial_step(generator, discriminator, torch.randn(2, 3, 64, 64),
                                       g_optimizer, d_optimizer, r1_gamma=1, r1_batchnorm="error")
            except ValueError:
                pass
            else:
                raise AssertionError("BN guard did not reject the requested noBN-only R1 policy.")
    print("PASS analytic R1 value/gradient, finite noBN R1, fixed-stat BN R1, lazy factor", flush=True)


def test_sn_and_modes():
    generator, discriminator, g_optimizer, d_optimizer = make_pair(spectral_norm=True)
    assert not any(isinstance(child, nn.BatchNorm2d) for child in discriminator.modules())
    assert len([key for key in discriminator.state_dict() if key.endswith("weight_orig")]) == 5
    metrics = model.adversarial_step(generator, discriminator, torch.randn(2, 3, 64, 64),
                                     g_optimizer, d_optimizer, r1_gamma=1)
    assert metrics["g_gradient_norm"] > 0 and metrics["r1_penalty"] > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in discriminator.parameters())
    d_optimizer.zero_grad(set_to_none=True)
    before = deepcopy(discriminator.state_dict())
    metrics = model.adversarial_step(generator, discriminator, torch.randn(2, 3, 64, 64),
                                     g_optimizer, d_optimizer,
                                     update_discriminator=False, d_eval_for_g=True)
    equal_state(before, discriminator.state_dict())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert metrics["d_loss"] is None and metrics["d_updated"] == 0
    assert metrics["g_gradient_norm"] > 0 and discriminator.training
    for eval_for_g, expected in [(False, 3), (True, 2)]:
        generator, discriminator, g_optimizer, d_optimizer = make_pair()
        model.adversarial_step(generator, discriminator, torch.randn(2, 3, 64, 64),
                               g_optimizer, d_optimizer, d_eval_for_g=eval_for_g)
        assert all(buffer.item() == expected for name, buffer in discriminator.named_buffers()
                   if name.endswith("num_batches_tracked"))
    print("PASS SN forwards/R1/backprop, frozen SN input gradient, skip semantics, BN G-mode counts", flush=True)


def test_ema():
    generator, _, _, _ = make_pair()
    shadow = model.make_ema_generator(generator)
    before = deepcopy(shadow.state_dict())
    with torch.no_grad():
        for value in generator.state_dict().values():
            value.add_(2)
    model.update_ema(shadow, generator, 0.75)
    for key, value in shadow.state_dict().items():
        if value.is_floating_point():
            assert torch.allclose(value, before[key]+0.5, rtol=1e-6, atol=1e-7)
        else:
            assert torch.equal(value, before[key]+2)
    assert not shadow.training and not any(parameter.requires_grad for parameter in shadow.parameters())
    before = deepcopy(shadow.state_dict())
    with torch.inference_mode():
        assert torch.isfinite(shadow(torch.randn(2, 8))).all()
    equal_state(before, shadow.state_dict())
    print("PASS EMA float parameters/buffers, integer buffers, eval inference", flush=True)


def test_reproducibility_and_flip():
    def run():
        torch.manual_seed(987)
        generator, discriminator, g_optimizer, d_optimizer = make_pair()
        shadow = model.make_ema_generator(generator)
        real = torch.randn(2, 3, 64, 64)
        noise_rng = torch.Generator().manual_seed(15)
        augmentation_rng = torch.Generator().manual_seed(16)
        metrics = model.adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer,
                                         noise_rng, ema_generator=shadow, ema_decay=0.9,
                                         horizontal_flip=True, augmentation_rng=augmentation_rng)
        return generator.state_dict(), discriminator.state_dict(), shadow.state_dict(), metrics
    first, second = run(), run()
    for left, right in zip(first[:3], second[:3]):
        equal_state(left, right)
    assert first[3] == second[3]
    images = torch.arange(2*3*4*4).reshape(2, 3, 4, 4)
    before = images.clone()
    flipped = model.horizontal_flip_batch(images, torch.Generator().manual_seed(7))
    assert torch.equal(before, images)
    for original, augmented in zip(images, flipped):
        assert torch.equal(augmented, original) or torch.equal(augmented, original.flip(-1))
    print("PASS seeded initialization/noise/augmentation/EMA reproducibility; exact flip mapping", flush=True)


def test_checkpoint_loading():
    generator, _, _, _ = make_pair()
    shadow = model.make_ema_generator(generator)
    with torch.no_grad():
        next(shadow.parameters()).add_(0.1)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        old_path = Path(directory)/"old.pt"
        torch.save({"config": {"latent_dim": 8, "base_channels": 2},
                    "generator_state": generator.state_dict()}, old_path)
        loaded, _ = utils.load_generator(old_path, "cpu")
        equal_state(loaded.state_dict(), generator.state_dict())
        assert not loaded.training
        try:
            utils.load_generator(old_path, "cpu", weights="ema")
        except ValueError:
            pass
        else:
            raise AssertionError("Missing EMA was silently accepted.")
        new_path = Path(directory)/"new.pt"
        torch.save({"config": {"latent_dim": 8, "g_base_channels": 2, "base_channels": 64},
                    "generator_state": generator.state_dict(),
                    "generator_ema_state": shadow.state_dict()}, new_path)
        raw, _ = utils.load_generator(new_path, "cpu")
        ema, _ = utils.load_generator(new_path, "cpu", weights="ema")
        equal_state(raw.state_dict(), generator.state_dict())
        equal_state(ema.state_dict(), shadow.state_dict())
        assert not raw.training and not ema.training
    print("PASS legacy/new raw checkpoints, explicit EMA loading, missing-EMA error", flush=True)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    for check in [test_shapes, test_exact_baseline, test_gradient_isolation, test_r1,
                  test_sn_and_modes, test_ema, test_reproducibility_and_flip,
                  test_checkpoint_loading]:
        check()
    print("All tiny CPU synthetic checks passed. No trained-image-quality claim.", flush=True)


if __name__ == "__main__":
    main()
