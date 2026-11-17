"""Tiny CPU-only transform/toy-gradient checks; no data or full GAN training."""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import warnings

import torch
from torch import nn

from hw2_gan import diffaugment as augmentation
from hw2_gan import model
source_spec = importlib.util.spec_from_file_location("official_diffaugment", Path(__file__).with_name("official_DiffAugment_pytorch.py"))
official = importlib.util.module_from_spec(source_spec)
source_spec.loader.exec_module(official)


class ToyGenerator(nn.Module):
    latent_dim = 4

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(self.latent_dim, 3*8*8)

    def forward(self, latent):
        return self.linear(latent).reshape(-1, 3, 8, 8).tanh()


class ToyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 2, 3, 1, 1), nn.BatchNorm2d(2),
                                    nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(), nn.Linear(2, 1))

    def forward(self, images):
        return self.layers(images).flatten()


def equal_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def check_official_parity():
    for shape in [(4, 3, 8, 10), (3, 3, 7, 9)]:
        images = torch.linspace(-1, 1, torch.tensor(shape).prod().item()).reshape(shape)
        for policy in ("color", "translation", "cutout", "color,translation,cutout",
                       "translation,cutout,color"):
            with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                torch.manual_seed(17)
                expected = official.DiffAugment(images, policy=policy)
            augmenter = augmentation.DifferentiableAugment(policy, seed=17)
            observed = augmenter(images)
            assert torch.equal(observed, expected), (shape, policy, (observed-expected).abs().max())
    print("PASS exact pinned-author-code parity for all transforms, policy orders, odd/even shapes", flush=True)


def check_identity_rng_state_and_ranges():
    images = torch.linspace(-1, 1, 2*3*8*8).reshape(2, 3, 8, 8)
    empty = augmentation.DifferentiableAugment("", seed=20)
    private_before, global_before = empty.rng.get_state().clone(), torch.get_rng_state().clone()
    assert empty(images) is images
    assert torch.equal(empty.rng.get_state(), private_before)
    assert torch.equal(torch.get_rng_state(), global_before)
    for policy in ("translation", "cutout", "translation,cutout"):
        transformed = augmentation.DifferentiableAugment(policy, seed=20)(images)
        assert transformed.shape == images.shape and transformed.dtype == images.dtype
        assert transformed.min() >= -1 and transformed.max() <= 1
    color = augmentation.DifferentiableAugment("color", seed=42)(torch.ones_like(images))
    assert color.max() > 1, "Color must retain authors' unclamped behavior."
    assert torch.isfinite(color).all()
    augmenter = augmentation.DifferentiableAugment("color,translation,cutout", seed=20)
    initial = deepcopy(augmenter.state_dict())
    global_before = torch.get_rng_state().clone()
    first = augmenter(images)
    assert torch.equal(torch.get_rng_state(), global_before), "Augmentation consumed global RNG."
    augmenter.load_state_dict(initial)
    assert torch.equal(first, augmenter(images))
    try:
        augmentation.DifferentiableAugment("translation", seed=20).load_state_dict(initial)
    except ValueError:
        pass
    else:
        raise AssertionError("A resumed policy changed silently.")
    print("PASS zero-policy identity/no RNG draws, private RNG restore, valid geometry ranges, unclamped color", flush=True)


def check_input_gradient():
    images = torch.linspace(-1, 1, 2*3*8*8).reshape(2, 3, 8, 8).requires_grad_(True)
    transformed = augmentation.DifferentiableAugment("color,translation,cutout", seed=33)(images)
    transformed.square().mean().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
    print("PASS finite nonzero input gradient through color/translation/cutout", flush=True)


def run_step(policy, *, seed=40, eval_for_g=False):
    torch.manual_seed(seed)
    generator, discriminator = ToyGenerator(), ToyDiscriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    images = torch.linspace(-1, 1, 2*3*8*8).reshape(2, 3, 8, 8)
    augmenter = augmentation.DifferentiableAugment(policy, seed=seed+1)
    noise = torch.Generator().manual_seed(seed+2)
    captured = {}
    original_step = d_optimizer.step
    def d_step(*args, **kwargs):
        assert all(parameter.grad is None for parameter in generator.parameters()), "D update reached G."
        result = original_step(*args, **kwargs)
        captured["d_parameters"] = [parameter.detach().clone() for parameter in discriminator.parameters()]
        return result
    d_optimizer.step = d_step
    trace = []
    values = augmentation.adversarial_step_diffaugment(generator, discriminator, images,
        g_optimizer, d_optimizer, noise, augmenter=augmenter, trace=trace, d_eval_for_g=eval_for_g)
    assert all(torch.equal(before, after) for before, after in zip(captured["d_parameters"], discriminator.parameters()))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in generator.parameters())
    assert values["g_gradient_norm"] > 0
    return generator.state_dict(), discriminator.state_dict(), values, augmenter.state_dict(), trace


def check_consistent_branches_and_reproducibility():
    first = run_step("color,translation,cutout")
    second = run_step("color,translation,cutout")
    equal_state(first[0], second[0]); equal_state(first[1], second[1])
    assert first[2] == second[2] and first[4] == second[4]
    assert torch.equal(first[3]["rng_state"], second[3]["rng_state"])
    assert first[4] == [{"stage": stage, "policy": "color,translation,cutout"}
                        for stage in ("real_d", "fake_d", "fake_g")]
    default_counter = first[1]["layers.1.num_batches_tracked"].item()
    evaluated = run_step("translation,cutout", eval_for_g=True)
    assert default_counter == 3 and evaluated[1]["layers.1.num_batches_tracked"].item() == 2
    print("PASS consistent real-D/fake-D/fake-G policy, detached D, G derivatives through augmented frozen D, reproducibility", flush=True)


def check_zero_policy_exact_default():
    torch.manual_seed(40)
    generator, discriminator = ToyGenerator(), ToyDiscriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    images = torch.linspace(-1, 1, 2*3*8*8).reshape(2, 3, 8, 8)
    values = model.adversarial_step(generator, discriminator, images, g_optimizer, d_optimizer,
                                    torch.Generator().manual_seed(42))
    observed = run_step("")
    equal_state(generator.state_dict(), observed[0]); equal_state(discriminator.state_dict(), observed[1])
    assert values == observed[2] and observed[4] == []
    print("PASS empty-policy update exactly equals frozen historical-default step", flush=True)


def main():
    torch.set_num_threads(1)
    for check in [check_official_parity, check_identity_rng_state_and_ranges, check_input_gradient,
                  check_consistent_branches_and_reproducibility, check_zero_policy_exact_default]:
        check()
    print("All tiny optional DiffAugment checks passed; no production training or image-quality claim.", flush=True)


if __name__ == "__main__":
    main()
