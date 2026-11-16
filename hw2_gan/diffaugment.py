"""Optional seeded DiffAugment; historical model/train drafts stay unchanged.

Adapted from the authors' portable PyTorch implementation, pinned to
mit-han-lab/data-efficient-gans commit 0b6339510be1e3859d360cbf464d0a425514a6b4.
Copyright (c) 2020 Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han.
Redistribution terms and disclaimer: LICENSE.diffaugment.BSD-2-Clause.txt.

The adaptation adds an isolated checkpointable RNG, NCHW validation, explicit
policy metadata, and a lightweight adapter applying the same stochastic policy
to real-D, fake-D, and fake-G inputs. Color outputs are intentionally not clamped.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


POLICY_VERSION = "zhao2020-pytorch-seeded-v1"
SOURCE_COMMIT = "0b6339510be1e3859d360cbf464d0a425514a6b4"
POLICIES = ("color", "translation", "cutout")


def parse_policy(policy):
    names = tuple(part.strip() for part in policy.split(",") if part.strip())
    if len(names) != len(set(names)) or any(name not in POLICIES for name in names):
        raise ValueError("Use each of color,translation,cutout at most once, or the empty policy.")
    return names


class DifferentiableAugment:
    """A policy distribution shared by all D branches, with independent draws.

    Randomness is image-wise, independent across calls, and confined to this
    generator. Call even during the frozen-D G step: gradients must pass through
    augmentation. Evaluate the unwrapped G/D without this object for diagnostics.
    """
    def __init__(self, policy="", *, seed=45, device="cpu", translation_ratio=0.125,
                 cutout_ratio=0.5):
        self.policy = parse_policy(policy)
        self.device = torch.device(device)
        if not 0 <= seed < 2**63:
            raise ValueError("Augmentation seed must lie in [0,2**63).")
        if not math.isfinite(translation_ratio) or not 0 <= translation_ratio <= 1:
            raise ValueError("Translation ratio must be finite and lie in [0,1].")
        if not math.isfinite(cutout_ratio) or not 0 < cutout_ratio <= 1:
            raise ValueError("Cutout ratio must be finite and lie in (0,1].")
        self.seed = int(seed)
        self.translation_ratio = float(translation_ratio)
        self.cutout_ratio = float(cutout_ratio)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)

    @property
    def config(self):
        return {"version": POLICY_VERSION, "source_commit": SOURCE_COMMIT,
                "policy": ",".join(self.policy), "seed": self.seed,
                "device_type": self.device.type, "translation_ratio": self.translation_ratio,
                "cutout_ratio": self.cutout_ratio,
                "branch_policy": "same policy, independent draws: real_d, fake_d, fake_g",
                "color_range": "unclamped; values may extend beyond [-1,1]"}

    def state_dict(self):
        return {"config": self.config, "rng_state": self.rng.get_state()}

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("DiffAugment policy/config differs from its checkpoint.")
        self.rng.set_state(state["rng_state"])

    def __call__(self, images):
        # Preserve identity AND all RNG state for a zero-policy baseline.
        if not self.policy:
            return images
        if images.ndim != 4 or images.shape[1] != 3 or not len(images) \
                or not torch.is_floating_point(images):
            raise ValueError("Expected nonempty floating RGB NCHW images.")
        if images.device.type != self.device.type:
            raise ValueError("DiffAugment RNG must use the image device type.")
        output = images
        for operation in self.policy:
            if operation == "color":
                output = self._color(output)
            elif operation == "translation":
                output = self._translation(output)
            else:
                output = self._cutout(output)
        return output.contiguous()

    def _uniform(self, images):
        return torch.rand(len(images), 1, 1, 1, dtype=images.dtype,
                          device=images.device, generator=self.rng)

    def _color(self, images):
        # Authors' brightness [-.5,.5), saturation [0,2), contrast [.5,1.5).
        images = images+(self._uniform(images)-0.5)
        channel_mean = images.mean(dim=1, keepdim=True)
        images = (images-channel_mean)*(2*self._uniform(images))+channel_mean
        image_mean = images.mean(dim=(1, 2, 3), keepdim=True)
        return (images-image_mean)*(self._uniform(images)+0.5)+image_mean

    def _translation(self, images):
        n, _, height, width = images.shape
        max_y = int(height*self.translation_ratio+0.5)
        max_x = int(width*self.translation_ratio+0.5)
        shift_y = torch.randint(-max_y, max_y+1, (n, 1, 1),
                                device=images.device, generator=self.rng)
        shift_x = torch.randint(-max_x, max_x+1, (n, 1, 1),
                                device=images.device, generator=self.rng)
        batches = torch.arange(n, device=images.device)[:, None, None]
        rows = torch.arange(height, device=images.device)[None, :, None]
        columns = torch.arange(width, device=images.device)[None, None, :]
        rows = (rows+shift_y+1).clamp(0, height+1)
        columns = (columns+shift_x+1).clamp(0, width+1)
        # Outside-image coordinates map to padded zero, the midpoint of [-1,1].
        padded = F.pad(images, (1, 1, 1, 1)).permute(0, 2, 3, 1)
        return padded[batches, rows, columns].permute(0, 3, 1, 2).contiguous()

    def _cutout(self, images):
        n, _, height, width = images.shape
        cut_height = int(height*self.cutout_ratio+0.5)
        cut_width = int(width*self.cutout_ratio+0.5)
        center_y = torch.randint(0, height+(1-cut_height % 2), (n, 1, 1),
                                 device=images.device, generator=self.rng)
        center_x = torch.randint(0, width+(1-cut_width % 2), (n, 1, 1),
                                 device=images.device, generator=self.rng)
        rows = torch.arange(height, device=images.device)[None, :, None]
        columns = torch.arange(width, device=images.device)[None, None, :]
        top, left = center_y-cut_height//2, center_x-cut_width//2
        inside = (rows >= top) & (rows < top+cut_height) \
            & (columns >= left) & (columns < left+cut_width)
        return images*(~inside).to(images.dtype).unsqueeze(1)


class _AugmentedDiscriminator(nn.Module):
    """Ephemeral step adapter; checkpoint the original D, never this wrapper."""
    def __init__(self, discriminator, augmenter, stages, trace):
        super().__init__()
        self.backbone = discriminator
        self.augmenter = augmenter
        self.stages = stages
        self.trace = trace
        self.calls = 0
        self.training = discriminator.training

    def forward(self, images):
        if self.calls >= len(self.stages):
            raise RuntimeError("Unexpected additional D forward in the optional augmentation step.")
        if self.trace is not None:
            self.trace.append({"stage": self.stages[self.calls], "policy": ",".join(self.augmenter.policy)})
        self.calls += 1
        return self.backbone(self.augmenter(images))


def adversarial_step_diffaugment(generator, discriminator, real, g_optimizer, d_optimizer,
                                 rng=None, *, augmenter=None, trace=None, **step_options):
    """Optional intervention using the unchanged historical-step implementation.

    All D branches receive the same policy distribution; their draws are
    independent. Detached fake images remain detached in the D update, while
    fake-G augmentation retains the derivative to G. Empty policy calls the
    original step directly, so default architecture/math/RNG remain exact.

    The initial optional arm isolates DiffAugment from real-only flips and R1.
    Combining R1 requires a separately specified derivative convention (original
    versus augmented coordinates) and is intentionally rejected here.
    """
    from .model import adversarial_step
    if augmenter is None or not augmenter.policy:
        return adversarial_step(generator, discriminator, real, g_optimizer, d_optimizer,
                                rng, **step_options)
    if step_options.get("r1_gamma", 0) != 0:
        raise ValueError("This isolated DiffAugment arm does not combine an unspecified R1 convention.")
    if step_options.get("horizontal_flip", False):
        raise ValueError("Isolate DiffAugment from the real-only horizontal-flip arm.")
    update_d = step_options.get("update_discriminator", True)
    stages = ("real_d", "fake_d", "fake_g") if update_d else ("fake_g",)
    adapted = _AugmentedDiscriminator(discriminator, augmenter, stages, trace)
    result = adversarial_step(generator, adapted, real, g_optimizer, d_optimizer, rng, **step_options)
    if adapted.calls != len(stages):
        raise RuntimeError("An expected augmented discriminator branch did not execute.")
    return result
