"""Tiny native128 shape/range checks; widths2/4, CPU inference only."""

import torch
from torch import nn

from hw2_gan.model128 import Generator128, Discriminator128, parameter_counts


def observed_stages(network, inputs, convolution):
    shapes = [tuple(inputs.shape)]
    handles = [layer.register_forward_hook(lambda layer, arguments, output: shapes.append(tuple(output.shape)))
               for layer in network.modules() if isinstance(layer, convolution)]
    try:
        output = network(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return output, shapes


def main():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    with torch.inference_mode():
        for width in (2, 4):
            generator = Generator128(8, width).eval()
            discriminator = Discriminator128(width).eval()
            latent = torch.randn(1, 8, 1, 1)
            images, g_shapes = observed_stages(generator, latent, nn.ConvTranspose2d)
            logits, d_shapes = observed_stages(discriminator, images, nn.Conv2d)
            assert g_shapes == generator.stage_shapes()
            assert d_shapes == discriminator.stage_shapes()
            assert images.shape == (1, 3, 128, 128) and logits.shape == (1,)
            assert torch.isfinite(images).all() and images.min() >= -1 and images.max() <= 1
            assert torch.isfinite(logits).all()
            counts = parameter_counts(8, width, width)
            assert counts["generator"] == sum(parameter.numel() for parameter in generator.parameters())
            assert counts["discriminator"] == sum(parameter.numel() for parameter in discriminator.parameters())
            try:
                discriminator(torch.zeros(1, 3, 64, 64))
            except ValueError:
                pass
            else:
                raise AssertionError("RGB64 images accepted by native128 D.")
        no_bn = Discriminator128(2, batch_norm=False)
        assert not any(isinstance(layer, nn.BatchNorm2d) for layer in no_bn.modules())
        assert sum(parameter.numel() for parameter in no_bn.parameters()) == parameter_counts(8, 2, 2, d_batch_norm=False)["discriminator"]
    assert parameter_counts() == {"generator": 12786560, "discriminator": 11164416}
    assert parameter_counts(g_base_channels=32, d_base_channels=32) == {"generator": 3608000, "discriminator": 2796928}
    print("PASS native G128/D128 stage tensors, shapes/range/logits, wrong-resolution rejection, exact count formulas.")
    print("Only widths2/4 CPU inference executed; no dataset processing, GPU work or training.")


if __name__ == "__main__":
    main()
