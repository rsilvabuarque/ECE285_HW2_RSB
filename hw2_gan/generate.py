"""Generate fresh paintings from a selected checkpoint and save their latents."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from .utils import require_compute, resolve_device, load_generator, save_grid, save_json, file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--num-samples",type=int,default=64)
    parser.add_argument("--seed",type=int,default=45)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--weights",choices=("raw","ema"),default="raw")
    parser.add_argument("--expected-generator-state-sha256")
    args=parser.parse_args()
    require_compute();torch.set_num_threads(2)
    if args.output_dir.exists(): raise FileExistsError("Use a new output directory.")
    if args.num_samples<1 or args.batch_size<1: raise ValueError("Counts must be positive.")
    device=resolve_device(args.device)
    G,state=load_generator(args.checkpoint,device,weights=args.weights)
    from .evaluate import _state_hash
    from .evaluate_protocols import _inference_context, array_sha256, EVALUATION_BACKEND
    generator_hash=_state_hash(G)
    if args.expected_generator_state_sha256 and generator_hash!=args.expected_generator_state_sha256:
        raise ValueError("Loaded generator tensors differ from the frozen selection.")
    rng=torch.Generator().manual_seed(args.seed)
    latents=torch.randn(args.num_samples,G.latent_dim,generator=rng)
    with _inference_context(device):
        images=torch.cat([G(z.to(device)).cpu() for z in latents.split(args.batch_size)])
    if not torch.isfinite(images).all(): raise FloatingPointError("Nonfinite image pixels.")
    args.output_dir.mkdir(parents=True)
    pixels=((images.clamp(-1,1)+1)*127.5).permute(0,2,3,1).numpy().round().astype(np.uint8)
    for index,array in enumerate(pixels): Image.fromarray(array).save(args.output_dir/f"sample_{index:05d}.png")
    save_grid(images[:64],args.output_dir/"samples_grid.png")
    save_grid(images[:16],args.output_dir/"samples_first16.png",columns=4)
    save_grid(images[:8],args.output_dir/"prior_samples_report.png",columns=4)
    torch.save(latents,args.output_dir/"latents.pt")
    save_json(args.output_dir/"generation.json",{
        "checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":file_sha256(args.checkpoint),
        "checkpoint_epoch":state["epoch"],"prior_seed":args.seed,"num_samples":args.num_samples,
        "generator_variant":args.weights,"generator_state_sha256":generator_hash,
        "training_seed":state["config"].get("seed"),
        "latent_file_sha256":file_sha256(args.output_dir/"latents.pt"),
        "latent_array_sha256":array_sha256(latents.numpy()),
        "backend_flags_during_generation":EVALUATION_BACKEND,
        "native_image_size":state["config"].get("image_size",64),
        "method":"Independent standard-normal latent draws; generator only; no input painting.",
        "display":"Fixed [-1,1] to [0,255] rounding; no per-image normalization.",
        "latent_dim":G.latent_dim,"batch_size":args.batch_size,"device":str(device)})
    print("Saved",len(images),"paintings to",args.output_dir)


if __name__=="__main__": main()
