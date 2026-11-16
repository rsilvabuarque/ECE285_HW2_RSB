"""Tutorial-compatible probability-feature FID/IS for RGB painting generators.

Run inside a compute allocation. Validation only reads its declared split;
real features are cached by exact input indices and evaluation protocol.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import CachedPaintings


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def _real_cache_lock(path):
    """Serialize first creation; process exit releases the advisory lock."""
    with path.with_suffix(".lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_real_cache(path, identity, count):
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["metadata"].item())) != identity:
            raise ValueError("Real-feature cache metadata mismatch.")
        real = torch.from_numpy(saved["probabilities"].copy())
    if real.shape != (count, 1000) or not torch.isfinite(real).all():
        raise ValueError("Real-feature cache shape/content mismatch.")
    return real


def _state_hash(G):
    result = hashlib.sha256()
    for name, value in sorted(G.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def preprocess_inception(images):
    """Map RGB NCHW [-1,1] into the pretrained classifier's normalized RGB299."""
    if images.ndim != 4 or images.shape[1] != 3 or not len(images):
        raise ValueError("Expected a nonempty RGB NCHW batch.")
    if not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
        raise ValueError("Expected finite RGB pixels in [-1,1].")
    rgb = F.interpolate((images.float() + 1) / 2, (299, 299), mode="bilinear",
                        align_corners=False, antialias=True)
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


class InceptionProbabilities(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.network = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                    transform_input=True).eval()
        self.network.requires_grad_(False)

    def forward(self, images):
        return self.network(preprocess_inception(images)).softmax(dim=1)


def compute_feature_metrics(real, generated):
    """Aggregate IS and FID on cached 1000-probability features, using Ignite."""
    from ignite.metrics import FID, InceptionScore
    if real.ndim != 2 or generated.shape != real.shape or len(real) < 2:
        raise ValueError("Need equal real/generated probability matrices with N >= 2.")
    for probabilities in (real, generated):
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Invalid classifier probabilities.")
        if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]),
                              rtol=1e-5, atol=1e-5):
            raise ValueError("Probability rows must sum to one.")
    real, generated = real.cpu().double(), generated.cpu().double()
    fid = FID(num_features=real.shape[1], feature_extractor=nn.Identity())
    fid.update((generated, real))
    scores = []
    for features in (generated, real):
        metric = InceptionScore(num_features=real.shape[1], feature_extractor=nn.Identity())
        metric.update(features)
        scores.append(float(metric.compute()))
    raw = float(fid.compute())
    if not all(math.isfinite(value) for value in (raw, *scores)) or raw < -1e-6:
        raise FloatingPointError(f"Invalid metric result: FID={raw}, IS={scores}.")
    return {"fid": max(raw, 0.0), "fid_raw_before_roundoff_clamp": raw,
            "inception_score": scores[0], "real_inception_score_reference": scores[1]}


def evaluate_generator(G, cache_dir, split, num_samples, seed, device, output_dir,
                       batch_size=64, save_probabilities=False, cache_mode="readwrite"):
    """Evaluate without advancing training RNG or changing generator BN/mode.

    num_samples=0 uses the whole split; larger requests are capped to its size
    and both requested/actual counts are recorded. Prior draws use their own
    CPU generator, independent of real-reference selection and batch chunking.
    """
    if cache_mode not in ("readwrite", "readonly"):
        raise ValueError("cache_mode must be readwrite or readonly.")
    if split not in ("val", "test") or num_samples < 0 or batch_size < 1:
        raise ValueError("Use val/test, nonnegative num_samples, and positive batch_size.")
    if os.environ.get("NERSC_HOST") == "perlmutter" and (
            not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
        raise RuntimeError("Evaluate inside a compute allocation.")
    cache_dir, output_dir, device = Path(cache_dir), Path(output_dir), torch.device(device)
    if (output_dir / "metrics.json").exists():
        raise FileExistsError("Choose a new evaluation directory to preserve existing metrics.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CachedPaintings(cache_dir, split)
    count = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))
    if count < 2:
        raise ValueError("At least two reference and generated images are required.")
    selection_rng = torch.Generator().manual_seed(seed + 100000)
    local_indices = torch.randperm(len(dataset), generator=selection_rng)[:count].tolist()
    global_indices = np.asarray(dataset.indices)[local_indices].tolist()
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "pytorch-ignite", "numpy", "scipy")}
    protocol = {"features": "1000 ImageNet class probabilities",
                "weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
                "input": "RGB [-1,1] mapped to [0,1]",
                "resize": "299x299 bilinear, align_corners=False, antialias=True",
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225], "transform_input": True,
                "inception_score": "aggregate, without split averaging",
                "feature_device": str(device), "precision": "float32, autocast disabled",
                "versions": versions}
    real_key = {"fingerprint": dataset.manifest["fingerprint"], "split": split,
                "global_indices": global_indices, "protocol_hash": _digest(protocol)}
    feature_dir = cache_dir / "inception_probabilities"
    feature_dir.mkdir(exist_ok=True)
    real_path = feature_dir / (_digest(real_key) + ".npz")
    real = None
    if real_path.exists():
        real = _read_real_cache(real_path, real_key, count)
    elif cache_mode == "readonly":
        raise FileNotFoundError("Precompute the immutable real cache before parallel evaluation.")
    real_cache_hit = real is not None
    # Preserve every module's original mode, including any deliberately frozen BN.
    modes = [(module, module.training) for module in G.modules()]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        G.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), \
                torch.autocast(device_type=device.type, enabled=False):
            extractor = InceptionProbabilities().to(device).eval()
            if real is None:
                with _real_cache_lock(real_path):
                    if real_path.exists():
                        real = _read_real_cache(real_path, real_key, count)
                        real_cache_hit = True
                    else:
                        loader = DataLoader(Subset(dataset, local_indices), batch_size=batch_size,
                                            num_workers=0, shuffle=False,
                                            generator=torch.Generator().manual_seed(seed + 200000))
                        real = torch.cat([extractor(images.to(device)).cpu() for images in loader])
                        fd, name = tempfile.mkstemp(prefix=real_path.name+".", suffix=".tmp", dir=feature_dir)
                        temporary = Path(name)
                        try:
                            with os.fdopen(fd, "wb") as handle:
                                np.savez_compressed(handle, probabilities=real.numpy(),
                                                    metadata=np.array(json.dumps(real_key, sort_keys=True)))
                            temporary.replace(real_path)
                        finally:
                            temporary.unlink(missing_ok=True)
            prior_rng = torch.Generator().manual_seed(seed)
            latents = torch.randn(count, G.latent_dim, generator=prior_rng)
            generated = torch.cat([extractor(G(z.to(device))).cpu()
                                   for z in latents.split(batch_size)])
            del extractor
        values = compute_feature_metrics(real, generated)
    finally:
        for module, training in modes:
            module.training = training
    if save_probabilities:
        np.savez_compressed(output_dir / "probabilities.npz", real=real.numpy(),
                            generated=generated.numpy(), global_indices=global_indices)
    result = {**values, "reference_split": split, "num_samples_requested": num_samples,
              "num_real_images": count, "num_generated_images": count, "prior_seed": seed,
              "selection_seed": seed + 100000, "batch_size": batch_size, "device": str(device),
              "cache_fingerprint": real_key["fingerprint"],
              "generator_state_sha256": _state_hash(G), "reference_indices": global_indices,
              "reference_files": [dataset.manifest["files"][i] for i in global_indices],
              "real_feature_cache": str(real_path.resolve()), "real_cache_hit": real_cache_hit,
              "protocol": protocol, "protocol_hash": _digest(protocol),
              "fid_roundoff_guard": "-1e-6; provisional numerical check, not a quality target",
              "limitations": ["Probability-feature FID differs from 2048-pool-feature FID.",
                               "ImageNet scores do not establish artistic quality or originality.",
                               "Finite-sample scores are biased and sample-sensitive."]}
    _save_json(output_dir / "metrics.json", result)
    print(f"{split}: {count} real/generated; FID={values['fid']:.6f}, "
          f"IS={values['inception_score']:.6f}; real cache hit={real_cache_hit}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--cache-mode", choices=("readwrite", "readonly"), default="readwrite")
    args = parser.parse_args()
    if os.environ.get("NERSC_HOST") == "perlmutter" and (
            not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
        raise RuntimeError("Evaluate inside a compute allocation.")
    from .utils import load_generator
    device = torch.device(args.device)
    G, state = load_generator(args.checkpoint, device)
    result = evaluate_generator(G, args.cache_dir, args.split, args.num_samples, args.seed,
                                device, args.output_dir, args.batch_size, args.save_probabilities,
                                args.cache_mode)
    result.update(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=_file_hash(args.checkpoint), checkpoint_epoch=state["epoch"])
    _save_json(args.output_dir / "metrics.json", result)


if __name__ == "__main__":
    main()
