"""Isolated, resumable RGB64 GAN campaign; the historical train.py is retained.

Fresh matched arms should explicitly use --data-order epoch-v1 --ema-decay .999.
The legacy data-order default exists only to reproduce the original baseline
loader protocol and permit its separate continuation. Checkpoints are written
only at completed epoch boundaries; interrupted partial epochs are replayed.
"""

import argparse
from contextlib import contextmanager
from copy import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from .data import CachedPaintings
from .model import (Generator, Discriminator, initialize_weights, adversarial_step,
                    make_ema_generator, module_mode)
from .utils import (require_compute, seed_everything, resolve_device, save_json,
                    save_grid, save_checkpoint, record_provenance, file_sha256)


INFERENCE_EPOCHS = (2, 25, 50, 75, 100, 150, 200, 250, 300)
PREVIEW_EPOCHS = (1, 25, 50, 75, 100, 150, 200, 250, 300)
DYNAMICS_KEYS = (
    "image_size", "diffaugment_policy", "latent_dim", "base_channels", "g_base_channels", "d_base_channels", "batch_size",
    "learning_rate", "g_learning_rate", "d_learning_rate", "beta1", "beta2", "seed",
    "num_workers", "data_order", "d_update_interval", "r1_gamma", "r1_interval",
    "r1_batchnorm", "d_spectral_norm", "d_no_batchnorm", "d_batch_norm",
    "d_eval_for_g", "horizontal_flip", "ema_decay", "fixed_seed", "diagnostic_seed",
    "diagnostic_samples", "integration_smoke", "train_limit", "max_batches",
    "save_every", "save_epochs", "evaluation_batch_size", "legacy_reference",
    "pooled_reference", "defer_pooled_score",
    "learning_rate_schedule",
)


class RecordingParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        result = super().parse_args(argv, namespace)
        supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        result._explicit = {action.dest for action in self._actions
                            if supplied.intersection(action.option_strings)}
        return result


def epoch_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if not value:
        return []
    epochs = [int(item) for item in value.split(",")]
    if any(epoch < 1 for epoch in epochs) or len(epochs) != len(set(epochs)):
        raise argparse.ArgumentTypeError("Epoch lists require distinct positive integers.")
    return sorted(epochs)


def build_parser():
    parser = RecordingParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--diffaugment-policy", default="",
                        help="Optional second-wave policy: color,translation,cutout or a subset.")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--g-base-channels", type=int)
    parser.add_argument("--d-base-channels", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--g-learning-rate", type=float)
    parser.add_argument("--d-learning-rate", type=float)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-order", choices=("legacy", "epoch-v1"), default="legacy")
    parser.add_argument("--d-update-interval", type=int, default=1)
    parser.add_argument("--r1-gamma", type=float, default=0.0)
    parser.add_argument("--r1-interval", type=int, default=1)
    parser.add_argument("--r1-batchnorm", choices=("eval", "error"), default="eval")
    parser.add_argument("--d-spectral-norm", action="store_true")
    parser.add_argument("--d-no-batchnorm", action="store_true")
    parser.add_argument("--d-eval-for-g", action="store_true")
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--pooled-epochs", type=epoch_list, default=[100, 150, 200, 250, 300])
    parser.add_argument("--diagnostic-epochs", type=epoch_list,
                        default=[25, 50, 100, 150, 200, 250, 300])
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--save-epochs", type=epoch_list, default=list(INFERENCE_EPOCHS))
    parser.add_argument("--fixed-seed", type=int, default=43)
    parser.add_argument("--diagnostic-seed", type=int, default=4343)
    parser.add_argument("--diagnostic-samples", type=int, default=256)
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--legacy-reference", default="validation_legacy1024")
    parser.add_argument("--pooled-reference", default="validation_pool4000_a")
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--defer-pooled-score", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fork-linear-decay", action="store_true",
                        help="Fork only a full constant-rate epoch200 state into an empty directory; rates decay at epoch starts201..300.")
    parser.add_argument("--integration-smoke", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser


class EpochRandomSampler(Sampler):
    """Epoch-addressable permutation; independent of worker lifetime and RNG."""
    def __init__(self, dataset, seed, epoch=1):
        self.dataset, self.seed, self.epoch = dataset, int(seed), int(epoch)

    def __iter__(self):
        # Versioned convention: epoch1 uses seed+1000003, modulo 2**63.
        rng = torch.Generator().manual_seed((self.seed+1000003*self.epoch) % (2**63))
        return iter(torch.randperm(len(self.dataset), generator=rng).tolist())

    def __len__(self):
        return len(self.dataset)


def normalized_config(values):
    config = dict(values)
    config.setdefault("learning_rate_schedule", "constant")
    config.setdefault("image_size", 64)
    config.setdefault("diffaugment_policy", "")
    if config["diffaugment_policy"]:
        from .diffaugment import parse_policy
        config["diffaugment_policy"] = ",".join(parse_policy(config["diffaugment_policy"]))
    config["g_base_channels"] = config.get("g_base_channels") or config.get("base_channels", 64)
    config["d_base_channels"] = config.get("d_base_channels") or config.get("base_channels", 64)
    config["g_learning_rate"] = config.get("g_learning_rate") or config.get("learning_rate", 2e-4)
    config["d_learning_rate"] = config.get("d_learning_rate") or config.get("learning_rate", 2e-4)
    config["d_batch_norm"] = not (config.get("d_spectral_norm", False) or config.get("d_no_batchnorm", False))
    return config


def _plain_namespace(args):
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith("_")}


def resolve_config(args, state):
    requested = normalized_config(_plain_namespace(args))
    requested["learning_rate_schedule"] = "linear-201-300" if args.fork_linear_decay else "constant"
    if args.fork_linear_decay and (not state or state.get("epoch") != 200
            or state.get("config", {}).get("learning_rate_schedule", "constant") != "constant"):
        raise ValueError("Linear decay forks require a full constant-rate epoch200 parent.")
    if not state:
        return requested
    # Backfill the historical config with old dynamics defaults, not new campaign
    # eval/save cadence; unspecified legacy additions were inactive historically.
    defaults = _plain_namespace(build_parser().parse_args([
        "--cache-dir", str(args.cache_dir), "--output-dir", str(args.output_dir)]))
    saved = normalized_config(defaults | state["config"])
    explicit = getattr(args, "_explicit", set())
    for key in ("evaluation_plan", "feature_cache", "eval_every", "pooled_epochs", "diagnostic_epochs"):
        if key not in explicit and key in state["config"]:
            requested[key] = state["config"][key]
    allow_ema_initialization = "generator_ema_state" not in state and "ema_decay" in explicit \
        and requested["ema_decay"] > 0 and not state.get("campaign_schema_version")
    for key in DYNAMICS_KEYS:
        if key == "learning_rate_schedule" and args.fork_linear_decay:
            continue
        if key == "ema_decay" and allow_ema_initialization:
            continue
        # g/d derived aliases are checked after inheritance of legacy aliases.
        if key in explicit and requested[key] != saved[key]:
            raise ValueError(f"Resume cannot change {key}: saved={saved[key]!r}, requested={requested[key]!r}.")
        requested[key] = saved[key]
    # Explicit legacy aliases must not silently alter saved independent values.
    if "base_channels" in explicit and requested["base_channels"] != saved["base_channels"]:
        raise ValueError("Resume cannot change the legacy width alias.")
    if "learning_rate" in explicit and requested["learning_rate"] != saved["learning_rate"]:
        raise ValueError("Resume cannot change the legacy learning-rate alias.")
    if allow_ema_initialization:
        requested["ema_decay"] = args.ema_decay
    if "evaluation_plan_sha256" in state.get("config", {}):
        expected = state["config"]["evaluation_plan_sha256"]
        actual = file_sha256(requested["evaluation_plan"]) if requested["evaluation_plan"] else None
        if actual != expected:
            raise ValueError("Resume evaluation plan differs from the saved immutable plan.")
    return requested


def validate_config(config):
    if config["learning_rate_schedule"] == "linear-201-300" and not 201 <= config["epochs"] <= 300:
        raise ValueError("The declared linear schedule is defined only through epoch300.")
    if config["image_size"] not in (64, 128):
        raise ValueError("Only native RGB64 and the separately marked RGB128 demo are supported.")
    if config["diffaugment_policy"] and (config["r1_gamma"] or config["horizontal_flip"]):
        raise ValueError("The isolated DiffAugment arm excludes R1 and real-only flips.")
    positive_ints = ["epochs", "latent_dim", "base_channels", "g_base_channels", "d_base_channels",
                     "d_update_interval", "r1_interval", "save_every", "diagnostic_samples",
                     "evaluation_batch_size"]
    if any(config[key] < 1 for key in positive_ints) or config["batch_size"] < 2 \
            or config["num_workers"] < 0 or config["eval_every"] < 0:
        raise ValueError("Invalid count, batch size, worker count, or interval.")
    if not 0 <= config["seed"] < 2**63:
        raise ValueError("Seed must be in [0,2**63).")
    if any(not math.isfinite(config[key]) or config[key] <= 0
           for key in ["learning_rate", "g_learning_rate", "d_learning_rate"]):
        raise ValueError("Learning rates must be finite and positive.")
    if not all(math.isfinite(config[key]) and 0 <= config[key] < 1
               for key in ["beta1", "beta2", "ema_decay"]):
        raise ValueError("Adam beta values and EMA decay must lie in [0,1).")
    if not math.isfinite(config["r1_gamma"]) or config["r1_gamma"] < 0:
        raise ValueError("R1 gamma must be finite and nonnegative.")
    if any(config.get(key) is not None for key in ("train_limit", "max_batches")) \
            and not config["integration_smoke"]:
        raise ValueError("Data/batch truncation requires explicit --integration-smoke.")
    if any(config.get(key) is not None and config[key] < 1 for key in ("train_limit", "max_batches")):
        raise ValueError("Smoke truncation counts must be positive.")
    if bool(config["evaluation_plan"]) != bool(config["feature_cache"]):
        raise ValueError("Provide both --evaluation-plan and --feature-cache.")
    if not config["evaluation_plan"] and not config["integration_smoke"]:
        raise ValueError("Production runs require an immutable evaluation plan and precomputed feature cache.")
    if config["r1_gamma"] and config["d_batch_norm"] and config["r1_batchnorm"] == "error":
        raise ValueError("Primary R1 requires --d-no-batchnorm; evalBN is a separate named ablation.")


def _numpy_rng_state():
    kind, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"kind": kind, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached_gaussian": cached_gaussian}


def capture_global_rng():
    return {"torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(), "numpy_rng_state": _numpy_rng_state()}


def restore_global_rng(state):
    torch.set_rng_state(state["torch_rng_state"])
    if state.get("cuda_rng_states") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        value = state["numpy_rng_state"]
        np.random.set_state((value["kind"], np.asarray(value["keys"], dtype=np.uint32),
                             value["position"], value["has_gauss"], value["cached_gaussian"]))


@contextmanager
def preserved_rng():
    before = capture_global_rng()
    try:
        yield
    finally:
        restore_global_rng(before)


@contextmanager
def run_lock(output):
    lock = output/".campaign.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o660)
    token = uuid.uuid4().hex
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"token": token, "pid": os.getpid(), "host": socket.gethostname(),
                   "job_id": os.environ.get("SLURM_JOB_ID")}, handle)
    try:
        yield
    finally:
        if lock.exists() and json.loads(lock.read_text()).get("token") == token:
            lock.unlink()


def write_history(path, rows):
    # Old baseline rows retain their original keys and values; new columns are
    # added to a stable union, with empty CSV cells for previously absent values.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def shared_split(dataset, split):
    view = copy(dataset)
    view.indices = np.asarray(dataset.manifest["splits"][split], dtype=np.int64)
    return view


def diagnostics(generator, discriminator, dataset, device, config, output, epoch, shadow=None):
    """Fixed real references, fresh fake prior, same eval-mode D, no augmentation."""
    count = config["diagnostic_samples"]
    selection_rng = torch.Generator().manual_seed(config["diagnostic_seed"])
    train_indices = torch.randperm(len(dataset), generator=selection_rng)[:count].tolist()
    validation = shared_split(dataset, "val")
    val_indices = torch.randperm(len(validation), generator=selection_rng)[:count].tolist()
    if not train_indices or not val_indices:
        raise ValueError("Diagnostics need nonempty train and validation splits.")
    prior_rng = torch.Generator().manual_seed(config["diagnostic_seed"]+2)
    latents = torch.randn(count, generator.latent_dim, generator=prior_rng)
    fixed = torch.randn(64, generator.latent_dim,
                        generator=torch.Generator().manual_seed(config["fixed_seed"]))
    buffers = {id(module): {name: value.detach().clone() for name, value in module.named_buffers()}
               for module in [generator, discriminator]+([shadow] if shadow is not None else [])}
    result = {"epoch": epoch, "mode": "eval for all D/G forwards; no augmentation",
              "train_global_indices": dataset.indices[train_indices].tolist(),
              "validation_global_indices": validation.indices[val_indices].tolist(),
              "prior_seed": config["diagnostic_seed"]+2,
              "fixed_prior_seed": config["fixed_seed"], "variants": {}}

    def real_stats(view, indices):
        logits = []
        for start in range(0, len(indices), config["evaluation_batch_size"]):
            batch = torch.stack([view[index] for index in indices[start:start+config["evaluation_batch_size"]]])
            logits.append(discriminator(batch.to(device)).cpu())
        return _discrimination_stats(torch.cat(logits))

    with preserved_rng(), module_mode(generator, False), module_mode(discriminator, False), torch.inference_mode():
        result["train_real"] = real_stats(dataset, train_indices)
        result["validation_real"] = real_stats(validation, val_indices)
        for name, current in [("raw", generator)]+([("ema", shadow)] if shadow is not None else []):
            with module_mode(current, False):
                logits = torch.cat([discriminator(current(batch.to(device))).cpu()
                                    for batch in latents.split(config["evaluation_batch_size"])])
                images = torch.cat([current(batch.to(device)).cpu()
                                    for batch in fixed.split(config["evaluation_batch_size"])])
            save_grid(images, output/"previews"/f"epoch_{epoch:03d}_{name}.png")
            result["variants"][name] = {
                "fresh_fake": _discrimination_stats(logits),
                "fixed_sample_count": len(images),
                "pixel_std_across_samples_mean": images.std(dim=0, unbiased=False).mean().item(),
                "pixel_variance_across_samples_mean": images.var(dim=0, unbiased=False).mean().item(),
                "saturated_pixel_fraction_abs_gt_0_95": (images.abs() > 0.95).float().mean().item(),
                "diagnostic_caveat": "Pixel diversity/saturation are descriptive, not perceptual-quality selection criteria.",
            }
    for current in [generator, discriminator]+([shadow] if shadow is not None else []):
        assert all(torch.equal(value, buffers[id(current)][name]) for name, value in current.named_buffers()), \
            "Diagnostics mutated model buffers."
    save_json(output/"diagnostics"/f"epoch_{epoch:03d}.json", result)
    return result


def _discrimination_stats(logits):
    probabilities = logits.sigmoid()
    return {"count": len(logits), "logit_mean": logits.mean().item(),
            "logit_std": logits.std(unbiased=False).item(),
            "probability_mean": probabilities.mean().item(),
            "probability_std": probabilities.std(unbiased=False).item(),
            "fraction_probability_above_half": (probabilities > 0.5).float().mean().item()}


def preview(generator, shadow, device, config, output, epoch):
    fixed = torch.randn(64, generator.latent_dim,
                        generator=torch.Generator().manual_seed(config["fixed_seed"]))
    with preserved_rng(), torch.inference_mode():
        for name, current in [("raw", generator)]+([("ema", shadow)] if shadow is not None else []):
            with module_mode(current, False):
                images = torch.cat([current(batch.to(device)).cpu()
                                    for batch in fixed.split(config["evaluation_batch_size"])])
                save_grid(images, output/"previews"/f"epoch_{epoch:03d}_{name}.png")


def evaluate_variant(generator, variant, reference, protocols, config, output, epoch, pooled=False):
    from .evaluate_protocols import extract_generator_features, score_feature_artifact
    # Sparse restart checkpoints can replay an epoch whose first evaluation was
    # completed before interruption. Keep each attempt as immutable evidence.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output/"validation"/f"epoch_{epoch:03d}"/variant/reference/f"attempt_{stamp}_{uuid.uuid4().hex[:8]}"
    artifact = extract_generator_features(
        generator, config["cache_dir"], config["evaluation_plan"], reference,
        config["feature_cache"], destination, protocol_ids=protocols,
        device=config["device"], batch_size=config["evaluation_batch_size"],
        generator_variant=variant,
        checkpoint_metadata={"epoch": epoch, "run_directory": str(output.resolve()),
                             "selection_scope": "validation candidate; not frozen final selection"})
    if pooled and config["defer_pooled_score"]:
        return {"feature_artifact": str(artifact), "score_status": "deferred", "metrics": {}}
    # CPU covariance/eigendecomposition remains inside this allocated process.
    result = score_feature_artifact(artifact, config["evaluation_plan"], config["feature_cache"],
                                    destination/"metrics.json")
    return result | {"feature_artifact": str(artifact), "score_status": "completed"}


def construct_models(config, device):
    """Isolate fresh-arm initialization from SN constructor random draws.

    The legacy path retains its original order. Fresh epoch-v1 experiments share
    private CPU streams; their matched C0F control uses the same convention.
    CPU default_generator seeding does not alter any CUDA random stream.
    """
    # Only select classes here; preserve the existing component-v1 private streams.
    if config.get("image_size", 64) == 64:
        generator_class, discriminator_class = Generator, Discriminator
    else:
        from .model128 import Generator128, Discriminator128
        generator_class, discriminator_class = Generator128, Discriminator128

    def generator():
        return generator_class(config["latent_dim"], config["g_base_channels"])

    def discriminator():
        return discriminator_class(config["d_base_channels"], spectral_norm=config["d_spectral_norm"],
                             batch_norm=config["d_batch_norm"])

    if config["data_order"] == "legacy":
        g, d = generator().to(device), discriminator().to(device)
        g.apply(initialize_weights); d.apply(initialize_weights)
        return g, d
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"]+2001)
        g = generator()
        torch.random.default_generator.manual_seed(config["seed"]+1001)
        g.apply(initialize_weights)
        torch.random.default_generator.manual_seed(config["seed"]+2002)
        d = discriminator()
        torch.random.default_generator.manual_seed(config["seed"]+1002)
        d.apply(initialize_weights)
    return g.to(device), d.to(device)


def apply_epoch_learning_rates(config, epoch, g_optimizer, d_optimizer):
    """Apply the declared base rates after restoring Adam moments.

    The paired intervention starts with f(201)=1 and reaches f(300)=0.5.
    Adam first/second moments, counters and all random streams are unchanged.
    """
    factor = 1.0
    if config["learning_rate_schedule"] == "linear-201-300":
        factor = max(0.5, min(1.0, 1.0-0.5*(epoch-201)/99))
    for optimizer, rate in ((g_optimizer, config["g_learning_rate"]),
                            (d_optimizer, config["d_learning_rate"])):
        for group in optimizer.param_groups:
            group["lr"] = factor*rate
    return factor


def _source_record(output, config, resume_path=None, resume_hash=None):
    record_provenance(output, config)
    repository = Path(__file__).resolve().parent.parent
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(repository), *arguments],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle_path = repository/"SOURCE_MANIFEST.json"
    bundle = json.loads(bundle_path.read_text()) if bundle_path.is_file() else None
    if bundle is not None:
        for name, expected in bundle["files"].items():
            if file_sha256(repository/name) != expected:
                raise ValueError(f"Executed source bundle changed: {name}")
    save_json(output/"provenance"/f"campaign_{stamp}.json", {
        "source_commit": git("rev-parse", "HEAD") if bundle is None else None,
        "source_status": git("status", "--porcelain") if bundle is None else None,
        "source_bundle_manifest_sha256": file_sha256(bundle_path) if bundle else None,
        "source_bundle_base_commit": bundle.get("base_commit") if bundle else None,
        "source_bundle_code_commit": bundle.get("code_commit") if bundle else None,
        "resume_path": str(resume_path) if resume_path else None, "resume_sha256": resume_hash,
        "backend_flags": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                          "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "gpu_visible_properties": [str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())],
        "gpu_visible_uuids": [str(getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable")) for i in range(torch.cuda.device_count())]})


def train(args):
    require_compute()
    torch.set_num_threads(2)
    state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = resolve_config(args, state)
    validate_config(config)
    if state:
        needed = {"generator_state", "discriminator_state", "g_optimizer_state", "d_optimizer_state",
                  "data_rng_state", "noise_rng_state", "torch_rng_state", "epoch", "history"}
        if not needed.issubset(state):
            raise ValueError(f"Resume requires full last.pt, not inference weights; missing {sorted(needed-set(state))}.")
        if config["epochs"] <= state["epoch"]:
            raise ValueError("Requested target epochs are already complete.")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        if args.fork_linear_decay:
            raise FileExistsError("A decay fork requires a fresh empty output directory.")
        if not args.resume or args.resume.parent.resolve() != output.resolve():
            raise FileExistsError("Use an empty run directory, or resume its own last.pt.")
        if args.resume.name != "last.pt":
            raise ValueError("Only the run's last.pt may resume a nonempty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        return _train_locked(args, config, state, output)


def _train_locked(args, config, state, output):
    device = resolve_device(config["device"])
    config["device"] = str(device)
    seed_everything(config["seed"])
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if config["image_size"] == 128:
        from .data128 import CachedPaintings128
        dataset = CachedPaintings128(config["cache_dir"], "train")
        config["model_family"] = "native-rgb128-dcgan-demo-v1"
        config["comparison_scope"] = "Separate RGB128 demonstration; outside core RGB64 ranking."
    else:
        dataset = CachedPaintings(config["cache_dir"], "train")
    expected_shape = (config["image_size"], config["image_size"], 3)
    if tuple(dataset.images.shape[1:]) != expected_shape:
        raise ValueError("Training image size differs from the model; RGB64 upsampling is not a native RGB128 cache.")
    fingerprint = dataset.manifest["fingerprint"]
    if state and state["cache_fingerprint"] != fingerprint:
        raise ValueError("Dataset cache fingerprint differs from the saved run.")
    training = Subset(dataset, range(min(config["train_limit"], len(dataset)))) \
        if config["train_limit"] is not None else dataset
    if not len(training):
        raise ValueError("Training split is empty.")
    if len(training) % config["batch_size"] == 1:
        # G's first BN sees spatial4x4, so singleton batches remain valid; no drop.
        pass
    if config["evaluation_plan"]:
        from .evaluate_protocols import load_reference_plan
        plan, _ = load_reference_plan(config["evaluation_plan"], cache_dir=config["cache_dir"],
                                      verify_pixels=True)
        config["processed_cache_identity"] = plan["dataset_identity"]
        if state and state["config"].get("processed_cache_identity") not in (None, plan["dataset_identity"]):
            raise ValueError("Processed pixels or data protocol changed across resume.")
        config["evaluation_plan_sha256"] = file_sha256(config["evaluation_plan"])
    else:
        config["evaluation_plan_sha256"] = None
    config["campaign_schema_version"] = 1
    config["initialization_definition"] = "historical global RNG and construction order" \
        if config["data_order"] == "legacy" else \
        "component-v1 CPU constructor seeds seed+2001/2002; explicit G/D initialization seed+1001/1002; global RNG restored"
    config["data_order_definition"] = "legacy shared worker/sampler RNG" if config["data_order"] == "legacy" \
        else "epoch-v1: randperm seed=(seed+1000003*epoch)%2**63; separate restored worker RNG; workers recreated each epoch"
    config["r1_definition"] = "gamma/2 * mean_N(sum_CHW(input-logit-gradient squared)); lazy multiplier=interval"
    config["r1_bn_interaction"] = "separate fixed-running-stat eval-mode real-logit forward" \
        if config["r1_gamma"] and config["d_batch_norm"] else "no train-mode minibatch coupling in penalty"
    config["ema_buffer_policy"] = "average floating parameters/buffers; copy integer buffers; eval inference"
    data_rng = torch.Generator().manual_seed(config["seed"])
    worker_rng = torch.Generator().manual_seed(config["seed"]+100)
    noise_rng = torch.Generator(device=device).manual_seed(config["seed"]+2)
    augmentation_rng = torch.Generator(device=device).manual_seed(config["seed"]+3)
    diffaugment = None
    step_function = adversarial_step
    if config["diffaugment_policy"]:
        from .diffaugment import DifferentiableAugment, adversarial_step_diffaugment
        diffaugment = DifferentiableAugment(config["diffaugment_policy"], seed=config["seed"]+3, device=device)
        config["diffaugment_definition"] = diffaugment.config
        if state:
            if "diffaugment_state" not in state:
                raise ValueError("Enabled DiffAugment resume requires its own saved RNG/config state.")
            diffaugment.load_state_dict(state["diffaugment_state"])
        step_function = adversarial_step_diffaugment
    generator, discriminator = construct_models(config, device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["g_learning_rate"],
                                   betas=(config["beta1"], config["beta2"]))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["d_learning_rate"],
                                   betas=(config["beta1"], config["beta2"]))
    history, start_epoch, best_val, best_epoch = [], 1, math.inf, None
    global_g_steps, global_d_steps, ema_initialization = 0, 0, None
    if state:
        generator.load_state_dict(state["generator_state"])
        discriminator.load_state_dict(state["discriminator_state"])
        g_optimizer.load_state_dict(state["g_optimizer_state"])
        d_optimizer.load_state_dict(state["d_optimizer_state"])
        data_rng.set_state(state["data_rng_state"])
        noise_rng.set_state(state["noise_rng_state"])
        if "worker_rng_state" in state:
            worker_rng.set_state(state["worker_rng_state"])
        if "augmentation_rng_state" in state:
            augmentation_rng.set_state(state["augmentation_rng_state"])
        history = state["history"]
        start_epoch = state["epoch"]+1
        best_val, best_epoch = state.get("best_val_fid", math.inf), state.get("best_epoch")
        inferred_steps = state["epoch"]*math.ceil(len(training)/config["batch_size"])
        global_g_steps = state.get("global_g_steps", inferred_steps)
        global_d_steps = state.get("global_d_steps", global_g_steps)
        ema_initialization = state.get("ema_initialization")
    shadow = make_ema_generator(generator) if config["ema_decay"] > 0 else None
    if shadow is not None:
        if state and "generator_ema_state" in state:
            shadow.load_state_dict(state["generator_ema_state"])
        elif state:
            ema_initialization = {"source": "raw_generator_at_resume", "epoch": state["epoch"],
                                  "global_g_steps": global_g_steps, "decay": config["ema_decay"]}
            print("EMA initialized at continuation boundary:", ema_initialization, flush=True)
        else:
            ema_initialization = {"source": "raw_generator_at_fresh_initialization", "epoch": 0,
                                  "global_g_steps": 0, "decay": config["ema_decay"]}
    if state:
        restore_global_rng(state)
    if args.resume:
        resume_hash = file_sha256(args.resume)
        save_json(output/"resume_sources"/f"epoch_{state['epoch']:03d}_{resume_hash[:16]}.json", {
            "source": str(args.resume.resolve()), "sha256": resume_hash, "epoch": state["epoch"],
            "original_config": state["config"],
            "global_steps_inferred_from_completed_legacy_epochs": "global_g_steps" not in state,
            "new_augmentation_stream_seed_if_absent": config["seed"]+3,
            "legacy_resume_caveat": "Persistent-worker base-seed draws may differ after restart; continuation is a separate legacy control." \
                if config["data_order"] == "legacy" else None})
        prior_history = output/"history.csv"
        if prior_history.exists():
            with prior_history.open(newline="") as handle:
                prior_rows = list(csv.DictReader(handle))
            latest = max((int(row["epoch"]) for row in prior_rows), default=0)
            if latest > state["epoch"]:
                recovery = output/"recovery"
                recovery.mkdir(exist_ok=True)
                digest = file_sha256(prior_history)
                preserved = recovery/f"history_after_checkpoint_{state['epoch']:03d}_{digest[:16]}.csv"
                if not preserved.exists():
                    preserved.write_bytes(prior_history.read_bytes())
                save_json(recovery/f"history_after_checkpoint_{state['epoch']:03d}_{digest[:16]}.json", {
                    "checkpoint_epoch": state["epoch"], "latest_uncheckpointed_epoch": latest,
                    "preserved_history": str(preserved),
                    "policy": "Replay completed epochs after last.pt; retain original history/evaluation attempts as evidence."})
    else:
        resume_hash = None
    save_json(output/"config.json", config)
    summary_keys = ["dataset", "dataset_version", "fingerprint", "preprocessing", "split_rule",
                    "split_seed", "style_counts", "failures", "limitations"]
    save_json(output/"dataset_summary.json", {key: dataset.manifest.get(key) for key in summary_keys} |
              {"split_counts": {key: len(value) for key, value in dataset.manifest["splits"].items()}})
    _source_record(output, config, args.resume, resume_hash)
    # Source/environment writes must not change the state from which training
    # resumes, even if an installed package import consumes a random stream.
    if state:
        restore_global_rng(state)
    legacy_loader = None
    if config["data_order"] == "legacy":
        legacy_loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
            num_workers=config["num_workers"], drop_last=False, pin_memory=device.type == "cuda",
            generator=data_rng, persistent_workers=config["num_workers"] > 0)
    print("Parameters:", sum(parameter.numel() for parameter in generator.parameters()), "G;",
          sum(parameter.numel() for parameter in discriminator.parameters()), "D", flush=True)
    for epoch in range(start_epoch, config["epochs"]+1):
        lr_factor = apply_epoch_learning_rates(config, epoch, g_optimizer, d_optimizer)
        generator.train(); discriminator.train()
        loader = legacy_loader if legacy_loader is not None else DataLoader(training,
            batch_size=config["batch_size"], sampler=EpochRandomSampler(training, config["seed"], epoch),
            num_workers=config["num_workers"], drop_last=False, pin_memory=device.type == "cuda",
            generator=worker_rng, persistent_workers=False)
        start = time.monotonic()
        sums, weights, epoch_g_steps, epoch_d_steps, seen = {}, {}, 0, 0, 0
        for batch, real in enumerate(loader, 1):
            if config["max_batches"] is not None and batch > config["max_batches"]:
                break
            real = real.to(device, non_blocking=True)
            update_d = global_g_steps % config["d_update_interval"] == 0
            apply_r1 = global_d_steps % config["r1_interval"] == 0
            optional_step = {"augmenter": diffaugment} if diffaugment is not None else {}
            metrics = step_function(generator, discriminator, real, g_optimizer, d_optimizer,
                noise_rng, **optional_step, update_discriminator=update_d, r1_gamma=config["r1_gamma"],
                r1_interval=config["r1_interval"], apply_r1=apply_r1,
                r1_batchnorm=config["r1_batchnorm"], d_eval_for_g=config["d_eval_for_g"],
                ema_generator=shadow, ema_decay=config["ema_decay"],
                horizontal_flip=config["horizontal_flip"], augmentation_rng=augmentation_rng)
            global_g_steps += 1; epoch_g_steps += 1
            global_d_steps += int(update_d); epoch_d_steps += int(update_d)
            for key, value in metrics.items():
                if value is None or key in ("d_updated", "r1_applied"):
                    continue
                # R1 zeros on non-D batches must not dilute its D-update mean.
                if key.startswith("r1_") and not update_d:
                    continue
                sums[key] = sums.get(key, 0.0)+len(real)*value
                weights[key] = weights.get(key, 0)+len(real)
            seen += len(real)
            if batch % 200 == 0:
                print(f"epoch {epoch} batch {batch}/{len(loader)} G={metrics['g_loss']:.4f} D={metrics['d_loss']}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"epoch": epoch, "seconds": time.monotonic()-start,
               "g_learning_rate": g_optimizer.param_groups[0]["lr"],
               "d_learning_rate": d_optimizer.param_groups[0]["lr"], "learning_rate_factor": lr_factor,
               **{key: sums[key]/weights[key] for key in sums},
               "train_images": seen, "g_steps": epoch_g_steps, "d_steps": epoch_d_steps,
               "global_g_steps": global_g_steps, "global_d_steps": global_d_steps,
               "val_fid": None, "val_is": None, "val_ema_fid": None, "val_ema_is": None,
               "val_pool_fid": None, "val_ema_pool_fid": None, "validation_seconds": 0.0}
        if not seen:
            raise RuntimeError("No minibatches were executed.")
        monitor_selected = False
        if config["evaluation_plan"]:
            from .evaluate_protocols import LEGACY, POOLED
            validation_start = time.monotonic()
            variants = [("raw", generator)]+([("ema", shadow)] if shadow is not None else [])
            if config["eval_every"] and (epoch % config["eval_every"] == 0 or epoch == config["epochs"]):
                for variant, current in variants:
                    with preserved_rng():
                        result = evaluate_variant(current, variant, config["legacy_reference"],
                                                  (LEGACY,), config, output, epoch)
                    values = result["metrics"][LEGACY]
                    row[f"val_{variant}_legacy_feature_artifact"] = result["feature_artifact"]
                    if variant == "raw":
                        row.update(val_fid=values["fid"], val_is=values["inception_score"])
                    else:
                        row.update(val_ema_fid=values["fid"], val_ema_is=values["inception_score"])
            if epoch in config["pooled_epochs"]:
                for variant, current in variants:
                    with preserved_rng():
                        result = evaluate_variant(current, variant, config["pooled_reference"],
                                                  (LEGACY, POOLED), config, output, epoch, pooled=True)
                    if POOLED in result["metrics"]:
                        row["val_pool_fid" if variant == "raw" else "val_ema_pool_fid"] = result["metrics"][POOLED]["fid"]
                    row[f"val_{variant}_pooled_feature_artifact"] = result["feature_artifact"]
            row["validation_seconds"] = time.monotonic()-validation_start
            if row["val_fid"] is not None and row["val_fid"] < best_val:
                best_val, best_epoch, monitor_selected = row["val_fid"], epoch, True
        if epoch in config["diagnostic_epochs"]:
            diagnostics(generator, discriminator, dataset, device, config, output, epoch, shadow)
        elif epoch in PREVIEW_EPOCHS or epoch == config["epochs"]:
            preview(generator, shadow, device, config, output, epoch)
        history.append(row)
        write_history(output/"history.csv", history)
        inference = {"campaign_schema_version": 1, "generator_state": generator.state_dict(),
                     "epoch": epoch, "config": config, "history": history,
                     "cache_fingerprint": fingerprint, "best_val_fid": best_val,
                     "best_epoch": best_epoch, "ema_initialization": ema_initialization,
                     "global_g_steps": global_g_steps, "global_d_steps": global_d_steps,
                     "selection_scope": "validation monitoring candidate; final selection remains unfrozen"}
        if shadow is not None:
            inference["generator_ema_state"] = shadow.state_dict()
        if epoch % config["save_every"] == 0 or epoch == config["epochs"]:
            last = inference | {"discriminator_state": discriminator.state_dict(),
                "g_optimizer_state": g_optimizer.state_dict(), "d_optimizer_state": d_optimizer.state_dict(),
                "data_rng_state": data_rng.get_state(), "worker_rng_state": worker_rng.get_state(),
                "noise_rng_state": noise_rng.get_state(), "augmentation_rng_state": augmentation_rng.get_state(),
                **capture_global_rng()}
            if diffaugment is not None:
                last["diffaugment_state"] = diffaugment.state_dict()
            save_checkpoint(output/"last.pt", last)
        if epoch in config["save_epochs"]:
            save_checkpoint(output/"checkpoints"/f"epoch_{epoch:03d}.pt", inference)
        if monitor_selected:
            save_checkpoint(output/"best_monitor_raw.pt", inference)
        print(f"Epoch {epoch}/{config['epochs']}: G={row['g_loss']:.4f} D={row.get('d_loss')}, "
              f"probFID={row['val_fid']}, EMAprobFID={row['val_ema_fid']}, "
              f"G/D steps={global_g_steps}/{global_d_steps}, train={row['seconds']:.1f}s", flush=True)
    return {"output_dir": str(output), "last_checkpoint": str(output/"last.pt"),
            "completed_epoch": config["epochs"], "global_g_steps": global_g_steps,
            "global_d_steps": global_d_steps, "monitor_best_epoch": best_epoch}


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
