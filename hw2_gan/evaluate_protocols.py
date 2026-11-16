"""Versioned HW2 evaluation; keep evaluate.py unchanged for historical replay.

Install this draft next to hw2_gan/evaluate.py. Feature extraction and production
scoring require an allocation. Plan construction and tiny synthetic tests do not.
The pooled protocol uses torchvision weights/preprocessing and is NOT numerically
interchangeable with TensorFlow FID, pytorch-fid, or Clean-FID published scores.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LEGACY = "legacy1000prob-v1"
POOLED = "torchvision2048pool-v1"
PROTOCOLS = (LEGACY, POOLED)
PLAN_VERSION = "wikiart-fixed-references-v1"
EVALUATION_BACKEND = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                      "cudnn_benchmark": False, "cudnn_deterministic": True}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(f"{value.dtype.str}:{value.shape}".encode())
    result.update(memoryview(value).cast("B"))
    return result.hexdigest()


def require_allocation():
    if os.environ.get("NERSC_HOST") == "perlmutter" and (
        not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")
    ):
        raise RuntimeError("Production feature extraction and scoring require a compute allocation.")


def _atomic_write(path, writer, *, replace=False):
    """Unique same-filesystem temporary + fsync + atomic commit.

    Callers that share a destination must hold its lock. The hard-link commit
    prevents an uncoordinated writer from replacing an existing immutable file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_once(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    _atomic_write(path, lambda handle: handle.write(payload))


@contextmanager
def _exclusive_lock(path, timeout=1800.0):
    """Advisory POSIX lock; lock files persist, and process exit releases locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for cache lock {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class ImmutableFeatureCache:
    """Content-addressed NPZ files with independent read-only consumers.

    Metadata identities include data, ordered reference indices, and protocol.
    No writer replaces a committed entry, even if the old entry is corrupt:
    corruption raises and must be investigated instead of silently overwritten.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, identity):
        return self.directory / identity["protocol_id"] / f"{digest(identity)}.npz"

    def read(self, identity):
        path = self.path(identity)
        with np.load(path, allow_pickle=False) as saved:
            envelope = json.loads(str(saved["metadata"].item()))
            if envelope["identity"] != identity:
                raise ValueError(f"Cache metadata mismatch: {path}")
            arrays = {key: saved[key].copy() for key in saved.files if key != "metadata"}
        if set(arrays) != set(envelope["arrays"]):
            raise ValueError(f"Cache array inventory mismatch: {path}")
        for key, value in arrays.items():
            description = envelope["arrays"][key]
            if list(value.shape) != description["shape"] or value.dtype.str != description["dtype"]:
                raise ValueError(f"Cache shape/dtype mismatch: {path}:{key}")
            if not np.isfinite(value).all() or array_sha256(value) != description["sha256"]:
                raise ValueError(f"Cache content integrity failure: {path}:{key}")
            value.flags.writeable = False
        return arrays

    def get_or_create(self, identity, builder=None, *, mode="readonly", lock_timeout=1800.0):
        if mode not in ("readonly", "create"):
            raise ValueError("cache mode must be readonly or create")
        path = self.path(identity)
        if path.exists():
            return self.read(identity), True
        if mode == "readonly":
            raise FileNotFoundError(f"Precompute immutable real cache before launching workers: {path}")
        if builder is None:
            raise ValueError("A cache builder is required in create mode")
        with _exclusive_lock(path.with_suffix(".lock"), lock_timeout):
            if path.exists():
                return self.read(identity), True
            arrays = {key: np.ascontiguousarray(value) for key, value in builder().items()}
            if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Cannot cache empty or nonfinite feature arrays")
            envelope = {
                "identity": identity,
                "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str,
                                  "sha256": array_sha256(value)} for key, value in arrays.items()},
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(path, lambda handle: np.savez_compressed(
                handle, metadata=np.array(canonical_json(envelope)), **arrays))
            return self.read(identity), False


def dataset_identity(cache_dir, *, pixel_cache_sha256):
    """Hash manifest provenance; pixel SHA256 must be measured once in allocation.

    The historical fingerprint covers ordered file names and raw byte hashes;
    it alone does not identify preprocessing or the split assignment.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(pixel_cache_sha256, str) or len(pixel_cache_sha256) != 64:
        raise ValueError("Pass the measured images.npy SHA256 (64 hexadecimal characters)")
    int(pixel_cache_sha256, 16)
    # Historical RGB64 behavior remains exact. RGB128 binds the actual stored
    # producer source rather than incorrectly labeling it as data.py.
    preprocessing_source_hash = file_sha256(Path(__file__).with_name("data.py"))
    if manifest.get("image_size", 64) == 128:
        preprocessing_source_hash = file_sha256(cache_dir/"preprocessing_source.py")
        if preprocessing_source_hash != manifest["preprocessing_source_sha256"]:
            raise ValueError("RGB128 stored preprocessing source identity differs.")
    identity = {
        "dataset": manifest["dataset"], "dataset_version": manifest["dataset_version"],
        "raw_inventory_fingerprint": manifest.get("source_raw_fingerprint", manifest["fingerprint"]),
        "raw_content_sha256_digest": digest(manifest["content_sha256"]),
        "ordered_files_digest": digest(manifest["files"]),
        "manifest_sha256": file_sha256(manifest_path),
        "preprocessing": manifest["preprocessing"],
        "preprocessing_source_sha256": preprocessing_source_hash,
        "preprocessed_pixels_sha256": pixel_cache_sha256.lower(),
        "split_mapping_digest": digest(manifest["splits"]),
        "split_seed": manifest["split_seed"], "split_rule": manifest["split_rule"],
        "split_counts": {key: len(value) for key, value in manifest["splits"].items()},
    }
    if manifest.get("image_size", 64) == 128:
        identity["image_size"] = 128
        identity["processed_cache_fingerprint"] = manifest["fingerprint"]
    return manifest, identity


def _unique_content_indices(indices, manifest, *, excluded_hashes=()):
    seen = set(excluded_hashes)
    unique = []
    # Canonical representative is the first index in the original split order.
    for index in indices:
        raw_hash = manifest["content_sha256"][index]
        if raw_hash not in seen:
            seen.add(raw_hash)
            unique.append(index)
    return unique


def _permuted(indices, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(indices), generator=rng).tolist()
    return [indices[i] for i in order]


def freeze_reference_plan(cache_dir, output_dir, historical_test_metrics, *,
                          pixel_cache_sha256, latent_dim=100, count=4000,
                          validation_prior_seed=20261115, validation_reference_seed=20261116,
                          test_prior_seed=20261117, test_reference_seed=20261118,
                          legacy_seed=43):
    """Freeze explicit index lists and exact latent arrays before model selection.

    N=4096 cannot support two disjoint halves of 8155 validation images.
    N=4000 is fixed for every pooled finalist, its real-v-real control, and final
    test. New references weight each byte-unique painting once; training and the
    historical probability protocol retain their original file weighting.
    """
    if count != 4000:
        raise ValueError("Campaign v1 preregisters exactly N=4000")
    manifest, identity = dataset_identity(cache_dir, pixel_cache_sha256=pixel_cache_sha256)
    old = json.loads(Path(historical_test_metrics).read_text())
    old_indices = old["reference_indices"]
    if old["reference_split"] != "test" or len(old_indices) != 4096:
        raise ValueError("Expected the audited historical 4096-image test artifact")
    raw_fingerprint = manifest.get("source_raw_fingerprint", manifest["fingerprint"])
    if old["cache_fingerprint"] != raw_fingerprint or not set(old_indices) <= set(manifest["splits"]["test"]):
        raise ValueError("Historical test reference does not belong to this dataset")
    val = _permuted(_unique_content_indices(manifest["splits"]["val"], manifest), validation_reference_seed)
    excluded_hashes = {manifest["content_sha256"][i] for i in old_indices}
    old_index_set = set(old_indices)
    unused_test = [i for i in manifest["splits"]["test"] if i not in old_index_set]
    test = _permuted(_unique_content_indices(unused_test, manifest, excluded_hashes=excluded_hashes), test_reference_seed)
    if len(val) < 2 * count or len(test) < count:
        raise ValueError("Insufficient byte-unique disjoint references at preregistered N")
    legacy = _permuted(manifest["splits"]["val"], legacy_seed + 100000)[:1024]
    references = {
        "validation_legacy1024": {"split": "val", "global_indices": legacy, "prior": "legacy", "protocol_ids": [LEGACY]},
        "validation_pool4000_a": {"split": "val", "global_indices": val[:count], "prior": "validation", "protocol_ids": list(PROTOCOLS)},
        "validation_pool4000_b": {"split": "val", "global_indices": val[count:2*count], "prior": None, "protocol_ids": list(PROTOCOLS)},
        "final_test4000": {"split": "test", "global_indices": test[:count], "prior": "test", "protocol_ids": list(PROTOCOLS)},
        "historical_test4096": {"split": "test", "global_indices": old_indices, "prior": "historical_test", "protocol_ids": [LEGACY]},
    }
    seeds = {"legacy": legacy_seed, "validation": validation_prior_seed,
             "test": test_prior_seed, "historical_test": old["prior_seed"]}
    counts = {"legacy": 1024, "validation": count, "test": count, "historical_test": len(old_indices)}
    priors = {key: torch.randn(counts[key], latent_dim,
                generator=torch.Generator(device="cpu").manual_seed(seed)).numpy()
              for key, seed in seeds.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prior_path = output_dir / "fixed_priors.npz"
    _atomic_write(prior_path, lambda handle: np.savez_compressed(handle, **priors))
    plan = {
        "plan_version": PLAN_VERSION, "dataset_identity": identity,
        "reference_sampling": "new pool/test: uniform permutation of first representative per raw-content SHA256; legacy: original file indices",
        "validation_reference_seed": validation_reference_seed, "test_reference_seed": test_reference_seed,
        "legacy_reference_seed": legacy_seed + 100000,
        "references": references, "latent_dim": latent_dim, "prior_seeds": seeds,
        "prior_vectors_file": prior_path.name, "prior_vectors_file_sha256": file_sha256(prior_path),
        "prior_array_sha256": {key: array_sha256(value) for key, value in priors.items()},
        "historical_test_metrics_sha256": file_sha256(historical_test_metrics),
        "historical_test_protocol": old["protocol"], "historical_test_protocol_hash": old["protocol_hash"],
        "test_complement_counts": {"indices_after_old_index_exclusion": len(unused_test),
            "unique_contents_after_old_raw_hash_exclusion": len(test)},
        "decision_rule": "Freeze checkpoint/variant selection on validation_pool4000_a pooled FID; IS and fixed grids support interpretation; final_test4000 is used only after selection is frozen.",
        "disclosures": ["N4000 allows equal-N disjoint validation real-v-real; two N4096 halves cannot fit 8155.",
                        "Byte-identical contents are deduplicated only for new pooled/test references; near duplicates and artist dependence remain unaudited.",
                        "Original test split has earlier evaluation exposure; new references exclude old used indices and all their raw hashes."],
    }
    plan["plan_sha256"] = digest(plan)
    write_json_once(output_dir / "reference_plan.json", plan)
    return plan


def load_reference_plan(path, *, cache_dir=None, verify_pixels=False):
    path = Path(path)
    plan = json.loads(path.read_text())
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_version"] != PLAN_VERSION or digest(body) != plan["plan_sha256"]:
        raise ValueError("Reference plan checksum/version mismatch")
    prior_path = path.parent / plan["prior_vectors_file"]
    if file_sha256(prior_path) != plan["prior_vectors_file_sha256"]:
        raise ValueError("Fixed-prior file checksum mismatch")
    with np.load(prior_path, allow_pickle=False) as saved:
        priors = {key: saved[key].copy() for key in saved.files}
    for key, value in priors.items():
        if array_sha256(value) != plan["prior_array_sha256"][key] or value.dtype != np.float32:
            raise ValueError("Fixed-prior array checksum/dtype mismatch")
        value.flags.writeable = False
    if cache_dir is not None:
        _, observed = dataset_identity(cache_dir, pixel_cache_sha256=plan["dataset_identity"]["preprocessed_pixels_sha256"])
        if observed != plan["dataset_identity"]:
            raise ValueError("Dataset identity differs from the frozen reference plan")
        if verify_pixels:
            require_allocation()
            if file_sha256(Path(cache_dir)/"images.npy") != observed["preprocessed_pixels_sha256"]:
                raise ValueError("Preprocessed pixel cache checksum mismatch")
    return plan, priors


def protocol_spec(protocol_id, device):
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol_id}")
    common = {
        "protocol_id": protocol_id,
        "features": "1000 ImageNet class probabilities" if protocol_id == LEGACY else "2048 average-pooled Inception-v3 activations before dropout/fc",
        "weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
        "weights_url": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
        "input": "RGB [-1,1] mapped to [0,1]",
        "resize": "299x299 bilinear, align_corners=False, antialias=True",
        "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
        "transform_input": True, "effective_first_conv_input": "2 * resized_RGB_[0,1] - 1",
        "inception_score": "aggregate over all generated softmax rows; no split averaging",
        "feature_device_type": torch.device(device).type, "precision": "float32; autocast disabled",
        "runtime_numeric_settings": dict(EVALUATION_BACKEND),
        "torchvision_inception_source_sha256": file_sha256(
            Path(importlib.util.find_spec("torchvision").origin).parent / "models" / "inception.py"),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "pytorch-ignite", "numpy", "scipy")},
        "covariance": "unbiased sample covariance, denominator N-1",
        "metric_implementation": "historical hw2_gan.evaluate.compute_feature_metrics/Ignite" if protocol_id == LEGACY else "float64 symmetric PSD eigendecomposition of C_real^(1/2) C_gen C_real^(1/2)",
    }
    return common


def cache_identity(plan, reference_name, protocol):
    reference = plan["references"][reference_name]
    return {"cache_format": "immutable-npz-v1", "dataset_identity": plan["dataset_identity"],
            "reference_split": reference["split"], "global_indices": reference["global_indices"],
            "protocol_id": protocol["protocol_id"], "protocol": protocol,
            "statistics_version": "float64-unbiased-cov-symmetric-sqrt-v1" if protocol["protocol_id"] == POOLED else None}


class _ReferencePixels(Dataset):
    """Read-only mmap avoids one 1-GB resident copy per evaluator worker."""
    def __init__(self, cache_dir, global_indices):
        path = Path(cache_dir)/"images.npy"
        try:
            self.images = np.load(path, mmap_mode="r")
            self.loading_method = "numpy-mmap-readonly"
        except OSError as error:
            if error.errno not in {errno.ENOSYS, errno.ENODEV, errno.EOPNOTSUPP}:
                raise
            self.images = np.load(path)
            self.loading_method = f"numpy-load-mmap-unsupported-errno{error.errno}"
        self.global_indices = global_indices

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, index):
        pixels = np.array(self.images[self.global_indices[index]], copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1)


class InceptionFeatures(nn.Module):
    """One network forward yields pool features and classifier probabilities."""
    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        self.network = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                    transform_input=True).eval().requires_grad_(False)

    def forward(self, images):
        from .evaluate import preprocess_inception
        pooled = []
        hook = self.network.avgpool.register_forward_hook(
            lambda module, inputs, output: pooled.append(output.flatten(1)))
        try:
            logits = self.network(preprocess_inception(images))
        finally:
            hook.remove()
        if len(pooled) != 1 or pooled[0].shape != (len(images), 2048):
            raise RuntimeError("Unexpected Inception average-pool output")
        return {"features": pooled[0], "probabilities": logits.softmax(dim=1)}


@contextmanager
def _inference_context(device):
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    old_settings = {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        torch.backends.cuda.matmul.allow_tf32 = EVALUATION_BACKEND["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = EVALUATION_BACKEND["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = EVALUATION_BACKEND["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = EVALUATION_BACKEND["cudnn_deterministic"]
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_settings["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = old_settings["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = old_settings["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = old_settings["cudnn_deterministic"]


def _collect(extractor, batches, device):
    chunks = {"features": [], "probabilities": []}
    for images in batches:
        output = extractor(images.to(device))
        for key in chunks:
            chunks[key].append(output[key].cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    _validate_features(arrays)
    return arrays


def _validate_features(arrays):
    pool, probabilities = arrays["features"], arrays["probabilities"]
    if pool.ndim != 2 or pool.shape[1] != 2048 or probabilities.shape != (len(pool), 1000) or len(pool) < 2:
        raise ValueError("Expected N>=2 pooled2048 and softmax1000 arrays")
    if not np.isfinite(pool).all() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Nonfinite features or invalid probabilities")
    if not np.allclose(probabilities.sum(1), 1, rtol=1e-5, atol=1e-5):
        raise ValueError("Probability rows do not sum to one")


def feature_moments(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("Finite Nxd features with N>=2 required")
    if features.shape[1] > 16:
        require_allocation()
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = (centered.T @ centered) / (len(features)-1)
    return mean, (covariance + covariance.T)/2


def _psd_eigh(matrix):
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)/2
    if matrix.shape[0] > 16:
        require_allocation()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Numerical guard only: dimension-scaled float64 backward-error heuristic.
    # Its observed size is reported; it is never an image-quality threshold.
    tolerance = 64 * np.finfo(np.float64).eps * len(matrix) * max(np.abs(eigenvalues).max(), np.finfo(np.float64).tiny)
    if eigenvalues.min() < -tolerance:
        raise FloatingPointError(f"Covariance not PSD: min={eigenvalues.min()}, guard={tolerance}")
    return np.maximum(eigenvalues, 0), eigenvectors, tolerance


def moment_bundle(features):
    mean, covariance = feature_moments(features)
    eigenvalues, eigenvectors, _ = _psd_eigh(covariance)
    covariance_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    return {"mean": mean, "covariance": covariance, "covariance_sqrt": covariance_sqrt}


def pooled_fid(real_moments, generated_features):
    mean_g, covariance_g = feature_moments(generated_features)
    mean_r = real_moments["mean"]
    covariance_r = real_moments["covariance"]
    sqrt_r = real_moments["covariance_sqrt"]
    middle = sqrt_r @ covariance_g @ sqrt_r
    eigenvalues, _, tolerance = _psd_eigh(middle)
    mean_term = float(np.dot(mean_r-mean_g, mean_r-mean_g))
    trace_term = float(np.trace(covariance_r)+np.trace(covariance_g))
    cross_term = float(2*np.sqrt(eigenvalues).sum())
    raw = mean_term + trace_term - cross_term
    if not math.isfinite(raw) or raw < -1e-6:
        raise FloatingPointError(f"Invalid pooled FID: {raw}")
    return {"fid": max(raw, 0.0), "fid_raw_before_roundoff_clamp": raw,
            "mean_distance_squared": mean_term, "covariance_trace_sum": trace_term,
            "twice_covariance_cross_trace": cross_term, "psd_eigenvalue_roundoff_guard": tolerance,
            "negative_fid_roundoff_guard": 1e-6}


def aggregate_inception_score(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) < 2 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Finite nonnegative Nx1000 probabilities required")
    if not np.allclose(probabilities.sum(1), 1, rtol=1e-5, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    marginal = probabilities.mean(0)
    # 0 log(0/q)=0. Zero-marginal classes have zero probability in every row.
    log_marginal = np.zeros_like(marginal)
    np.log(marginal, out=log_marginal, where=marginal > 0)
    log_conditional = np.zeros_like(probabilities)
    np.log(probabilities, out=log_conditional, where=probabilities > 0)
    return float(np.exp(np.sum(probabilities * (log_conditional-log_marginal), axis=1).mean()))


def precompute_real_features(cache_dir, plan_path, feature_cache_dir, reference_names, *,
                             protocol_ids=PROTOCOLS, device="cuda", batch_size=64,
                             verify_pixel_checksum=True):
    """Single owner runs once, then parallel workers consume mode=readonly.

    If both protocols are requested, one image forward supplies both arrays.
    Real pooled means/covariances/square roots are calculated once on allocated
    CPU and stored in the immutable pooled cache.
    """
    require_allocation()
    plan, _ = load_reference_plan(plan_path, cache_dir=cache_dir, verify_pixels=verify_pixel_checksum)
    device = torch.device(device)
    cache = ImmutableFeatureCache(feature_cache_dir)
    extractor = None
    receipts = []
    with _inference_context(device):
        for reference_name in reference_names:
            reference = plan["references"][reference_name]
            requested = [pid for pid in protocol_ids if pid in reference["protocol_ids"]]
            shared = {}

            def extract_once():
                nonlocal extractor
                if "arrays" not in shared:
                    if extractor is None:
                        extractor = InceptionFeatures().to(device).eval()
                    dataset = _ReferencePixels(cache_dir, reference["global_indices"])
                    shared["pixel_loading_method"] = dataset.loading_method
                    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False,
                                        generator=torch.Generator().manual_seed(0))
                    shared["arrays"] = _collect(extractor, loader, device)
                return shared["arrays"]

            for protocol_id in requested:
                protocol = protocol_spec(protocol_id, device)
                identity = cache_identity(plan, reference_name, protocol)

                def build(protocol_id=protocol_id):
                    arrays = extract_once()
                    if protocol_id == LEGACY:
                        return {"probabilities": arrays["probabilities"]}
                    return {**arrays, **moment_bundle(arrays["features"])}

                _, hit = cache.get_or_create(identity, build, mode="create")
                receipts.append({"reference": reference_name, "protocol_id": protocol_id,
                                 "path": str(cache.path(identity)), "cache_hit": hit,
                                 "pixel_loading_method": shared.get("pixel_loading_method", "cache-hit-no-pixel-load")})
    return receipts


def extract_generator_features(G, cache_dir, plan_path, reference_name, feature_cache_dir,
                               output_dir, *, protocol_ids=PROTOCOLS, device="cuda",
                               batch_size=64, generator_variant="raw", checkpoint_metadata=None):
    """GPU inference only; create an immutable candidate artifact, score later.

    G is already the explicitly selected raw or EMA generator. Every requested
    real cache must exist before any inference. Training RNG, every submodule's
    mode and TF32/cuDNN settings are restored even if extraction raises. Backend
    settings are pinned in the versioned protocol and recorded in its identity.
    """
    require_allocation()
    if generator_variant not in ("raw", "ema") or batch_size < 1:
        raise ValueError("Use raw/ema and a positive batch size")
    plan, priors = load_reference_plan(plan_path, cache_dir=cache_dir)
    reference = plan["references"][reference_name]
    if reference["prior"] is None:
        raise ValueError("A real-v-real control reference has no generator prior")
    if getattr(G, "latent_dim", None) != plan["latent_dim"]:
        raise ValueError("Generator latent dimension differs from frozen priors")
    device = torch.device(device)
    protocols = [protocol_spec(pid, device) for pid in protocol_ids]
    if any(p["protocol_id"] not in reference["protocol_ids"] for p in protocols):
        raise ValueError("Protocol not allowed by this frozen reference")
    cache = ImmutableFeatureCache(feature_cache_dir)
    real_paths = {}
    for protocol in protocols:
        identity = cache_identity(plan, reference_name, protocol)
        cache.get_or_create(identity, mode="readonly")
        real_paths[protocol["protocol_id"]] = str(cache.path(identity).resolve())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    modes = [(module, module.training) for module in G.modules()]
    try:
        G.eval()
        with _inference_context(device):
            extractor = InceptionFeatures().to(device).eval()
            latents = torch.from_numpy(priors[reference["prior"]].copy())
            batches = (G(z.to(device)) for z in latents.split(batch_size))
            arrays = _collect(extractor, batches, device)
    finally:
        for module, training in modes:
            module.training = training
    from .evaluate import _state_hash
    metadata = {"artifact_version": "candidate-inception-features-v1",
                "plan_sha256": plan["plan_sha256"], "reference_name": reference_name,
                "num_generated_images": len(latents), "prior_seed": plan["prior_seeds"][reference["prior"]],
                "prior_array_sha256": plan["prior_array_sha256"][reference["prior"]],
                "generator_variant": generator_variant, "generator_state_sha256": _state_hash(G),
                "checkpoint": checkpoint_metadata or {}, "protocols": protocols,
                "real_feature_caches": real_paths, "batch_size": batch_size, "device": str(device),
                "arrays": {key: array_sha256(value) for key, value in arrays.items()}}
    feature_path = output_dir / "generated_features.npz"
    _atomic_write(feature_path, lambda handle: np.savez_compressed(
        handle, metadata=np.array(canonical_json(metadata)), **arrays))
    write_json_once(output_dir / "extraction.json", {**metadata, "feature_file_sha256": file_sha256(feature_path)})
    return feature_path


def _read_candidate(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        arrays = {key: saved[key].copy() for key in ("features", "probabilities")}
    _validate_features(arrays)
    if any(array_sha256(value) != metadata["arrays"][key] for key, value in arrays.items()):
        raise ValueError("Generated feature checksum mismatch")
    return metadata, arrays


def score_feature_artifact(candidate_path, plan_path, feature_cache_dir, output_path):
    """Allocated CPU scoring from saved arrays; no network/GPU inference."""
    require_allocation()
    plan, _ = load_reference_plan(plan_path)
    metadata, generated = _read_candidate(candidate_path)
    if metadata["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Candidate and reference plans differ")
    cache = ImmutableFeatureCache(feature_cache_dir)
    metrics = {}
    for protocol in metadata["protocols"]:
        identity = cache_identity(plan, metadata["reference_name"], protocol)
        real, _ = cache.get_or_create(identity, mode="readonly")
        if len(real["probabilities"]) != len(generated["probabilities"]):
            raise ValueError("Real/generated sample counts differ")
        if protocol["protocol_id"] == LEGACY:
            from .evaluate import compute_feature_metrics
            values = compute_feature_metrics(torch.from_numpy(real["probabilities"].copy()),
                                             torch.from_numpy(generated["probabilities"]))
        else:
            values = pooled_fid(real, generated["features"])
            values.update(inception_score=aggregate_inception_score(generated["probabilities"]),
                          real_inception_score_reference=aggregate_inception_score(real["probabilities"]))
        metrics[protocol["protocol_id"]] = {**values, "protocol": protocol, "protocol_hash": digest(protocol),
                                            "num_real_images": len(real["probabilities"]),
                                            "num_generated_images": len(generated["probabilities"])}
    result = {**metadata, "metrics": metrics, "candidate_file_sha256": file_sha256(candidate_path),
              "reference_indices": plan["references"][metadata["reference_name"]]["global_indices"],
              "limitations": ["Pooled torchvision FID is protocol-specific; do not compare to probability FID or a different implementation.",
                              "Finite-sample FID is biased; compare models at equal N and identical references.",
                              "IS and ImageNet features do not establish artistic quality or originality."]}
    write_json_once(output_path, result)
    return result


def score_real_control(plan_path, feature_cache_dir, output_path, *, device="cuda"):
    """Equal-N, disjoint raw-content validation A-vs-B FID, scored once."""
    require_allocation()
    plan, _ = load_reference_plan(plan_path)
    a, b = (plan["references"][f"validation_pool4000_{letter}"] for letter in ("a", "b"))
    if len(a["global_indices"]) != 4000 or len(b["global_indices"]) != 4000 or set(a["global_indices"]) & set(b["global_indices"]):
        raise ValueError("Control must have 4000 disjoint indices per half")
    cache = ImmutableFeatureCache(feature_cache_dir)
    metrics = {}
    for protocol_id in PROTOCOLS:
        protocol = protocol_spec(protocol_id, device)
        real_a = cache.read(cache_identity(plan, "validation_pool4000_a", protocol))
        real_b = cache.read(cache_identity(plan, "validation_pool4000_b", protocol))
        if protocol_id == LEGACY:
            from .evaluate import compute_feature_metrics
            values = compute_feature_metrics(torch.from_numpy(real_a["probabilities"].copy()),
                                             torch.from_numpy(real_b["probabilities"].copy()))
        else:
            values = pooled_fid(real_a, real_b["features"])
        metrics[protocol_id] = {**values, "protocol_hash": digest(protocol), "protocol": protocol}
    result = {"artifact_version": "disjoint-real-control-v1", "plan_sha256": plan["plan_sha256"],
              "reference_a": a, "reference_b": b, "N_per_half": 4000, "metrics": metrics,
              "interpretation": "One finite-sample/context control, not an unbiased correction or universal lower bound for generator FID."}
    write_json_once(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-plan")
    freeze.add_argument("--cache-dir", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--historical-test-metrics", type=Path, required=True)
    freeze.add_argument("--pixel-cache-sha256", required=True)
    precompute = commands.add_parser("precompute")
    precompute.add_argument("--cache-dir", type=Path, required=True)
    precompute.add_argument("--plan", type=Path, required=True)
    precompute.add_argument("--feature-cache-dir", type=Path, required=True)
    precompute.add_argument("--references", nargs="+", required=True)
    precompute.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    precompute.add_argument("--device", default="cuda")
    precompute.add_argument("--batch-size", type=int, default=64)
    score = commands.add_parser("score")
    score.add_argument("--candidate", type=Path, required=True)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--feature-cache-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    control = commands.add_parser("real-control")
    control.add_argument("--plan", type=Path, required=True)
    control.add_argument("--feature-cache-dir", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "freeze-plan":
        result = freeze_reference_plan(args.cache_dir, args.output_dir, args.historical_test_metrics,
                                      pixel_cache_sha256=args.pixel_cache_sha256)
        print(result["plan_sha256"])
    elif args.command == "precompute":
        print(json.dumps(precompute_real_features(args.cache_dir, args.plan, args.feature_cache_dir,
                    args.references, protocol_ids=args.protocols, device=args.device, batch_size=args.batch_size), indent=2))
    elif args.command == "score":
        score_feature_artifact(args.candidate, args.plan, args.feature_cache_dir, args.output)
    else:
        score_real_control(args.plan, args.feature_cache_dir, args.output, device=args.device)


if __name__ == "__main__":
    main()
