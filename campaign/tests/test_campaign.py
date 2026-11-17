"""Tiny full-orchestrator, epoch-boundary interrupted/resumed CPU check."""

import importlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import torch

draft = Path(__file__).parent
from hw2_gan import campaign_train as campaign


def assert_equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], path+"."+str(key))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, path+f"[{i}]")
    else:
        assert left == right, (path, left, right)


def fixture(cache):
    cache.mkdir()
    images = np.random.default_rng(63).integers(0, 256, (16, 64, 64, 3), dtype=np.uint8)
    np.save(cache/"images.npy", images)
    manifest = {"fingerprint": hashlib.sha256(images.tobytes()).hexdigest(),
                "splits": {"train": list(range(8)), "val": list(range(8, 12)), "test": list(range(12, 16))},
                "dataset": "SYNTHETIC CPU integration fixture", "dataset_version": 1,
                "preprocessing": "synthetic uint8 RGB64; [-1,1] mapping identical to real cache"}
    (cache/"manifest.json").write_text(json.dumps(manifest))


def check_initialization_isolation():
    config = {"latent_dim": 8, "g_base_channels": 2, "d_base_channels": 2,
              "d_spectral_norm": False, "d_batch_norm": False,
              "data_order": "epoch-v1", "seed": 42}
    before = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    g0, d0 = campaign.construct_models(config, torch.device("cpu"))
    g1, d1 = campaign.construct_models(config | {"d_spectral_norm": True}, torch.device("cpu"))
    assert_equal(g0.state_dict(), g1.state_dict(), "matched_G_initialization")
    conv0 = [m for m in d0.modules() if isinstance(m, torch.nn.Conv2d)]
    conv1 = [m for m in d1.modules() if isinstance(m, torch.nn.Conv2d)]
    for a,b in zip(conv0,conv1):
        assert torch.equal(a.weight, b.weight_orig), "SN/noBN underlying convolution initialization differs"
    assert torch.equal(before, torch.get_rng_state()), "Component initialization changed global CPU RNG"
    if before_cuda:
        assert all(torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all()))
    print("PASS shared fresh G and R0/SN original-conv initialization; global CPU/CUDA RNG preserved", flush=True)


def check(worker_count, root):
    cache = root/"cache"
    fixture(cache)
    common = ["--cache-dir", str(cache), "--device", "cpu", "--integration-smoke",
              "--latent-dim", "8", "--base-channels", "2", "--batch-size", "2",
              "--num-workers", str(worker_count), "--data-order", "epoch-v1", "--ema-decay", ".9",
              "--horizontal-flip", "--d-update-interval", "2", "--r1-gamma", "1",
              "--r1-interval", "2", "--d-no-batchnorm", "--diagnostic-epochs", "2",
              "--diagnostic-samples", "3", "--evaluation-batch-size", "2"]
    uninterrupted = root/"continuous"
    interrupted = root/"resumed"
    campaign.train(campaign.build_parser().parse_args(common+[
        "--output-dir", str(uninterrupted), "--epochs", "2"]))
    campaign.train(campaign.build_parser().parse_args(common+[
        "--output-dir", str(interrupted), "--epochs", "1"]))
    campaign.train(campaign.build_parser().parse_args([
        "--cache-dir", str(cache), "--output-dir", str(interrupted), "--device", "cpu",
        "--epochs", "2", "--resume", str(interrupted/"last.pt")]))
    first = torch.load(uninterrupted/"last.pt", weights_only=True)
    second = torch.load(interrupted/"last.pt", weights_only=True)
    keys = ["generator_state", "discriminator_state", "generator_ema_state", "g_optimizer_state",
            "d_optimizer_state", "data_rng_state", "worker_rng_state", "noise_rng_state",
            "augmentation_rng_state", "torch_rng_state", "python_rng_state", "numpy_rng_state",
            "global_g_steps", "global_d_steps", "ema_initialization"]
    for key in keys:
        assert_equal(first[key], second[key], key)
    assert first["global_g_steps"] == 8 and first["global_d_steps"] == 4
    for left, right in zip(first["history"], second["history"]):
        for key in left:
            if key not in ("seconds", "validation_seconds"):
                assert_equal(left[key], right[key], f"history.{left['epoch']}.{key}")
    assert not (interrupted/".campaign.lock").exists()
    assert (interrupted/"checkpoints"/"epoch_002.pt").exists()
    assert (interrupted/"diagnostics"/"epoch_002.json").exists()
    for variant in ("raw", "ema"):
        assert (interrupted/"previews"/f"epoch_002_{variant}.png").exists()
    # Changed dynamics must reject before touching model state or output artifacts.
    try:
        campaign.train(campaign.build_parser().parse_args([
            "--cache-dir", str(cache), "--output-dir", str(interrupted), "--device", "cpu",
            "--epochs", "3", "--resume", str(interrupted/"last.pt"), "--g-learning-rate", ".001"]))
    except ValueError as error:
        assert "g_learning_rate" in str(error)
    else:
        raise AssertionError("Resume changed G LR without rejection.")
    print(f"PASS full two-epoch versus one+resume exact states/RNG/metrics, workers={worker_count}", flush=True)


def check_legacy_fixture(root):
    cache = root/"cache"
    fixture(cache)
    output = root/"legacy"
    campaign.train(campaign.build_parser().parse_args([
        "--cache-dir", str(cache), "--output-dir", str(output), "--device", "cpu",
        "--integration-smoke", "--latent-dim", "8", "--base-channels", "2",
        "--batch-size", "2", "--num-workers", "0", "--epochs", "1"]))
    checkpoint = torch.load(output/"last.pt", weights_only=True)
    historical_keys = ["cache_dir", "output_dir", "epochs", "latent_dim", "base_channels", "batch_size",
                       "learning_rate", "beta1", "seed", "num_workers", "eval_every", "device", "resume"]
    checkpoint["config"] = {key: checkpoint["config"][key] for key in historical_keys}
    # This one annotation keeps the synthetic fixture explicitly marked; no real
    # production baseline receives a truncation or an evaluation bypass.
    checkpoint["config"]["integration_smoke"] = True
    for key in ["campaign_schema_version", "global_g_steps", "global_d_steps", "worker_rng_state",
                "augmentation_rng_state", "python_rng_state", "numpy_rng_state", "ema_initialization"]:
        checkpoint.pop(key, None)
    checkpoint["history"] = [{key: row[key] for key in ("epoch", "seconds", "g_loss", "d_loss", "val_fid", "val_is")}
                             for row in checkpoint["history"]]
    historical_row = dict(checkpoint["history"][0])
    torch.save(checkpoint, output/"last.pt")
    # Simulate evidence from completed epochs newer than a sparse last.pt.
    campaign.write_history(output/"history.csv", checkpoint["history"]+[
        {"epoch": 2, "g_loss": 999.0, "d_loss": 999.0, "uncheckpointed_fixture": True}])
    campaign.train(campaign.build_parser().parse_args([
        "--cache-dir", str(cache), "--output-dir", str(output), "--device", "cpu",
        "--epochs", "2", "--resume", str(output/"last.pt"), "--ema-decay", ".9"]))
    result = torch.load(output/"last.pt", weights_only=True)
    assert result["history"][0] == historical_row
    assert result["global_g_steps"] == 8 and result["global_d_steps"] == 8
    assert result["ema_initialization"]["source"] == "raw_generator_at_resume"
    assert result["ema_initialization"]["epoch"] == 1
    assert "generator_ema_state" in result and result["config"]["data_order"] == "legacy"
    assert list((output/"recovery").glob("history_after_checkpoint_001_*.csv"))
    header = (output/"history.csv").read_text().splitlines()[0]
    assert "g_gradient_norm" in header and "g_loss" in header
    print("PASS legacy-format optimizer resume, preserved old rows, counter inference, explicit EMA initialization, recovery evidence", flush=True)


def main():
    torch.set_num_threads(2)
    check_initialization_isolation()
    with tempfile.TemporaryDirectory(dir=draft) as temporary:
        root = Path(temporary)
        for workers in (0, 1):
            directory = root/f"workers{workers}"
            directory.mkdir()
            check(workers, directory)
        legacy = root/"legacy_fixture"
        legacy.mkdir()
        check_legacy_fixture(legacy)
    print("All campaign CPU integration checks passed on synthetic images; no quality claim.", flush=True)


if __name__ == "__main__":
    main()
