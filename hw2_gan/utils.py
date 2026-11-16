"""Checkpoint, image, and provenance helpers."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch


def require_compute():
    if os.environ.get("NERSC_HOST") == "perlmutter" and (
            not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
        raise RuntimeError("Run numerical work inside a compute allocation.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto"):
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary); temporary.replace(path)


def load_generator(path, device, *, weights="raw"):
    """Load raw (historical default) or explicitly requested EMA inference weights."""
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be 'raw' or 'ema'.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    width = config.get("g_base_channels", config.get("base_channels", 64))
    image_size = config.get("image_size", 64)
    if image_size == 64:
        from .model import Generator
        model = Generator(config["latent_dim"], width)
    elif image_size == 128:
        from .model128 import Generator128
        model = Generator128(config["latent_dim"], width)
    else:
        raise ValueError(f"Unsupported trained image size: {image_size}.")
    key = "generator_state" if weights == "raw" else "generator_ema_state"
    if key not in state:
        raise ValueError(f"Checkpoint has no {weights} generator weights ({key}).")
    model.load_state_dict(state[key])
    return model.to(device).eval(), state


def save_grid(images, path, columns=8):
    """Fixed [-1,1] to uint8 mapping; no per-image contrast enhancement."""
    pixels = ((images.detach().cpu().float().clamp(-1, 1) + 1) * 127.5)
    pixels = pixels.permute(0, 2, 3, 1).numpy().round().astype(np.uint8)
    n, height, width, _ = pixels.shape
    columns = min(columns, n); rows = (n + columns - 1) // columns
    panel = Image.new("RGB", (columns*(width+2)+2, rows*(height+2)+2))
    for i, array in enumerate(pixels):
        panel.paste(Image.fromarray(array), (2+(i%columns)*(width+2), 2+(i//columns)*(height+2)))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); panel.save(path)


def record_provenance(directory, config):
    directory = Path(directory)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = directory / "provenance" / stamp
    output.mkdir(parents=True)
    hashes = {}
    for source in Path(__file__).parent.glob("*.py"):
        (output/source.name).write_bytes(source.read_bytes())
        hashes[source.name] = file_sha256(source)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "numpy", "Pillow", "matplotlib", "pytorch-ignite", "scipy"]}
    save_json(output/"execution.json", {"utc": stamp, "argv": sys.argv, "config": config,
              "source_sha256": hashes, "versions": versions, "host": socket.gethostname(),
              "allocation": {key: os.environ.get(key) for key in
                             ["SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "CUDA_VISIBLE_DEVICES"]}})
