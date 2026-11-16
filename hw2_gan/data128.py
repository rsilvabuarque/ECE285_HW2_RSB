"""Native RGB128 cache from original ZIP bytes and the exact RGB64 split manifest.

This is a separate demonstration pipeline. It never opens images.npy from the
RGB64 cache or interpolates RGB64 training pixels. Full preprocessing is guarded
for allocated compute; the pure byte decoder supports tiny synthetic checks.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch.utils.data import Dataset

from .utils import require_compute, save_json


IMAGE_SIZE = 128
PROTOCOL_VERSION = "wikiart-native-rgb128-bilinear-centercrop-v1"
_archive = None


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value):
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_raw_image(raw):
    """Decode original image bytes into uint8 HWC RGB128 using legacy rounding.

    PIL conversion/resize conventions match RGB64 except for the native target:
    shorter edge 128, integer truncation of the other resized dimension, Python
    round for center offsets, no EXIF orientation transpose, no contrast scaling.
    """
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            shape = (128, int(128*height/width))
        else:
            shape = (int(128*width/height), 128)
        image = image.resize(shape, Image.Resampling.BILINEAR)
        left = round((shape[0]-128)/2)
        top = round((shape[1]-128)/2)
        pixels = np.array(image.crop((left, top, left+128, top+128)), dtype=np.uint8)
    if pixels.shape != (128, 128, 3):
        raise ValueError(f"Unexpected native RGB128 shape: {pixels.shape}.")
    return pixels, [width, height]


def preprocessing_identity(source_bytes):
    return {"protocol_version": PROTOCOL_VERSION, "image_size": 128,
            "input": "original raw bytes from verified full WikiArt ZIP; never RGB64 cache pixels",
            "color": "PIL convert RGB; no EXIF transpose",
            "resize": "PIL bilinear; shorter edge 128; other dimension int(128*long/short)",
            "crop": "128x128 center; offsets Python round((resized_dimension-128)/2)",
            "storage": "ordinary numpy.save uint8 NHWC; no mmap",
            "training_normalization": "float32 RGB NCHW; uint8/127.5-1",
            "pillow_version": pillow_version, "numpy_version": np.__version__,
            "preprocessor_file_sha256": _sha256_bytes(source_bytes),
            "preprocessor_function_sha256": _sha256_bytes(inspect.getsource(preprocess_raw_image).encode())}


def archive_inventory(source):
    """Use the identical name filter and sorting rule as historical data.py."""
    return sorted(info.filename for info in source.infolist()
                  if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
                  and not any(part.startswith(".") or part == "__MACOSX"
                              for part in Path(info.filename).parts))


def validate_reference(reference):
    required = {"dataset", "dataset_version", "files", "content_sha256", "fingerprint", "splits", "original_sizes"}
    if not required.issubset(reference):
        raise ValueError(f"Original RGB64 manifest missing {sorted(required-set(reference))}.")
    if reference["dataset"] != "steubk/wikiart" or reference["dataset_version"] != 1:
        raise ValueError("Expected the original steubk/wikiart version 1 manifest.")
    files, hashes, sizes = reference["files"], reference["content_sha256"], reference["original_sizes"]
    if not files or len(files) != len(hashes) or len(files) != len(sizes) or len(set(files)) != len(files):
        raise ValueError("Original inventory names/hash/size lengths or unique names are invalid.")
    raw_fingerprint = _sha256_bytes(json.dumps(list(zip(files, hashes)), separators=(",", ":")).encode())
    if raw_fingerprint != reference["fingerprint"]:
        raise ValueError("Original raw inventory fingerprint does not match its names/content hashes.")
    if set(reference["splits"]) != {"train", "val", "test"}:
        raise ValueError("Original manifest must have exactly train/val/test splits.")
    flat = [index for split in ("train", "val", "test") for index in reference["splits"][split]]
    if len(flat) != len(set(flat)) or any(type(index) is not int or not 0 <= index < len(files) for index in flat):
        raise ValueError("Original split indices overlap or fall outside the raw inventory.")
    excluded = {item["file"] for item in reference.get("failures", [])}
    if not excluded.issubset(set(files)):
        raise ValueError("Original decode exclusions contain names outside the raw inventory.")
    if set(flat) != {index for index, name in enumerate(files) if name not in excluded}:
        raise ValueError("Original split coverage differs from its declared decode exclusions.")
    return raw_fingerprint


def _start_worker(path):
    global _archive
    _archive = zipfile.ZipFile(path)


def _process_original(item):
    index, name, expected_hash, expected_size, excluded = item
    raw = _archive.read(name)
    digest = _sha256_bytes(raw)
    if digest != expected_hash:
        raise ValueError(f"Original raw SHA256 mismatch at index {index}: {name}.")
    if excluded:
        return index, None, digest
    try:
        pixels, original_size = preprocess_raw_image(raw)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"Previously usable original image cannot be decoded at 128: {name}: {error}") from error
    if original_size != expected_size:
        raise ValueError(f"Original dimensions differ at index {index}: {name}: {original_size} vs {expected_size}.")
    return index, pixels, digest


def _bounded_results(pool, items, batch_size):
    # map only a bounded window: no unbounded queue of decoded RGB128 arrays.
    for start in range(0, len(items), batch_size):
        yield from pool.map(_process_original, items[start:start+batch_size], chunksize=1)


def prepare(archive, reference_manifest, output, workers=2):
    require_compute()
    if workers < 1:
        raise ValueError("Use a positive worker count inside the allocated memory budget.")
    archive, reference_manifest, output = Path(archive), Path(reference_manifest), Path(output)
    reference_bytes = reference_manifest.read_bytes()
    reference = json.loads(reference_bytes)
    raw_fingerprint = validate_reference(reference)
    reference_sha256 = _sha256_bytes(reference_bytes)
    if reference.get("archive_bytes") is not None and archive.stat().st_size != reference["archive_bytes"]:
        raise ValueError("ZIP size differs from the recorded original archive; verify original artifact provenance.")
    with zipfile.ZipFile(archive) as source:
        if archive_inventory(source) != reference["files"]:
            raise ValueError("ZIP inventory/order differs from the exact original manifest.")
    source_bytes = Path(__file__).read_bytes()
    identity = preprocessing_identity(source_bytes)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing/partial RGB128 artifacts; use a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    lock = output/".prepare128.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o660)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, handle)
    start = time.monotonic()
    processed = 0
    try:
        # This deliberate ordinary array needs N*128*128*3 bytes resident. At the
        # assigned inventory size it is about 4.0 GB, plus raw image worker memory.
        count = len(reference["files"])
        array_bytes = count*128*128*3
        estimate = array_bytes+4*len(reference_bytes)+1024*1024
        if shutil.disk_usage(output).free < estimate:
            raise OSError(f"Need at least {estimate} free output bytes for RGB128 array/metadata.")
        images = np.empty((count, 128, 128, 3), dtype=np.uint8)
        excluded = {item["file"] for item in reference.get("failures", [])}
        items = [(index, name, reference["content_sha256"][index], reference["original_sizes"][index], name in excluded)
                 for index, name in enumerate(reference["files"])]
        with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker,
                                 initargs=(str(archive),)) as pool:
            for index, pixels, digest in _bounded_results(pool, items, batch_size=workers*8):
                images[index] = 0 if pixels is None else pixels
                processed += 1
                if processed % 2000 == 0:
                    print(f"Prepared/verified {processed}/{count} original RGB128 images; {time.monotonic()-start:.1f}s", flush=True)
        temporary = output/f"images.npy.{uuid.uuid4().hex}.tmp"
        with temporary.open("xb") as handle:
            np.save(handle, images, allow_pickle=False)
            handle.flush(); os.fsync(handle.fileno())
        del images
        temporary.replace(output/"images.npy")
        image_sha256 = file_sha256(output/"images.npy")
        if Path(__file__).read_bytes() != source_bytes:
            raise RuntimeError("Preprocessor source changed during execution; preserve partial artifact and audit ownership.")
        (output/"preprocessing_source.py").write_bytes(source_bytes)
        (output/"source_rgb64_manifest.json").write_bytes(reference_bytes)
        payload = {"source_raw_fingerprint": raw_fingerprint,
                   "source_manifest_sha256": reference_sha256,
                   "source_split_sha256": _canonical_hash(reference["splits"]),
                   "image_size": 128, "preprocessing_identity": identity,
                   "images_npy_sha256": image_sha256}
        fingerprint = _canonical_hash(payload)
        style_counts = {split: dict(Counter(Path(reference["files"][index]).parts[0] for index in indices))
                        for split, indices in reference["splits"].items()}
        if reference.get("style_counts") is not None and style_counts != reference["style_counts"]:
            raise ValueError("Copied raw names/splits disagree with original style-count metadata.")
        manifest = {key: reference[key] for key in ["dataset", "dataset_version", "files", "content_sha256",
                    "original_sizes", "splits"]}
        manifest.update({"fingerprint": fingerprint, "cache_fingerprint_payload": payload,
            "source_raw_fingerprint": raw_fingerprint, "source_manifest_sha256": reference_sha256,
            "source_manifest_path": str(reference_manifest.resolve()), "archive": str(archive.resolve()),
            "archive_bytes": archive.stat().st_size, "image_size": 128,
            "preprocessing_protocol_version": PROTOCOL_VERSION, "preprocessing_identity": identity,
            "preprocessing": "Original raw RGB; bilinear shorter-edge resize to 128; rounded center crop 128x128; uint8 NHWC; training[-1,1]",
            "preprocessing_source_sha256": identity["preprocessor_file_sha256"],
            "images_npy_sha256": image_sha256, "images_shape": [count, 128, 128, 3],
            "images_dtype": "uint8", "images_array_bytes": array_bytes,
            "split_seed": reference.get("split_seed"), "split_rule": reference.get("split_rule"),
            "style_counts": style_counts, "failures": reference.get("failures", []),
            "limitations": reference.get("limitations"),
            "rgb128_demo_scope": "Native RGB128 data/model demonstration, separate from RGB64 campaign ranking.",
            "raw_hashes_verified": processed, "seconds": time.monotonic()-start})
        save_json(output/"summary.json", {key: value for key, value in manifest.items()
            if key not in ["files", "content_sha256", "original_sizes", "splits"]} |
            {"inventory_images": count, "split_counts": {key: len(value) for key, value in reference["splits"].items()}})
        # Manifest is committed last and is the completion marker.
        save_json(output/"manifest.json", manifest)
        print("Completed native RGB128 cache:", fingerprint, manifest["images_shape"], flush=True)
        return manifest
    except BaseException as error:
        save_json(output/"preparation_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "raw_images_processed": processed, "source_manifest_sha256": reference_sha256,
            "source_raw_fingerprint": raw_fingerprint, "preprocessing_identity": identity,
            "seconds": time.monotonic()-start,
            "status": "post-commit error; inspect completed manifest" if (output/"manifest.json").exists()
                else "partial; no complete manifest committed"})
        raise
    finally:
        lock.unlink(missing_ok=True)


class CachedPaintings128(Dataset):
    def __init__(self, cache_dir, split, *, verify_data_sha256=False):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir/"manifest.json").read_text())
        manifest = self.manifest
        if manifest.get("image_size") != 128 or manifest.get("preprocessing_protocol_version") != PROTOCOL_VERSION:
            raise ValueError("This loader requires a completed native RGB128 cache, never an RGB64 cache.")
        if _canonical_hash(manifest["cache_fingerprint_payload"]) != manifest["fingerprint"]:
            raise ValueError("RGB128 cache identity metadata is inconsistent.")
        payload = manifest["cache_fingerprint_payload"]
        if payload["image_size"] != 128 or payload["preprocessing_identity"] != manifest["preprocessing_identity"] \
                or payload["images_npy_sha256"] != manifest["images_npy_sha256"] \
                or payload["source_raw_fingerprint"] != manifest["source_raw_fingerprint"] \
                or payload["source_manifest_sha256"] != manifest["source_manifest_sha256"] \
                or payload["preprocessing_identity"]["preprocessor_file_sha256"] != manifest["preprocessing_source_sha256"]:
            raise ValueError("RGB128 manifest fields disagree with their bound cache identity.")
        if payload["source_split_sha256"] != _canonical_hash(manifest["splits"]):
            raise ValueError("RGB128 split mapping changed from its bound original mapping.")
        raw_fingerprint = _sha256_bytes(json.dumps(list(zip(manifest["files"], manifest["content_sha256"])),
                                                  separators=(",", ":")).encode())
        if raw_fingerprint != payload["source_raw_fingerprint"]:
            raise ValueError("RGB128 raw inventory/hash metadata changed.")
        if file_sha256(self.cache_dir/"preprocessing_source.py") != manifest["preprocessing_source_sha256"]:
            raise ValueError("Stored RGB128 preprocessing source differs from its bound identity.")
        if file_sha256(self.cache_dir/"source_rgb64_manifest.json") != manifest["source_manifest_sha256"]:
            raise ValueError("Stored source RGB64 manifest differs from its bound identity.")
        original = json.loads((self.cache_dir/"source_rgb64_manifest.json").read_text())
        validate_reference(original)
        if any(manifest[key] != original[key] for key in ["files", "content_sha256", "original_sizes", "splits"]):
            raise ValueError("RGB128 raw inventory/sizes/splits differ from the original source manifest.")
        if verify_data_sha256 and file_sha256(self.cache_dir/"images.npy") != manifest["images_npy_sha256"]:
            raise ValueError("RGB128 pixels differ from the completed cache artifact.")
        self.indices = np.asarray(manifest["splits"][split], dtype=np.int64)
        # Ordinary np.load intentionally avoids unsupported CFS mmap operations.
        self.images = np.load(self.cache_dir/"images.npy", allow_pickle=False)
        if self.images.shape != (len(manifest["files"]), 128, 128, 3) or self.images.dtype != np.uint8:
            raise ValueError("RGB128 storage shape/dtype differs from the manifest.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pixels = np.array(self.images[self.indices[index]], copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    prepare(args.archive, args.reference_manifest, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
