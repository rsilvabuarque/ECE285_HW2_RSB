"""Prepare a compact RGB cache from the exact assigned WikiArt archive."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import require_compute, save_json

_archive = None


def _start_worker(path):
    global _archive
    _archive = zipfile.ZipFile(path)


def _prepare_image(item):
    index, name = item
    raw = _archive.read(name)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width <= height:
                shape = (64, int(64*height/width))
            else:
                shape = (int(64*width/height), 64)
            image = image.resize(shape, Image.Resampling.BILINEAR)
            left = round((shape[0]-64)/2); top = round((shape[1]-64)/2)
            pixels = np.array(image.crop((left, top, left+64, top+64)), dtype=np.uint8)
        return index, name, pixels, digest, width, height, None
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return index, name, None, digest, 0, 0, str(error)


class CachedPaintings(Dataset):
    def __init__(self, cache_dir, split):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir/"manifest.json").read_text())
        self.indices = np.asarray(self.manifest["splits"][split], dtype=np.int64)
        # About 1 GB for this dataset; ordinary loading also works on mounts
        # without file-backed mmap support. Forked loader workers share it.
        self.images = np.load(self.cache_dir/"images.npy")

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        array = np.array(self.images[self.indices[index]], copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)


def prepare(archive, output, workers=8, seed=42):
    require_compute()
    output = Path(output)
    if (output/"manifest.json").exists():
        raise FileExistsError("A completed cache already exists; preserve it or choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        files = sorted(info.filename for info in source.infolist()
                       if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
                       and not any(part.startswith(".") or part=="__MACOSX" for part in Path(info.filename).parts))
    images = np.empty((len(files), 64, 64, 3), dtype=np.uint8)
    hashes = [None]*len(files); sizes = [None]*len(files); failures = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker,
                             initargs=(str(archive),)) as pool:
        for count, result in enumerate(pool.map(_prepare_image, enumerate(files), chunksize=16), 1):
            index, name, pixels, digest, width, height, error = result
            hashes[index] = digest; sizes[index] = [width, height]
            if error: failures.append({"file": name, "error": error})
            else: images[index] = pixels
            if count % 2000 == 0: print(f"Prepared {count}/{len(files)} paintings, {time.monotonic()-start:.1f}s", flush=True)
    with (output/"images.npy.tmp").open("wb") as handle:
        np.save(handle, images)
    del images
    (output/"images.npy.tmp").replace(output/"images.npy")
    bad = {item["file"] for item in failures}
    splits = {name: [] for name in ["train", "val", "test"]}
    for index, (name, digest) in enumerate(zip(files, hashes)):
        if name in bad: continue
        # Identical file bytes get the same split even under different filenames.
        u = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16], 16) / 2**64
        split = "train" if u < 0.8 else "val" if u < 0.9 else "test"
        splits[split].append(index)
    fingerprint = hashlib.sha256(json.dumps(list(zip(files, hashes)), separators=(",", ":")).encode()).hexdigest()
    style_counts = {split: dict(Counter(Path(files[i]).parts[0] for i in indices)) for split, indices in splits.items()}
    manifest = {"dataset": "steubk/wikiart", "dataset_version": 1,
                "archive": str(Path(archive).resolve()), "archive_bytes": Path(archive).stat().st_size,
                "files": files, "content_sha256": hashes, "original_sizes": sizes,
                "fingerprint": fingerprint, "splits": splits, "split_seed": seed,
                "split_rule": "SHA256(seed:raw-content-SHA256) first64bits / 2**64: train<.8,val<.9,test otherwise",
                "style_counts": style_counts, "failures": failures,
                "preprocessing": "RGB; bilinear shorter-edge resize to64; rounded center crop64x64; uint8 cache; training[-1,1]",
                "limitations": "Byte-identical files share a split; near-duplicates and artist independence are not audited.",
                "seconds": time.monotonic()-start}
    save_json(output/"manifest.json", manifest)
    save_json(output/"summary.json", {key: value for key,value in manifest.items()
              if key not in ["files", "content_sha256", "original_sizes", "splits"]} |
              {"inventory_images": len(files), "usable_images": sum(map(len,splits.values())),
               "split_counts": {key:len(value) for key,value in splits.items()},
               "unique_file_hashes": len(set(hashes))})
    print("Completed cache:", {key:len(value) for key,value in splits.items()}, "failures:", len(failures), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args()
    prepare(args.archive,args.output_dir,args.workers,args.seed)


if __name__=="__main__": main()
