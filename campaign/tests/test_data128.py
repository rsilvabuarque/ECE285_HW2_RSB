"""Tiny synthetic byte/image/identity checks; never call allocated prepare()."""

from copy import deepcopy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

from hw2_gan import data128

def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def check_decoder():
    for width, height, mode in [(9, 13, "L"), (14, 9, "RGB"), (17, 11, "RGBA")]:
        channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
        pixels = (np.arange(width*height*channels).reshape(height, width, channels) % 256).astype(np.uint8)
        image = Image.fromarray(pixels[:, :, 0] if channels == 1 else pixels, mode=mode)
        raw = encoded(image)
        observed, original_size = data128.preprocess_raw_image(raw)
        shape = (128, int(128*height/width)) if width <= height else (int(128*width/height), 128)
        expected = image.convert("RGB").resize(shape, Image.Resampling.BILINEAR)
        left, top = round((shape[0]-128)/2), round((shape[1]-128)/2)
        expected = np.asarray(expected.crop((left, top, left+128, top+128)))
        assert np.array_equal(observed, expected)
        assert original_size == [width, height]
        assert observed.shape == (128, 128, 3) and observed.dtype == np.uint8
    # Synthetic high-frequency image: direct128 has different pixels from a64
    # intermediate. No existing RGB64 cache is read or used by the draft.
    rows, columns = np.indices((211, 320))
    gray = ((rows+columns) % 2*255).astype(np.uint8)
    original = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), "RGB")
    native, _ = data128.preprocess_raw_image(encoded(original))
    low_shape = (int(64*320/211), 64)
    low = original.resize(low_shape, Image.Resampling.BILINEAR)
    left = round((low_shape[0]-64)/2)
    interpolated = np.asarray(low.crop((left, 0, left+64, 64)).resize((128, 128), Image.Resampling.BILINEAR))
    assert not np.array_equal(native, interpolated)
    print("PASS synthetic RGB/L/RGBA byte decoding, native128 shape/dtype, exact bilinear/crop rounding, differs from64 intermediate", flush=True)


def check_reference_and_identity():
    files = ["style/a.png", "style/b.png"]
    hashes = [hashlib.sha256(name.encode()).hexdigest() for name in files]
    fingerprint = hashlib.sha256(json.dumps(list(zip(files, hashes)), separators=(",", ":")).encode()).hexdigest()
    reference = {"dataset": "steubk/wikiart", "dataset_version": 1, "files": files,
                 "content_sha256": hashes, "fingerprint": fingerprint, "original_sizes": [[9, 13], [14, 9]],
                 "splits": {"train": [0], "val": [1], "test": []}, "failures": []}
    assert data128.validate_reference(reference) == fingerprint
    broken = deepcopy(reference)
    broken["content_sha256"][0] = "0"*64
    try:
        data128.validate_reference(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed raw hash was accepted.")
    broken = deepcopy(reference)
    broken["splits"]["test"] = [1]
    try:
        data128.validate_reference(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlapping original split indices were accepted.")
    source = Path(data128.__file__).read_bytes()
    identity = data128.preprocessing_identity(source)
    assert identity["image_size"] == 128
    assert identity["preprocessor_file_sha256"] == hashlib.sha256(source).hexdigest()
    payload = {"source_raw_fingerprint": fingerprint, "source_manifest_sha256": "1"*64,
               "source_split_sha256": data128._canonical_hash(reference["splits"]),
               "image_size": 128, "preprocessing_identity": identity, "images_npy_sha256": "2"*64}
    assert data128._canonical_hash(payload) != fingerprint
    print("PASS original raw fingerprint/split validation and distinct source-bound128 cache identity", flush=True)


if __name__ == "__main__":
    check_decoder()
    check_reference_and_identity()
    print("All tiny pure-byte/metadata checks passed; no archive preparation or dataset processing executed.", flush=True)
