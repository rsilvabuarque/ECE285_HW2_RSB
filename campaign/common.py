"""Small immutable JSON helpers; no Torch imports or numerical work."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def freeze(path, value):
    """Create once, accepting an identical existing value for phase restart."""
    path = Path(path)
    if path.exists():
        if load(path) != value:
            raise FileExistsError(f"Frozen decision differs: {path}; use a fresh reproduction root.")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".freeze_")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)
