from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, allow_nan=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Persist the rename as well as the file contents on supporting filesystems.
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                try:
                    os.fsync(directory)
                except OSError as exc:
                    if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                        raise
            finally:
                os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class ArtifactCache:
    """Keys include actual upstream content, not mutable filenames."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def get(self, stage: str, identity: dict, compute: Callable[[], Any],
            validate: Callable[[Any], Any] | None = None) -> Any:
        key = digest({"stage": stage, "identity": identity, "cache_version": 1})
        path = self.root / stage / f"{key}.json"
        if path.exists():
            try:
                value = json.loads(path.read_text())
                if validate:
                    validate(value)
                self.hits += 1
                return value
            except (ValueError, KeyError, TypeError):
                pass
        value = compute()
        if validate:
            validate(value)
        atomic_json(path, value)
        self.misses += 1
        return value
