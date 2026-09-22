"""Read a supplied download script as data; never execute it or log signed URLs."""
from __future__ import annotations

import os
import json
import http.client
import re
import shlex
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .cache import atomic_json, digest, file_hash
from .media import probe


def entries(path: str | Path) -> list[tuple[str, str]]:
    found = []
    names = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip().startswith("curl "):
            continue
        parts = shlex.split(line)
        if "-o" not in parts:
            continue
        name = parts[parts.index("-o") + 1]
        urls = [p for p in parts if p.startswith("https://")]
        if not re.fullmatch(r"[\w.-]+\.(mp4|webm|mov)", name) or len(urls) != 1:
            raise ValueError("Unsafe filename or ambiguous download URL")
        if urllib.parse.urlparse(urls[0]).hostname != "storage.googleapis.com":
            raise ValueError("Expected Code Four's Google Cloud Storage download host")
        # Case-folding also protects the default macOS case-insensitive filesystem.
        if name.casefold() in names:
            raise ValueError(f"Duplicate download filename: {name}")
        names.add(name.casefold())
        found.append((name, urls[0]))
    if not found:
        raise ValueError("No supported download entries found")
    return found


def verified_existing(target: Path, receipt: Path, source_id: str) -> bool:
    """Reuse only a file from a validated transfer whose bytes are still intact."""
    try:
        data = json.loads(receipt.read_text())
        return data.get("version") == 1 and data.get("source_id") == source_id and \
            target.is_file() and target.stat().st_size > 0 and target.stat().st_size == data.get("size") and \
            file_hash(target) == data.get("sha256")
    except (OSError, ValueError, AttributeError):
        return False


def download(script: str, directory: str, names: list[str]) -> list[str]:
    source = dict(entries(script))
    if not names or set(names) - source.keys():
        raise ValueError("Specify one or more filenames present in the download script")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    completed = []
    for name in names:
        target = root / name
        receipt = target.with_suffix(target.suffix + ".download.json")
        source_id = digest(source[name])
        if verified_existing(target, receipt, source_id):
            completed.append(str(target))
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with urllib.request.urlopen(source[name], timeout=60) as response, open(partial, "wb") as out:
                if response.status != 200:
                    raise RuntimeError(f"Download failed: HTTP {response.status}")
                if "xml" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Server returned an error document instead of video")
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected is not None else None
                size = 0
                while block := response.read(1024*1024):
                    out.write(block)
                    size += len(block)
            if not size or (expected is not None and size != expected):
                raise RuntimeError(f"Incomplete download for {name}")
            # A nonempty error document must never be published as a completed video.
            probe(partial)
            metadata = {"version": 1, "source_id": source_id, "size": size, "sha256": file_hash(partial)}
            os.replace(partial, target)
            atomic_json(receipt, metadata)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Download failed for {name}: HTTP {exc.code}. Signed URLs may have expired.") from None
        except urllib.error.URLError:
            raise RuntimeError(f"Network unavailable while downloading {name}; signed URL omitted.") from None
        except (http.client.HTTPException, TimeoutError):
            raise RuntimeError(f"Interrupted download for {name}; signed URL omitted.") from None
        finally:
            partial.unlink(missing_ok=True)
        completed.append(str(target))
    return completed
