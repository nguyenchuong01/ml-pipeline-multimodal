"""Small utilities: seeding, timing, hashing, robust downloading."""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import tarfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we use so that runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for tabular/text-only runs
        pass


@contextmanager
def timer():
    """Context manager that stores the elapsed seconds in ``t.seconds``."""

    class _T:
        seconds = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


def md5sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str | Path, expected_md5: str | None = None, retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` with ``wget`` (resumable) and verify the md5.

    A previously downloaded file is reused when its checksum matches, so running
    the notebook twice in the same Colab session does not download twice.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_md5 is None or md5sum(dest) == expected_md5):
        return dest
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["wget", "-q", "-c", url, "-O", str(dest)], check=True)
            if expected_md5 is not None and md5sum(dest) != expected_md5:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"md5 mismatch for {url}")
            return dest
        except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def safe_extract_tar(archive: str | Path, out_dir: str | Path) -> None:
    """Extract a tar archive with the ``data`` filter when the Python version supports it."""
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(out_dir, filter="data")
        except TypeError:  # Python < 3.12 without the filter argument
            tar.extractall(out_dir)


def safe_extract_zip(archive: str | Path, out_dir: str | Path) -> None:
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)
