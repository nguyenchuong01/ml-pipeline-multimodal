"""YAML configuration helpers.

Every experiment in this project is described by a YAML file in ``configs/``.
Notebooks only load a config, optionally override a few keys, and call the
modules. Changing the dataset options, the backbone or the classifier never
requires editing code.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of ``base`` updated recursively with ``overrides``.

    Dicts are merged key by key; every other value (including lists) is replaced.
    """
    out = copy.deepcopy(dict(base))
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the folder that contains ``modules/`` and ``configs/``."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "modules").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the project root (folder with modules/ and configs/).")
