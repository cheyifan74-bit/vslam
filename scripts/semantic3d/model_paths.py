"""Robust model-directory resolution for the semantic/3D nodes.

The same node runs from two layouts:
  source : <repo>/scripts/*.py            -> <repo>/data/models/<name>
  install: <ws>/install/vslam/lib/vslam/* -> <ws>/src/vslam/data/models/<name>

We do not copy multi-GB weights into install/, so resolve by searching the
ancestors of this file for a data/models dir, plus an env override:
  VSLAM_MODEL_DIR=/path/to/data/models  (or /path/to/data)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

_PATTERNS = (
    ("data", "models"),
    ("data", "model"),
    ("data",),
    ("src", "vslam", "data", "models"),
    ("src", "vslam", "data"),
)


def candidate_dirs() -> List[Path]:
    here = Path(__file__).resolve()
    out: List[Path] = []
    env = os.environ.get("VSLAM_MODEL_DIR") or os.environ.get("VSLAM_DATA_DIR")
    if env:
        out.append(Path(env))
    for a in here.parents:
        for pat in _PATTERNS:
            out.append(a.joinpath(*pat))
    seen = set()
    res: List[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            res.append(p)
    return res


def find_model_dir(name: str) -> Optional[Path]:
    for base in candidate_dirs():
        p = base / name
        if p.is_dir():
            return p
    return None


def default_siglip_dir() -> Path:
    found = find_model_dir("siglip2-so400m-patch14-384")
    if found is not None:
        return found
    return Path(__file__).resolve().parents[1] / "data" / "models" / "siglip2-so400m-patch14-384"


def _find_file_in_sam3_dirs(filename: str) -> Optional[str]:
    for base in candidate_dirs():
        for d in (base / "sam3", base):
            p = d / filename
            if p.is_file():
                return str(p)
    return None


def find_sam3_weights() -> Optional[str]:
    return _find_file_in_sam3_dirs("sam3.pt")


def find_sam3_bpe() -> Optional[str]:
    return _find_file_in_sam3_dirs("bpe_simple_vocab_16e6.txt.gz")
