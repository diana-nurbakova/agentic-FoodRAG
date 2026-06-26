"""Loader for pre-computed evaluation results (Evaluation Protocol §11.2, §13).

The Report tab (Flow C) reads versioned result tables — it does NOT compute
anything live (§11.2). Results live in ``eval_results/`` (or ``FOODRAG_EVAL_RESULTS``):

  - ``aggregate.csv``  — §13.2 per (experiment × config × stratum × metric)
  - ``pairwise.csv``   — §13.3 significance markers
  - ``manifest.json``  — {"sample": bool, ...} provenance

Files are cached read-only. Panels render conditionally on what's present
(§12.3: minimum viable = EXP-1 + EXP-3 + Safety).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


def results_dir() -> Path:
    env = os.environ.get("FOODRAG_EVAL_RESULTS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "eval_results"


@lru_cache(maxsize=2)
def _read_csv(name: str) -> Optional[pd.DataFrame]:
    path = results_dir() / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def load_aggregate() -> Optional[pd.DataFrame]:
    return _read_csv("aggregate.csv")


def load_pairwise() -> Optional[pd.DataFrame]:
    return _read_csv("pairwise.csv")


def manifest() -> dict:
    path = results_dir() / "manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def is_sample() -> bool:
    return bool(manifest().get("sample", True))


def real_experiments() -> list[str]:
    """Experiment ids whose rows are real (not synthetic placeholders)."""
    return list(manifest().get("real_experiments", []))


def available() -> bool:
    return load_aggregate() is not None


def clear_cache() -> None:
    _read_csv.cache_clear()
