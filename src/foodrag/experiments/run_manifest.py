"""Run manifests: snapshot the exact config behind every experiment run.

Each run writes ``data/runs/<run>/run_manifest.json`` recording the git sha,
locked model ids, key parameters and a timestamp, so a result can always be
traced back to what produced it.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from foodrag.config.settings import load_models_config, repo_root


def git_sha(short: bool = True) -> Optional[str]:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(args, cwd=repo_root(), capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root(), capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def runs_dir(data_path: str | Path = "data") -> Path:
    return Path(data_path) / "runs"


def write_run_manifest(
    run_name: str,
    params: Optional[dict[str, Any]] = None,
    data_path: str | Path = "data",
    extra: Optional[dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    cfg = load_models_config()
    manifest = {
        "run_name": run_name,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "models": {
            "llm": cfg.llm.get("model"),
            "embedding": cfg.embedding.get("model"),
            "embedding_dim": cfg.embedding.get("dim"),
            "judges": [m.get("model") for m in cfg.judge_models],
        },
        "params": params or {},
    }
    if extra:
        manifest.update(extra)

    out_dir = runs_dir(data_path) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return path
