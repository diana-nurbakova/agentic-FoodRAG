"""Bootstrap helper for the notebooks in ``notebooks/``.

Notebooks use repo-root-relative paths (``config/...``, ``./data``). Since they
now live in a subfolder, call :func:`setup` in the first cell so the working
directory is the repo root and .env is loaded::

    from foodrag.notebook import setup
    setup()                 # chdir to repo root + load .env
    # optional: also wire the global LLM/embedding
    setup(apply_models=True)
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup(apply_models: bool = False, chdir: bool = True, load_dotenv: bool = True) -> Path:
    """Prepare a notebook: chdir to repo root, load .env, optionally wire models.

    Returns the repo root path.
    """
    root = repo_root()
    if chdir and Path.cwd() != root:
        os.chdir(root)
    if load_dotenv:
        from foodrag.config.settings import load_env

        load_env()
    if apply_models:
        from foodrag.config.settings import apply_global_settings

        apply_global_settings()
    return root
