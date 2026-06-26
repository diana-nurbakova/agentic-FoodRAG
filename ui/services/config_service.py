"""Study-config persistence (spec §6.3, §11.5 config_service.py).

The admin-editable study config (``admin_config``) is stored as JSON so changes
take effect across sessions and restarts **without redeployment** (spec §6.3):

  - **Primary:** ``config/study_config.json`` in the HF Dataset repo (when
    ``FOODRAG_DATASET_REPO`` + ``HF_TOKEN`` are set).
  - **Local fallback:** ``data/study_config.json`` (gitignored runtime state) so
    persistence works on the container even without a remote repo.

On load, the persisted config is deep-merged over ``default_admin_config()`` so
configs written by an older build still pick up newly-added default keys. All
remote I/O is best-effort and never raises into the UI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..admin_config import default_admin_config

REPO_PATH = "config/study_config.json"   # path inside the HF Dataset repo
_remote_pull_done = False                # pull remote at most once per process


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_path() -> Path:
    """Local config file (override with ``FOODRAG_STUDY_CONFIG`` — used by tests)."""
    env = os.environ.get("FOODRAG_STUDY_CONFIG")
    if env:
        return Path(env)
    return _repo_root() / "data" / "study_config.json"


def _repo_id() -> Optional[str]:
    return os.environ.get("FOODRAG_DATASET_REPO") or None


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or None


def remote_enabled() -> bool:
    return bool(_repo_id() and _hf_token())


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``. Dicts merge key-wise; any
    other type (incl. lists) is replaced wholesale by the override value."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ── load ──────────────────────────────────────────────────────────────────
def _read_local() -> Optional[dict]:
    path = local_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_remote() -> Optional[dict]:
    if not remote_enabled():
        return None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=_repo_id(), filename=REPO_PATH, repo_type="dataset",
                               token=_hf_token())
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing file / network / auth → fall back
        return None


def load_persisted() -> Optional[dict]:
    """Return the raw persisted config (local first, then remote once), or None."""
    global _remote_pull_done
    cfg = _read_local()
    if cfg is not None:
        return cfg
    if not _remote_pull_done:
        _remote_pull_done = True
        remote = _read_remote()
        if remote is not None:
            # cache remote locally so subsequent loads are fast/offline
            try:
                _write_local(remote)
            except Exception:  # noqa: BLE001
                pass
            return remote
    return None


def effective_config() -> dict:
    """Defaults deep-merged with any persisted config (used at session init)."""
    return deep_merge(default_admin_config(), load_persisted() or {})


# ── save ──────────────────────────────────────────────────────────────────
def _write_local(cfg: dict) -> None:
    path = local_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_remote(cfg: dict) -> tuple[bool, str]:
    if not remote_enabled():
        return False, "remote disabled"
    try:
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi(token=_hf_token())
        try:
            api.create_repo(repo_id=_repo_id(), repo_type="dataset", private=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, ensure_ascii=False)
                tmp = fh.name
            api.upload_file(path_or_fileobj=tmp, path_in_repo=REPO_PATH,
                            repo_id=_repo_id(), repo_type="dataset")
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def save_config(cfg: dict) -> tuple[bool, str]:
    """Persist the config locally (always) and to the HF repo (if configured).

    Returns (ok, message). Local-write failure is the only hard failure; remote
    failure is reported in the message but does not flip ``ok`` to False when the
    local write succeeded."""
    try:
        _write_local(cfg)
    except OSError as exc:
        return False, f"local write failed: {exc}"
    if remote_enabled():
        ok, msg = _write_remote(cfg)
        return True, (f"saved locally + pushed to {_repo_id()}" if ok
                      else f"saved locally; remote push failed ({msg})")
    return True, "saved locally (no dataset repo configured)"


def status() -> dict:
    return {
        "local_path": str(local_path()),
        "local_exists": local_path().exists(),
        "remote_enabled": remote_enabled(),
        "repo_id": _repo_id(),
        "repo_path": REPO_PATH,
    }
