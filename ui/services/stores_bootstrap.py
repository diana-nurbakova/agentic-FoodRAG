"""Bring the backing stores up when the app launches (spec §2.2 infra).

Best-effort ``docker compose up -d`` for the pgvector Postgres (and Neo4j) so the
live retrievers (Dense/Sparse/Hybrid/Fusion) work without the user starting Docker
by hand. Idempotent and always non-fatal: if Docker is missing or the daemon is
down, the app still runs (mock data + the Text→Pandas retriever need no database).

Call ``ensure_stores_up()`` once per server process — app.py wraps it in
``st.cache_resource`` so it runs a single time regardless of Streamlit reruns.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path


def _repo_root() -> Path:
    """Directory holding docker-compose.yml (project root)."""
    here = Path(__file__).resolve().parents[2]  # ui/services/ -> project root
    if (here / "docker-compose.yml").exists():
        return here
    try:
        from foodrag.config.settings import repo_root

        cand = Path(repo_root())
        if (cand / "docker-compose.yml").exists():
            return cand
    except Exception:  # noqa: BLE001
        pass
    return here


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def stores_reachable() -> bool:
    """Cheap live check: is Postgres actually accepting connections right now?

    Independent of whether this process can invoke the Docker CLI — if the containers
    are up, this is True even when ``docker`` isn't on PATH."""
    return _port_open("localhost", int(os.environ.get("POSTGRES_PORT", "5432")))


def ensure_stores_up(wait_seconds: float = 40.0) -> dict:
    """Ensure the pgvector Postgres (and Neo4j) are running via docker compose.

    Returns ``{"ok": bool, "started": bool, "msg": str}``. Never raises."""
    # Never spawn containers under a test run or when explicitly disabled.
    if os.environ.get("FOODRAG_SKIP_STORES_BOOTSTRAP") or "PYTEST_CURRENT_TEST" in os.environ:
        return {"ok": False, "started": False, "msg": "Store bootstrap skipped (test / disabled)."}

    # In-container deploy (e.g. HF Spaces): Postgres + Neo4j are launched by the
    # container's process manager (supervisord), not `docker compose`. There is no
    # Docker daemon here, so short-circuit — the stores are already coming up on
    # localhost and the app just connects to them via POSTGRES_*/NEO4J_* env vars.
    if os.environ.get("FOODRAG_IN_CONTAINER"):
        return {"ok": True, "started": False,
                "msg": "In-container stores managed by supervisord; docker compose skipped."}

    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    if _port_open("localhost", port):
        return {"ok": True, "started": False, "msg": f"Postgres already reachable on :{port}."}

    root = _repo_root()
    if not (root / "docker-compose.yml").exists():
        return {"ok": False, "started": False, "msg": "docker-compose.yml not found; start Postgres manually."}

    # Docker daemon available?
    try:
        subprocess.run(["docker", "info"], cwd=str(root),
                       capture_output=True, timeout=20, check=True)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"ok": False, "started": False,
                "msg": "Docker isn't available — start Docker Desktop, or use the Text→Pandas retriever."}

    # Bring up the core stores (postgres required; neo4j optional for KG fusion).
    try:
        subprocess.run(["docker", "compose", "up", "-d", "postgres", "neo4j"],
                       cwd=str(root), capture_output=True, timeout=300, check=True)
    except subprocess.SubprocessError as exc:
        detail = getattr(exc, "stderr", b"")
        detail = detail.decode("utf-8", "replace")[-300:] if isinstance(detail, bytes) else str(exc)
        return {"ok": False, "started": False, "msg": f"`docker compose up` failed: {detail}"}

    # Wait for Postgres to accept TCP connections (first boot needs a few seconds).
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _port_open("localhost", port):
            return {"ok": True, "started": True, "msg": f"Started backing stores; Postgres up on :{port}."}
        time.sleep(1.5)
    return {"ok": False, "started": True,
            "msg": "Containers starting but Postgres isn't ready yet — retry the run shortly."}
