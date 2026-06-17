"""Parsing of ``config/data_manifest.yml`` into typed objects.

The manifest is the single source of truth for every external data file the
pipeline needs. :class:`DataManifest` only models and resolves the manifest;
acquisition logic (validate / unzip / download) lives in :mod:`foodrag.data.loader`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Roles map a dataset directory onto the get_hummus(...) location parameters.
ROLE_DATA = "data_location"
ROLE_GRAPH = "graph_location"
ROLE_ADDITIONAL = "additional_location"


def _repo_root() -> Path:
    """Repository root, derived from this file's location (src/foodrag/data/)."""
    return Path(__file__).resolve().parents[3]


def default_manifest_path() -> Path:
    return _repo_root() / "config" / "data_manifest.yml"


@dataclass
class Source:
    name: Optional[str] = None
    url: Optional[str] = None
    public: bool = False
    note: Optional[str] = None


@dataclass
class FileSpec:
    name: str
    size: Optional[int] = None
    sha256: Optional[str] = None
    unzip_from: Optional[str] = None
    download_url: Optional[str] = None
    required: bool = True
    note: Optional[str] = None


@dataclass
class Dataset:
    key: str
    dir: Path
    description: str = ""
    role: Optional[str] = None
    kind: str = "files"  # "files" | "directory"
    min_files: int = 0  # for kind == "directory"
    source: Source = field(default_factory=Source)
    files: list[FileSpec] = field(default_factory=list)

    def file_path(self, spec: FileSpec) -> Path:
        return self.dir / spec.name

    def archive_path(self, spec: FileSpec) -> Optional[Path]:
        return (self.dir / spec.unzip_from) if spec.unzip_from else None


@dataclass
class DataManifest:
    path: Path
    repo_root: Path
    datasets: dict[str, Dataset]

    @classmethod
    def load(cls, path: Optional[Path] = None, repo_root: Optional[Path] = None) -> "DataManifest":
        repo_root = Path(repo_root).resolve() if repo_root else _repo_root()
        path = Path(path).resolve() if path else default_manifest_path()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        datasets: dict[str, Dataset] = {}
        for key, d in (raw.get("datasets") or {}).items():
            src = d.get("source") or {}
            files = [
                FileSpec(
                    name=f["name"],
                    size=f.get("size"),
                    sha256=f.get("sha256"),
                    unzip_from=f.get("unzip_from"),
                    download_url=f.get("download_url"),
                    required=f.get("required", True),
                    note=f.get("note"),
                )
                for f in (d.get("files") or [])
            ]
            # Directory paths in the manifest are relative to the repo root.
            dir_path = Path(d["dir"])
            if not dir_path.is_absolute():
                dir_path = repo_root / dir_path
            datasets[key] = Dataset(
                key=key,
                dir=dir_path,
                description=d.get("description", ""),
                role=d.get("role"),
                kind=d.get("kind", "files"),
                min_files=d.get("min_files", 0),
                source=Source(
                    name=src.get("name"),
                    url=src.get("url"),
                    public=src.get("public", False),
                    note=src.get("note"),
                ),
                files=files,
            )
        return cls(path=path, repo_root=repo_root, datasets=datasets)

    def by_role(self, role: str) -> Dataset:
        for ds in self.datasets.values():
            if ds.role == role:
                return ds
        raise KeyError(f"No dataset in the manifest has role {role!r}.")

    def location(self, role: str) -> Path:
        """Resolved directory for a get_hummus location role."""
        return self.by_role(role).dir
