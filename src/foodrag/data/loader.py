"""Validate, unzip, and download the datasets described by the manifest.

Typical use::

    from foodrag.data import DataLoader
    loader = DataLoader.from_manifest()
    loader.prepare()          # unzip local archives + download public files
    report = loader.validate()  # raise if anything required is still missing
    data = get_hummus(loader.data_location, loader.graph_location,
                      loader.additional_location, data_path="data")
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from foodrag.data.manifest import (
    ROLE_ADDITIONAL,
    ROLE_DATA,
    ROLE_GRAPH,
    Dataset,
    DataManifest,
    FileSpec,
)

# File status codes (ordered roughly best -> worst).
PRESENT = "present"
WRONG_SIZE = "present_wrong_size"
CHECKSUM_MISMATCH = "checksum_mismatch"
UNZIPPABLE = "unzippable"  # missing, but a local archive can produce it
DOWNLOADABLE = "downloadable"  # missing, but a public URL is available
MISSING = "missing"  # missing and not recoverable automatically

_RECOVERABLE = {UNZIPPABLE, DOWNLOADABLE}


@dataclass
class FileStatus:
    dataset: str
    name: str
    path: Path
    status: str
    required: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PRESENT

    @property
    def recoverable(self) -> bool:
        return self.status in _RECOVERABLE


@dataclass
class DirStatus:
    dataset: str
    path: Path
    status: str  # PRESENT | MISSING
    required: bool
    n_files: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PRESENT


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class DataLoader:
    def __init__(self, manifest: DataManifest):
        self.manifest = manifest

    @classmethod
    def from_manifest(
        cls, path: Optional[Path] = None, repo_root: Optional[Path] = None
    ) -> "DataLoader":
        return cls(DataManifest.load(path=path, repo_root=repo_root))

    # -- location accessors (feed get_hummus) -----------------------------
    @property
    def data_location(self) -> Path:
        return self.manifest.location(ROLE_DATA)

    @property
    def graph_location(self) -> Path:
        return self.manifest.location(ROLE_GRAPH)

    @property
    def additional_location(self) -> Path:
        return self.manifest.location(ROLE_ADDITIONAL)

    def hummus_kwargs(self) -> dict[str, str]:
        """Location kwargs for ``foodrag.utils.load_util.get_hummus``.

        Returns string paths (load_util builds ``Path(location, filename)``).
        """
        return {
            "data_location": str(self.data_location),
            "graph_location": str(self.graph_location),
            "additional_location": str(self.additional_location),
        }

    # -- status ------------------------------------------------------------
    def _file_status(self, ds: Dataset, spec: FileSpec, verify_checksums: bool) -> FileStatus:
        path = ds.file_path(spec)
        if path.exists():
            if spec.size is not None and path.stat().st_size != spec.size:
                return FileStatus(
                    ds.key, spec.name, path, WRONG_SIZE, spec.required,
                    f"expected {spec.size} bytes, found {path.stat().st_size}",
                )
            if verify_checksums and spec.sha256:
                actual = sha256_of(path)
                if actual.lower() != spec.sha256.lower():
                    return FileStatus(
                        ds.key, spec.name, path, CHECKSUM_MISMATCH, spec.required,
                        f"sha256 {actual} != expected {spec.sha256}",
                    )
            return FileStatus(ds.key, spec.name, path, PRESENT, spec.required)

        archive = ds.archive_path(spec)
        if archive and archive.exists() and _archive_has_member(archive, spec.name):
            return FileStatus(
                ds.key, spec.name, path, UNZIPPABLE, spec.required,
                f"can extract from {archive.name}",
            )
        if spec.download_url:
            return FileStatus(
                ds.key, spec.name, path, DOWNLOADABLE, spec.required,
                f"can download from {spec.download_url}",
            )
        return FileStatus(ds.key, spec.name, path, MISSING, spec.required)

    def _dir_status(self, ds: Dataset) -> DirStatus:
        if not ds.dir.is_dir():
            return DirStatus(ds.key, ds.dir, MISSING, True, 0, "directory does not exist")
        n = sum(1 for p in ds.dir.iterdir() if p.is_file())
        if n < ds.min_files:
            return DirStatus(
                ds.key, ds.dir, MISSING, True, n,
                f"found {n} files, need at least {ds.min_files}",
            )
        return DirStatus(ds.key, ds.dir, PRESENT, True, n)

    def status(self, verify_checksums: bool = False) -> tuple[list[FileStatus], list[DirStatus]]:
        files: list[FileStatus] = []
        dirs: list[DirStatus] = []
        for ds in self.manifest.datasets.values():
            if ds.kind == "directory" and not ds.files:
                dirs.append(self._dir_status(ds))
            else:
                if ds.kind == "directory":
                    dirs.append(self._dir_status(ds))
                for spec in ds.files:
                    files.append(self._file_status(ds, spec, verify_checksums))
        return files, dirs

    # -- prepare (unzip + download) ---------------------------------------
    def prepare(self, unzip: bool = True, download: bool = True, verbose: bool = True) -> list[FileStatus]:
        """Recover what can be recovered automatically; return post-prepare status."""
        files, _ = self.status()
        for fs in files:
            if fs.status == UNZIPPABLE and unzip:
                ds = self.manifest.datasets[fs.dataset]
                spec = _find_spec(ds, fs.name)
                archive = ds.archive_path(spec)
                if verbose:
                    print(f"[unzip] {archive.name} -> {fs.path}")
                _extract_member(archive, spec.name, ds.dir)
            elif fs.status == DOWNLOADABLE and download:
                ds = self.manifest.datasets[fs.dataset]
                spec = _find_spec(ds, fs.name)
                if verbose:
                    print(f"[download] {spec.download_url} -> {fs.path}")
                _download(spec.download_url, fs.path, verbose=verbose)
        files, _ = self.status()
        return files

    # -- validate ----------------------------------------------------------
    def validate(self, require_optional: bool = False, verify_checksums: bool = False) -> "ValidationReport":
        files, dirs = self.status(verify_checksums=verify_checksums)
        problems: list[str] = []
        for fs in files:
            if fs.ok:
                continue
            if not fs.required and not require_optional:
                continue
            ds = self.manifest.datasets[fs.dataset]
            hint = _acquisition_hint(ds, fs)
            problems.append(f"[{fs.dataset}] {fs.name}: {fs.status} ({fs.detail}). {hint}")
        for dstat in dirs:
            if not dstat.ok:
                ds = self.manifest.datasets[dstat.dataset]
                problems.append(
                    f"[{dstat.dataset}] dir {dstat.path}: {dstat.status} ({dstat.detail}). "
                    f"{_source_hint(ds)}"
                )
        return ValidationReport(files=files, dirs=dirs, problems=problems)

    def require(self, **kwargs) -> None:
        report = self.validate(**kwargs)
        if not report.ok:
            raise FileNotFoundError(
                "Data validation failed:\n  - " + "\n  - ".join(report.problems)
            )


@dataclass
class ValidationReport:
    files: list[FileStatus]
    dirs: list[DirStatus]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def format(self) -> str:
        lines = ["Data manifest status:"]
        for fs in self.files:
            mark = "OK " if fs.ok else ("?? " if fs.recoverable else "XX ")
            opt = "" if fs.required else " (optional)"
            lines.append(f"  {mark}[{fs.dataset}] {fs.name}: {fs.status}{opt}")
        for d in self.dirs:
            mark = "OK " if d.ok else "XX "
            lines.append(f"  {mark}[{d.dataset}] {d.path.name}/: {d.status} ({d.n_files} files)")
        if self.problems:
            lines.append("\nUnresolved:")
            lines.extend(f"  - {p}" for p in self.problems)
        return "\n".join(lines)


# -- helpers ---------------------------------------------------------------
def _find_spec(ds: Dataset, name: str) -> FileSpec:
    for spec in ds.files:
        if spec.name == name:
            return spec
    raise KeyError(f"{name!r} not found in dataset {ds.key!r}")


def _archive_has_member(archive: Path, name: str) -> bool:
    try:
        with zipfile.ZipFile(archive) as zf:
            return _member_for(zf, name) is not None
    except (zipfile.BadZipFile, OSError):
        return False


def _member_for(zf: zipfile.ZipFile, name: str) -> Optional[str]:
    for member in zf.namelist():
        if member == name or Path(member).name == name:
            return member
    return None


def _extract_member(archive: Path, name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        member = _member_for(zf, name)
        if member is None:
            raise KeyError(f"{name!r} not found inside {archive}")
        target = dest_dir / name
        with zf.open(member) as src, open(target, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    return target


def _download(url: str, dest: Path, verbose: bool = True) -> Path:
    import requests  # local import; only needed when actually downloading

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if verbose and total:
                    pct = 100 * done / total
                    print(f"\r    {dest.name}: {pct:5.1f}%", end="", flush=True)
    if verbose and total:
        print()
    tmp.replace(dest)
    return dest


def _source_hint(ds: Dataset) -> str:
    bits = []
    if ds.source.url:
        bits.append(f"Source: {ds.source.url}")
    if ds.source.note:
        bits.append(ds.source.note)
    return " ".join(bits) if bits else "No source documented."


def _acquisition_hint(ds: Dataset, fs: FileStatus) -> str:
    if fs.status == UNZIPPABLE:
        return "Run `prepare()` to extract it from the local archive."
    if fs.status == DOWNLOADABLE:
        return "Run `prepare()` to download it."
    return _source_hint(ds)


def iter_required(loader: DataLoader) -> Iterable[FileStatus]:
    files, _ = loader.status()
    return (fs for fs in files if fs.required)
