"""Unit tests for the manifest-driven data loader (no real data needed)."""
from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest

from foodrag.data import DataLoader
from foodrag.data.loader import DOWNLOADABLE, MISSING, PRESENT, UNZIPPABLE

pytestmark = pytest.mark.unit


def _write_manifest(tmp: Path, body: str) -> Path:
    path = tmp / "manifest.yml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _loader(tmp: Path, body: str) -> DataLoader:
    return DataLoader.from_manifest(path=_write_manifest(tmp, body), repo_root=tmp)


def _status_map(loader: DataLoader) -> dict[str, str]:
    files, _ = loader.status()
    return {fs.name: fs.status for fs in files}


def test_present_file_is_ok(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "recipes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          hummus_raw:
            role: data_location
            dir: raw
            files:
              - name: recipes.csv
        """,
    )
    assert _status_map(loader)["recipes.csv"] == PRESENT
    assert loader.validate().ok


def test_unzippable_then_prepare_extracts(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "recipes.zip", "w") as zf:
        zf.writestr("recipes.csv", "a,b\n1,2\n")
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          hummus_raw:
            role: data_location
            dir: raw
            files:
              - name: recipes.csv
                unzip_from: recipes.zip
        """,
    )
    assert _status_map(loader)["recipes.csv"] == UNZIPPABLE
    assert not loader.validate().ok  # not yet on disk

    loader.prepare(download=False, verbose=False)

    assert (raw / "recipes.csv").exists()
    assert _status_map(loader)["recipes.csv"] == PRESENT
    assert loader.validate().ok


def test_missing_required_fails_validate(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          hummus_raw:
            role: data_location
            dir: raw
            files:
              - name: recipes.csv
        """,
    )
    assert _status_map(loader)["recipes.csv"] == MISSING
    report = loader.validate()
    assert not report.ok
    assert any("recipes.csv" in p for p in report.problems)
    with pytest.raises(FileNotFoundError):
        loader.require()


def test_optional_missing_is_tolerated(tmp_path: Path):
    (tmp_path / "g").mkdir()
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          foodkg:
            role: graph_location
            dir: g
            files:
              - name: food_kg_to_usda.pkl
                required: false
        """,
    )
    assert loader.validate().ok  # optional missing -> still ok
    assert not loader.validate(require_optional=True).ok


def test_wrong_size_is_flagged(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "recipes.csv").write_text("x", encoding="utf-8")
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          hummus_raw:
            role: data_location
            dir: raw
            files:
              - name: recipes.csv
                size: 999999
        """,
    )
    assert not loader.validate().ok


def test_downloadable_status_without_network(tmp_path: Path):
    (tmp_path / "g").mkdir()
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          foodkg:
            role: graph_location
            dir: g
            files:
              - name: foodon.owl
                download_url: "https://example.org/foodon.owl"
        """,
    )
    assert _status_map(loader)["foodon.owl"] == DOWNLOADABLE


def test_hummus_kwargs_resolves_roles(tmp_path: Path):
    for sub in ("raw", "g", "add"):
        (tmp_path / sub).mkdir()
    loader = _loader(
        tmp_path,
        """
        version: 1
        datasets:
          hummus_raw:
            role: data_location
            dir: raw
            files: []
          foodkg:
            role: graph_location
            dir: g
            files: []
          foodcom_tags:
            role: additional_location
            dir: add
            files: []
        """,
    )
    kw = loader.hummus_kwargs()
    assert kw["data_location"].endswith("raw")
    assert kw["graph_location"].endswith("g")
    assert kw["additional_location"].endswith("add")
