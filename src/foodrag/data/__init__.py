"""Manifest-driven data loading for the Food RAG system.

Exposes :class:`DataLoader` (validate / unzip / download) and
:class:`DataManifest` (typed view of ``config/data_manifest.yml``).
"""
from foodrag.data.loader import DataLoader, ValidationReport
from foodrag.data.manifest import DataManifest, Dataset, FileSpec

__all__ = ["DataLoader", "DataManifest", "Dataset", "FileSpec", "ValidationReport"]
