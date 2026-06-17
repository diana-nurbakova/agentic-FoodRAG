"""CLI: ``python -m foodrag.data {status|prepare|validate}``.

    status    show presence of every manifest file (no changes)
    prepare   unzip local archives + download public files, then show status
    validate  fail (non-zero exit) if anything required is still missing

Options: --manifest PATH, --verify-checksums, --require-optional,
--no-unzip, --no-download.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foodrag.data.loader import DataLoader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foodrag.data")
    parser.add_argument("command", choices=["status", "prepare", "validate"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--require-optional", action="store_true")
    parser.add_argument("--no-unzip", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    loader = DataLoader.from_manifest(path=args.manifest)

    if args.command == "prepare":
        loader.prepare(unzip=not args.no_unzip, download=not args.no_download)

    report = loader.validate(
        require_optional=args.require_optional,
        verify_checksums=args.verify_checksums,
    )
    print(report.format())

    if args.command == "validate" and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
