"""foodrag command-line entrypoint.

Reproducible pipeline commands are implemented in WS5. This is a minimal
dispatcher stub so the `foodrag` console script is valid after install.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "foodrag CLI (work in progress).\n"
            "Planned commands: init-data, embed, init-vector-store, "
            "init-graph-stores, gen-qa, run-agents, eval, bench-embeddings."
        )
        return 0
    print(f"Unknown or not-yet-implemented command: {argv[0]!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
