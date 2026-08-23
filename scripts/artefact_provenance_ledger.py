"""S'.2 — Artefact provenance ledger.

The aggregate schema has eleven columns and none of them records what happened to
the evidence. Four separate commits rewrote published per-question records; one of
them substituted model output for a canned string and re-judged it. None of that
is representable in ``aggregate.csv``.

This reconstructs the missing channel from version control, which was not designed
to serve as one. For every tracked ``eval_results*/*_per_question.{parquet,pkl}``
it emits one row per commit that touched the file:

    file · commit · date · subject · rows before -> after · schema delta

Row counts come from parquet footer metadata where possible (cheap — no full read).
Pickles must be loaded, so they are read only when small enough.

    foodrag-ledger                    # markdown table to stdout
    foodrag-ledger --csv out.csv      # also write it
"""
from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

MAX_PKL_BYTES = 200 * 1024 * 1024


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


def tracked_artefacts() -> list[str]:
    out = sh("git", "ls-files", "eval_results", "eval_results_llama")
    return sorted(p for p in out.splitlines()
                  if "_per_question." in p and p.endswith((".parquet", ".pkl")))


def commits_for(path: str) -> list[tuple[str, str, str]]:
    out = sh("git", "log", "--follow", "--format=%H\x1f%ad\x1f%s", "--date=short", "--", path)
    rows = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows[::-1]          # oldest first


def blob(commit: str, path: str) -> Optional[bytes]:
    r = subprocess.run(["git", "show", f"{commit}:{path}"],
                       capture_output=True, check=False)
    return r.stdout if r.returncode == 0 and r.stdout else None


def describe(data: bytes, suffix: str) -> tuple[Optional[int], set[str]]:
    """(row_count, column_set) for one blob, as cheaply as the format allows."""
    if data is None:
        return None, set()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            f = pq.ParquetFile(io.BytesIO(data))
            return f.metadata.num_rows, set(f.schema_arrow.names)
        except Exception:
            return None, set()
    if len(data) > MAX_PKL_BYTES:
        return None, set()
    try:
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        df = pd.read_pickle(tmp)
        Path(tmp).unlink(missing_ok=True)
        return len(df), set(df.columns)
    except Exception:
        return None, set()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv")
    args = p.parse_args(argv)

    files = tracked_artefacts()
    print(f"# S'.2 — Artefact provenance ledger\n\ntracked per-question artefacts: {len(files)}\n")

    rows: list[dict] = []
    for path in files:
        cs = commits_for(path)
        print(f"\n## `{path}`  — {len(cs)} commit(s)\n")
        print("| commit | date | subject | rows | schema delta |")
        print("|---|---|---|---|---|")
        prev_n: Optional[int] = None
        prev_cols: set[str] = set()
        suffix = Path(path).suffix
        for sha, date, subj in cs:
            n, cols = describe(blob(sha, path), suffix)
            if prev_n is None:
                rowtxt = f"{n:,}" if n is not None else "?"
            elif n is None:
                rowtxt = "?"
            elif n == prev_n:
                rowtxt = f"{n:,}"
            else:
                rowtxt = f"**{prev_n:,} → {n:,}**"
            added, removed = cols - prev_cols, prev_cols - cols
            if not prev_cols:
                delta = f"({len(cols)} cols)"
            elif added or removed:
                bits = []
                if added:
                    bits.append("+" + ", ".join(sorted(added)))
                if removed:
                    bits.append("−" + ", ".join(sorted(removed)))
                delta = "**" + " · ".join(bits) + "**"
            else:
                delta = "—"
            print(f"| `{sha[:7]}` | {date} | {subj[:58]} | {rowtxt} | {delta} |")
            rows.append({"file": path, "commit": sha[:7], "date": date, "subject": subj,
                         "rows": n, "rows_prev": prev_n,
                         "cols_added": ";".join(sorted(added)),
                         "cols_removed": ";".join(sorted(removed))})
            prev_n, prev_cols = (n if n is not None else prev_n), (cols or prev_cols)

    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nledger -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
