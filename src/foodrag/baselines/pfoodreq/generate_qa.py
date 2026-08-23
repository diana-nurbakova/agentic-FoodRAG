"""Drive the pFoodReq data pipeline (Path A): HUMMUS → KB → HUMMUS-derived KBQA.

Two steps, both torch-free:

1. :func:`build_recipe_kg` (this package) — HUMMUS recipes → ``recipe_kg.json``.
2. the **vendored** ``data_builder/src/generate_all_qa.py`` (Apache-2.0, see
   ``third_party/pfoodreq/NOTICE.md``) — KB → ``train/valid/test_qas.json``.

This is the faithful-retrain path: BAMnet trains on questions pFoodReq
itself generates over HUMMUS recipes, rather than on a lossy projection of our
QA set. The output ``*_qas.json`` are the inputs to the (still-to-be-ported)
BAMnet ``build_all_data.py`` → ``train.py`` stage (sub-task a).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from foodrag.baselines.pfoodreq.build_recipe_kg import build_recipe_kg
from foodrag.baselines.pfoodreq.validate import validate_recipe_kg


def _generator_dir() -> Path:
    from foodrag.config.settings import repo_root
    return repo_root() / "third_party" / "pfoodreq" / "data_builder" / "src"


def generate_kbqa(
    recipes_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    recipe_filter: Optional[set[str]] = None,
    max_dishes_per_tag: Optional[int] = 500,
    out_of_domain_ratio: float = 0.1,
    split_ratio: tuple[float, float] = (0.6, 0.2),
    validate: bool = True,
) -> dict:
    """Build the KB and run pFoodReq's generator; return paths + counts.

    :param recipe_filter: restrict the KB to these ``recipe_url`` dishes (bounded
        build — e.g. the tight stratum); ``None`` uses all of ``recipes_df``.
    :param max_dishes_per_tag: cap the fan-out tail per tag (see build_recipe_kg).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kb_path = out_dir / "recipe_kg.json"

    kb_stats = build_recipe_kg(recipes_df, kb_path, recipe_filter=recipe_filter,
                               max_dishes_per_tag=max_dishes_per_tag)
    if validate:
        report = validate_recipe_kg(kb_path)
        if not report["ok"]:
            raise RuntimeError(f"recipe_kg.json failed validation: {report['errors'][:5]}")

    proc = subprocess.run(
        [sys.executable, "generate_all_qa.py",
         "-recipe", str(kb_path.resolve()), "-o", str(out_dir.resolve()),
         "-out_of_domain_ratio", str(out_of_domain_ratio),
         "-split_ratio", str(split_ratio[0]), str(split_ratio[1])],
        cwd=str(_generator_dir()), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("generate_all_qa.py failed:\n" + proc.stderr[-2000:])

    counts = {split: _count_lines(out_dir / f"{split}_qas.json")
              for split in ("train", "valid", "test")}
    return {"kb_path": str(kb_path), "kb_stats": kb_stats, "qa_dir": str(out_dir),
            "qa_counts": counts, "generator_stdout": proc.stdout.strip()}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
