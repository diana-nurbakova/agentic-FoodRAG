"""Stage: run the deterministic baseline rungs and write Results-Schema files.

Runs the popularity rungs (0a raw, 0b health-aware) of the §5.7 baseline ladder
over a QA set and writes §13 per-question + aggregate outputs under
``eval_results/`` — real numbers to replace the synthetic placeholders the Report
tab currently ships with. These rungs need no LLM and no vector store.

    foodrag run-baselines [--qa data/qa/calib/recipe_question_answer_set.pkl]
                          [--recipes data/hummus/recipes_df.csv]
                          [--out eval_results] [--k 10] [--rungs 0a,0b]

The LLM rungs (2 ``LLMOnly``, 3 ``NaiveRAG``) need a built LLM/retriever and are
driven from the eval notebook via ``foodrag.baselines.run_baseline``.

The external KBQA rungs (``tog``, ``rog``, ``tfs_kbqa`` — §5.8) additionally need a
Neo4j graph; pass ``--neo4j`` to build them from env (with the main LLM as the
generator), or drive them from the notebook via
``foodrag.baselines.kbqa.build_baseline``. Use ``--experiment-id EXP-1`` for the
retrieval (Use 1) pass and the default ``EXP-7`` for full-pipeline (Use 2).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Optional

DEFAULT_QA = "data/qa/calib/recipe_question_answer_set.pkl"
DEFAULT_RECIPES = "data/hummus/recipes_df.csv"


def _load_qa(path: str | Path) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def run(
    qa_path: str = DEFAULT_QA,
    recipes_path: str = DEFAULT_RECIPES,
    *,
    out_dir: str = "eval_results",
    k: int = 10,
    rungs: Optional[list[str]] = None,
    experiment_id: str = "EXP-7",
    recipes_df: Any = None,
    food_on: Any = None,
    persona_config_path: str = "config/persona_config.yml",
    neo4j: bool = False,
    llm: Any = None,
    rog_checkpoint: Optional[str] = None,
    tfs_checkpoint: Optional[str] = None,
) -> dict:
    """Run the requested rungs; return a per-rung summary dict.

    Pass ``food_on`` (+ persona config) to enable FoodOn-hierarchy allergen
    expansion (§4.3); without it, ingredient matching is verbatim Food.com names,
    which matches the shipped QA set's generation config.

    Pass ``neo4j=True`` (and optionally a built ``llm`` + checkpoints) to enable the
    external KBQA rungs ``tog`` / ``rog`` / ``tfs_kbqa`` (§5.8); they build a graph
    client from env and the main LLM as the generator unless one is supplied.
    """
    from foodrag.baselines import (PopularityHealthAware, PopularityRaw,
                                   aggregate, run_baseline, write_results)
    from foodrag.baselines import constraints

    rungs = rungs or ["0a", "0b"]
    qa_set = _load_qa(qa_path)
    if recipes_df is None:
        from foodrag.baselines.runner import load_recipes_df
        recipes_df = load_recipes_df(recipes_path)

    expand = constraints.identity_expand
    if food_on is not None:
        from foodrag.utils import load_util
        persona_config = load_util.get_config(persona_config_path)
        expand = constraints.foodon_expander(food_on, persona_config)

    builders = {
        "0a": lambda: PopularityRaw(recipes_df),
        "0b": lambda: PopularityHealthAware(recipes_df, expand),
    }

    # External KBQA rungs (§5.8) are gated on a Neo4j graph being available.
    if neo4j or any(r in {"tog", "rog", "tfs_kbqa"} for r in rungs):
        from foodrag.baselines.kbqa import KBQA_NAMES, build_baseline, build_kg

        kg = build_kg()
        if llm is None:
            from foodrag.config.settings import build_llm
            llm = build_llm()
        rung_no = "1" if experiment_id == "EXP-1" else "4"
        for name in KBQA_NAMES:
            builders[name] = lambda name=name: build_baseline(
                name, kg, rung=rung_no, llm=llm,
                rog_checkpoint=rog_checkpoint, tfs_checkpoint=tfs_checkpoint)

    summary: dict[str, Any] = {}
    for rung in rungs:
        if rung not in builders:
            raise ValueError(f"Unknown rung {rung!r}; choose from {list(builders)}")
        baseline = builders[rung]()
        records = run_baseline(baseline, qa_set, recipes_df, k=k,
                               experiment_id=experiment_id, expand=expand)
        agg = aggregate(records)
        paths = write_results(records, agg, out_dir=out_dir)
        agg_view = agg[agg.stratum == "aggregate"].set_index("metric")["mean"].round(3).to_dict()
        summary[baseline.config_id] = {"n_questions": len(records), "aggregate": agg_view,
                                       "files": {kk: str(vv) for kk, vv in paths.items()}}
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run baseline rungs (0a/0b; KBQA with --neo4j).")
    p.add_argument("--qa", default=DEFAULT_QA, help="path to a QA set pickle")
    p.add_argument("--recipes", default=DEFAULT_RECIPES, help="path to recipes_df.csv")
    p.add_argument("--out", default="eval_results", help="output directory")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--rungs", default="0a,0b",
                   help="comma-separated rungs: 0a,0b and (with --neo4j) tog,rog,tfs_kbqa")
    p.add_argument("--experiment-id", default="EXP-7",
                   help="EXP-1 for retrieval (Use 1), EXP-7 for full-pipeline (Use 2)")
    p.add_argument("--neo4j", action="store_true", help="enable external KBQA rungs (needs Neo4j)")
    p.add_argument("--rog-checkpoint", default=None, help="fine-tuned RoG planner checkpoint dir")
    p.add_argument("--tfs-checkpoint", default=None, help="fine-tuned TFS parser checkpoint dir")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(qa_path=args.qa, recipes_path=args.recipes, out_dir=args.out,
                  k=args.k, rungs=args.rungs.split(","), experiment_id=args.experiment_id,
                  neo4j=args.neo4j, rog_checkpoint=args.rog_checkpoint,
                  tfs_checkpoint=args.tfs_checkpoint)
    for config_id, info in summary.items():
        print(f"[{config_id}] {info['n_questions']} questions -> {info['files']['aggregate']}")
        for metric, value in info["aggregate"].items():
            print(f"    {metric:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
