"""Stage: run the external KBQA baselines through the full pipeline (Use 2, §5.7).

Wires each KBQA system (ToG/RoG/TFS-KBQA) into FoodRAG's meal-plan assembly layer
and the dual-judge pass (EXP-7 rung 4), writing §13 records to ``eval_results/``
alongside the deterministic rungs. Unlike ``run-baselines`` this needs a Neo4j
graph, a generator LLM and judge credentials, so it is meant to be driven from the
eval notebook (or the CLI below when the environment is configured).

    from foodrag.pipeline import run_full_pipeline
    run_full_pipeline.run(systems=["tog", "rog", "tfs_kbqa"])

    # or, for the EXP-1-style retrieval pass reuse foodrag.pipeline.run_baselines

For the **full agentic pipeline** (Heuristic 1) on top of KBQA retrieval, build the
turnkey assemblers (the five-agent chain is assembled in code; just pass
``hummus_data``) and hand them in::

    from foodrag.baselines.kbqa import build_agentic_pipeline, build_kg
    kg = build_kg()
    assemblers = [build_agentic_pipeline(name, kg, recipes_df, llm=llm, hummus_data=hummus_data)
                  for name in ("tog", "rog", "tfs_kbqa")]
    run_full_pipeline.run(assemblers=assemblers, recipes_df=recipes_df, judges=judges)
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
    systems: Optional[list[str]] = None,
    experiment_id: str = "EXP-7",
    retrieve_only: bool = True,
    require_explanation: bool = False,
    recipes_df: Any = None,
    llm: Any = None,
    judges: Any = None,
    assemblers: Any = None,
    rog_checkpoint: Optional[str] = None,
    tfs_checkpoint: Optional[str] = None,
    run_idx: int = 1,
) -> dict:
    """Run KBQA systems through assembly + judging; return a summary.

    By default builds a recommend-layer assembler per ``systems`` name; pass
    ``assemblers`` (e.g. agentic ``WorkflowMealPlanAssembler`` built via
    ``build_agentic_pipeline``) to score the full agentic pipeline instead.
    """
    from foodrag.baselines.full_pipeline import run_full_pipeline
    from foodrag.baselines.judge import build_judges
    from foodrag.baselines.kbqa import KBQA_NAMES, build_full_pipeline, build_kg
    from foodrag.baselines.runner import aggregate, load_recipes_df, write_results

    qa_set = _load_qa(qa_path)
    if recipes_df is None:
        recipes_df = load_recipes_df(recipes_path)
    if judges is None:
        judges = build_judges()

    if assemblers is None:
        systems = systems or list(KBQA_NAMES)
        if llm is None:
            from foodrag.config.settings import build_llm
            llm = build_llm()
        kg = build_kg()
        assemblers = [build_full_pipeline(
            name, kg, recipes_df, llm=llm, config_id=name, retrieve_only=retrieve_only,
            require_explanation=require_explanation, rog_checkpoint=rog_checkpoint,
            tfs_checkpoint=tfs_checkpoint) for name in systems]

    summary: dict[str, Any] = {}
    for assembler in assemblers:
        name = assembler.config_id
        records = run_full_pipeline(assembler, qa_set, recipes_df, judges=judges, k=k,
                                    experiment_id=experiment_id, run=run_idx)
        agg = aggregate(records)
        paths = write_results(records, agg, out_dir=out_dir)
        agg_view = agg[agg.stratum == "aggregate"].set_index("metric")["mean"].round(3).to_dict()
        summary[name] = {"n_questions": len(records), "aggregate": agg_view,
                         "files": {kk: str(vv) for kk, vv in paths.items()}}
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run KBQA baselines through the full pipeline (Use 2).")
    p.add_argument("--qa", default=DEFAULT_QA)
    p.add_argument("--recipes", default=DEFAULT_RECIPES)
    p.add_argument("--out", default="eval_results")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--systems", default="tog,rog,tfs_kbqa", help="comma-separated KBQA systems")
    p.add_argument("--experiment-id", default="EXP-7")
    p.add_argument("--plain-rag", action="store_true",
                   help="let the query engine's LLM pick per slot (retrieve_only=False)")
    p.add_argument("--require-explanation", action="store_true")
    p.add_argument("--rog-checkpoint", default=None)
    p.add_argument("--tfs-checkpoint", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(qa_path=args.qa, recipes_path=args.recipes, out_dir=args.out, k=args.k,
                  systems=args.systems.split(","), experiment_id=args.experiment_id,
                  retrieve_only=not args.plain_rag, require_explanation=args.require_explanation,
                  rog_checkpoint=args.rog_checkpoint, tfs_checkpoint=args.tfs_checkpoint)
    for name, info in summary.items():
        print(f"[{name}] {info['n_questions']} questions -> {info['files']['aggregate']}")
        for metric, value in info["aggregate"].items():
            print(f"    {metric:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
