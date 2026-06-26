"""Stage 5: generate the QA evaluation set.

Builds persona-driven question/answer pairs for recipe and meal-plan retrieval,
saved under ``<data_path>/qa/<qa_name>/``. Uses the configured judge/LLM and is
resumable via the qa_generator's own stage files.

    foodrag-gen-qa [--data-path data] [--qa-name small_1] [--size 10]
                  [--types recipe,meal_plan]
"""
from __future__ import annotations

import argparse
from typing import Any, Optional


def run(
    data_path: str = "data",
    qa_name: str = "small_1",
    *,
    question_types: Optional[list[str]] = None,
    size: int = 10,
    batch_size: int = 10,
    storage_frequency: int = 10,
    nutrition_config_path: str = "config/nutrition_config.yml",
    persona_config_path: str = "config/persona_config.yml",
    llm: Any = None,
    hummus_data: Any = None,
    food_on: Any = None,
    config: Any = None,
    workers: int = 1,
) -> dict:
    """Generate the QA set(s); return the per-type counts.

    ``llm``, ``hummus_data`` and ``food_on`` are injectable for testing.
    """
    from foodrag.config.settings import build_llm, load_models_config
    from foodrag.persona import qa_generator
    from foodrag.persona.qa_generator import QuestionType
    from foodrag.utils import load_util, rdf_util

    cfg = config or load_models_config()
    nutrition_config = load_util.get_config(nutrition_config_path)
    persona_config = load_util.get_config(persona_config_path)
    # With workers>1 the CPU-bound simple_qa stage runs in subprocesses that each
    # load a light hummus subset + build their own FoodOn, so the parent must NOT
    # load the full ~7GB hummus / build FoodOn (it would risk OOM alongside the
    # workers). The later llm_qa/scored_qa stages only need the QA set + llm.
    parallel_simple = bool(workers and workers > 1)
    if hummus_data is None and not parallel_simple:
        hummus_data = load_util.get_current_hummus(data_path)
    if food_on is None and not parallel_simple:
        food_on = rdf_util.PreProcessedFoodOn(hummus_data).preprocess()
    if llm is None:
        llm = build_llm(config=cfg)

    type_map = {"recipe": QuestionType.RECIPE, "meal_plan": QuestionType.MEAL_PLAN}
    names = question_types or ["recipe", "meal_plan"]
    counts: dict[str, int] = {}
    for name in names:
        qtype = type_map[name]
        qa_set = qa_generator.generate_question_answer_set(
            hummus_data, qtype, persona_config, nutrition_config,
            data_path=data_path, sub_dir_name=qa_name, llm=llm,
            size=size, batch_size=batch_size, storage_frequency=storage_frequency, food_on=food_on,
            workers=workers,
        )
        counts[name] = len(qa_set) if qa_set is not None else 0
    return {"qa_name": qa_name, "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate the QA evaluation set.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--qa-name", default="small_1")
    p.add_argument("--types", help="comma-separated subset of recipe,meal_plan")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--storage-frequency", type=int, default=10,
                   help="save the QA set every N items; each save rewrites the whole pkl, so raise this "
                        "for large runs (e.g. 500 for size=10000) to avoid quadratic I/O")
    p.add_argument("--workers", type=int, default=1,
                   help="processes for the CPU-bound simple_qa stage (each loads a light hummus subset)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run(
        data_path=args.data_path, qa_name=args.qa_name,
        question_types=args.types.split(",") if args.types else None,
        size=args.size, batch_size=args.batch_size, workers=args.workers,
        storage_frequency=args.storage_frequency,
    )
    print(f"QA set '{out['qa_name']}': " + ", ".join(f"{k}={v}" for k, v in out["counts"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
