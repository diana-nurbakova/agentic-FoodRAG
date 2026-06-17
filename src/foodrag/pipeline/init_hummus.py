"""Stage 1: load and clean the HUMMUS dataset.

Resolves dataset locations from the manifest, optionally builds the LLM for
meal-type annotation, and calls ``load_util.get_hummus``. The processed data is
cached under ``<data_path>/hummus/``.

    foodrag-init-hummus [--data-path data] [--k-user 2] [--k-recipe 2]
                        [--no-usda] [--annotate-meal-type] [--random-meal-types]
"""
from __future__ import annotations

import argparse
from typing import Any, Optional


def run(
    data_path: str = "data",
    *,
    k_user: int = 2,
    k_recipe: int = 2,
    n_clusters: int = -1,
    food_class_strategy: str = "None",
    include_usda: bool = True,
    annotate_meal_type: bool = False,
    create_random_meal_types: bool = False,
    discard_non_tagged_recipes: bool = False,
    llm: Any = None,
    validate: bool = True,
    loader: Any = None,
) -> dict:
    """Initialise (or load cached) HUMMUS data and return the data dict.

    ``loader`` (a DataLoader) and ``llm`` are injectable for testing.
    Meal-type annotation uses the LLM and is therefore expensive; it is off by
    default. ``get_hummus`` itself caches and only re-initialises when the
    parameters change.
    """
    from foodrag.data import DataLoader
    from foodrag.utils import load_util

    loader = loader or DataLoader.from_manifest()
    if validate:
        loader.require()
    if annotate_meal_type and llm is None:
        from foodrag.config import settings

        llm = settings.build_llm()

    kw = loader.hummus_kwargs()
    return load_util.get_hummus(
        kw["data_location"],
        kw["graph_location"],
        kw["additional_location"],
        data_path=data_path,
        discard_non_tagged_recipes=discard_non_tagged_recipes,
        n_clusters=n_clusters,
        k_user=k_user,
        k_recipe=k_recipe,
        food_class_strategy=food_class_strategy,
        add_additional_meal_type_labels=True,
        clean_food_category=True,
        similarity_thresholds=False,
        include_usda=include_usda,
        do_annotate_meal_type=annotate_meal_type,
        create_random_meal_types=create_random_meal_types,
        llm=llm,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Initialise the HUMMUS dataset.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--k-user", type=int, default=2)
    p.add_argument("--k-recipe", type=int, default=2)
    p.add_argument("--no-usda", action="store_true", help="skip USDA nutrition data (less RAM)")
    p.add_argument("--annotate-meal-type", action="store_true", help="LLM meal-type annotation (expensive)")
    p.add_argument("--random-meal-types", action="store_true", help="random meal types (for test sets)")
    p.add_argument("--no-validate", action="store_true", help="skip manifest validation")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data = run(
        data_path=args.data_path,
        k_user=args.k_user,
        k_recipe=args.k_recipe,
        include_usda=not args.no_usda,
        annotate_meal_type=args.annotate_meal_type,
        create_random_meal_types=args.random_meal_types,
        validate=not args.no_validate,
    )
    recipes = data.get("recipes_df")
    n = len(recipes) if recipes is not None else "?"
    print(f"HUMMUS initialised: {n} recipes -> {args.data_path}/hummus/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
