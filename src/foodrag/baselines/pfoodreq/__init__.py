"""pFoodReq / BAMnet adaptation (baseline ladder rung 4, §5.8).

Currently implements the **data-adaptation layer** — the torch-independent half
that maps HUMMUS into the FoodKG-style input pFoodReq expects, and validates it
without the 2018-era dependency stack:

* :func:`build_recipe_kg` — HUMMUS → ``recipe_kg.json`` (tag-rooted ndjson).
* :func:`validate_recipe_kg` — structural check mirroring pFoodReq's field access.

Still TODO (follow-up): QA conversion (lossy — see the design note), BAMnet
modernization (torch 0.4.1 → current), bounded training/inference.
"""
from foodrag.baselines.pfoodreq.build_recipe_kg import NUTRIENT_MAP, build_recipe_kg
from foodrag.baselines.pfoodreq.validate import load_recipe_kg, validate_recipe_kg

__all__ = ["build_recipe_kg", "validate_recipe_kg", "load_recipe_kg", "NUTRIENT_MAP"]
