"""pFoodReq / BAMnet adaptation (baseline ladder rung 4, §5.8).

Currently implements the **data-adaptation layer** — the torch-independent half
that maps HUMMUS into the FoodKG-style input pFoodReq expects, and validates it
without the 2018-era dependency stack:

* :func:`build_recipe_kg` — HUMMUS → ``recipe_kg.json`` (tag-rooted ndjson).
* :func:`validate_recipe_kg` — structural check mirroring pFoodReq's field access.

Still TODO (follow-up): QA conversion (lossy — see the design note), BAMnet
modernization (torch 0.4.1 → current), bounded training/inference.
"""
from foodrag.baselines.pfoodreq.build_modern_w2v import (project_embedder,
                                                         vocab_from_vocab2id,
                                                         write_glove_format)
from foodrag.baselines.pfoodreq.build_recipe_kg import NUTRIENT_MAP, build_recipe_kg
from foodrag.baselines.pfoodreq.generate_qa import generate_kbqa
from foodrag.baselines.pfoodreq.glove_oov import (analyze as glove_oov_analyze,
                                                  analyze_with_gensim as glove_oov_proxy,
                                                  extract_food_vocab)
from foodrag.baselines.pfoodreq.validate import load_recipe_kg, validate_recipe_kg

# Model side (torch) imported lazily — keep package import torch-free for the
# data/OOV tools. Access via foodrag.baselines.pfoodreq.run_pfoodreq /
# .train_bamnet, e.g. `from foodrag.baselines.pfoodreq import run_pfoodreq`.

__all__ = ["build_recipe_kg", "generate_kbqa", "validate_recipe_kg",
           "load_recipe_kg", "NUTRIENT_MAP", "glove_oov_analyze", "glove_oov_proxy",
           "extract_food_vocab", "write_glove_format", "project_embedder",
           "vocab_from_vocab2id"]
