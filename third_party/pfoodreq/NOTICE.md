# Vendored: PFoodReq `data_builder`

Source: <https://github.com/hugochan/PFoodReq> (Chen et al., *Personalized Food
Recommendation as Constrained Question Answering over a Large-scale Food Knowledge
Graph*, WSDM 2021).

License: **Apache License 2.0** (see `LICENSE`).

## What is vendored and why

Both halves are vendored for the **faithful-retrain (Path A)** pFoodReq baseline
(see `specs/pFoodReq_Adaptation.md`):

- `data_builder/` — the QA-generation half (numpy-only): generate HUMMUS-derived
  KBQA from a converted `recipe_kg.json`.
- `BAMnet/` — the model half. Originally torch 0.4.1; it runs on the project's
  **torch 2.x** with only the small compat patches below (0.4.1 already merged
  `Variable` into `Tensor`, so there were no `Variable()`/`.data[0]`/`volatile`
  idioms to rewrite). Driven via `foodrag.baselines.pfoodreq.train_bamnet` /
  `run_pfoodreq` (which import only `BAMnetAgent`, never matchnn/bow/pbow).

Requires `rapidfuzz` (modern) for `build_all_data.py`; GloVe `glove.840B.300d`
for the faithful embedding, or the modern-embedding ablation file (§5.8.1).

## Modifications

Edits to vendored files are kept to unavoidable robustness fixes that do **not**
change the QA-generation method/semantics for non-degenerate inputs; everything
HUMMUS-specific lives in the FoodRAG wrapper (`src/foodrag/baselines/pfoodreq/`).

- `data_builder/src/generate_all_qa.py`, `generate_personal_preferences()`:
  guard an empty ingredient pool (`len(ingredients) == 0` → empty persona).
  Upstream assumed every topic tag has ≥1 ingredient left after sampling;
  HUMMUS-derived KBs contain tags whose dishes share too few distinct
  ingredients, which raised `ValueError: 'a' cannot be empty` from
  `np.random.choice`. The guard only affects those degenerate tags (they get a
  constraint-only question with no ingredient preference); all other questions
  are byte-identical to upstream behaviour.
- `BAMnet/src/core/utils/utils.py`, `get_config()`: `yaml.load(f)` →
  `yaml.load(f, Loader=yaml.FullLoader)` (PyYAML 6 requires an explicit Loader).
- `BAMnet/src/core/build_data/foodkg/build_data.py`, `delex_query_topic_ent()`:
  `for token, score in ret:` → `for token, score, *_ in ret:` (rapidfuzz 3.x
  `process.extract` yields `(choice, score, index)` 3-tuples).
- `BAMnet/src/core/bamnet/bamnet.py`: dropped `verbose=True` from
  `ReduceLROnPlateau` (the kwarg was removed in torch 2.x).

None of these change the model architecture, loss, or QA-generation semantics —
they are environment-compat fixes for torch 2.x / PyYAML 6 / rapidfuzz 3.x.
