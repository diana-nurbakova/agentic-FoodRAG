# pFoodReq / BAMnet on Google Colab (free GPU)

Run the **pFoodReq baseline** (Evaluation Protocol §5.8, rung 4) end-to-end on a
free Colab GPU: HUMMUS → `recipe_kg.json` → HUMMUS-derived KBQA → vectorize →
**train BAMnet** → evaluate. BAMnet trains slowly on CPU, so the GPU is the point
of doing this on Colab. Two variants:

- **`pfoodreq`** — faithful reproduction on the original **GloVe 840B.300d**.
- **`pfoodreq_modernemb`** — the §5.8.1 ablation: same pipeline, embedding matrix
  swapped for a modern embedder (one variable changed).

This file reproduces a Colab notebook: copy each fenced code block into its own
Colab cell, in order. Cells marked **[local]** run on your own machine.

> **Why a clone, not `pip install git+…`:** the pFoodReq code uses the vendored
> `third_party/pfoodreq/` tree (the BAMnet model + `data_builder`), which is *not*
> part of the installed `foodrag` wheel. We `git clone` and `pip install -e .` so
> those files are on disk and `settings.repo_root()` resolves to the clone.

---

## Prerequisites

1. **Push the repo to GitHub** so Colab can clone it:
   ```bash
   # [local]
   git push                      # pushes the refactor/uv-loader branch
   ```
2. **Put `recipes_df.csv` on Google Drive** (e.g. `MyDrive/foodrag/recipes_df.csv`,
   ~1.1 GB) — the annotated one is fine but not required (pFoodReq only uses
   tags/ingredients/nutrients/title).
3. *(Optional)* **Put a QA pickle on Drive** (e.g. the calib set,
   `MyDrive/foodrag/recipe_question_answer_set.pkl`) to bound the run to the
   **tight stratum** (1–50 valid answers) — the protocol's recommended scope.
4. In Colab: **Runtime → Change runtime type → T4 GPU**.

---

## Cell 1 — confirm the GPU

```python
import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
# Expected: CUDA: True | Tesla T4   (if False, set Runtime -> T4 GPU)
```

## Cell 2 — clone the repo and install it (editable)

Editable install keeps `third_party/pfoodreq/` on disk and `repo_root()` pointing
at the clone. Pulls foodrag's deps (incl. `rapidfuzz`); takes a few minutes.

```python
import os
!git clone --branch refactor/uv-loader https://github.com/diana-nurbakova/agentic-FoodRAG.git /content/foodrag
os.chdir('/content/foodrag')
!pip install -q -e .
import foodrag  # noqa
print("foodrag installed from", foodrag.__file__)
```

## Cell 3 — NLTK tokenizer data

`build_all_data` tokenizes with `nltk.word_tokenize` (the faithful path).

```python
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
print("nltk ready")
```

## Cell 4 — mount Drive and stage inputs

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
os.makedirs('/content/foodrag/data/hummus', exist_ok=True)
shutil.copy('/content/drive/MyDrive/foodrag/recipes_df.csv',
            '/content/foodrag/data/hummus/recipes_df.csv')
# optional: a QA pickle to derive the tight-stratum recipe filter
QA_PKL = '/content/drive/MyDrive/foodrag/recipe_question_answer_set.pkl'
print("recipes staged:",
      os.path.getsize('/content/foodrag/data/hummus/recipes_df.csv') // 2**20, "MB",
      "| QA pkl present:", os.path.exists(QA_PKL))
```

## Cell 5 — load recipes + choose the recipe subset (bound the fan-out)

The tag-rooted KB duplicates each recipe under every tag, so a full-corpus run is
infeasible (see `specs/pFoodReq_Adaptation.md`). Bound to the **tight stratum** if
a QA pkl is available, else take a sample.

```python
import os, pickle
from foodrag.baselines.runner import load_recipes_df
from foodrag.baselines.base import answer_set_stratum

recipes_df = load_recipes_df('/content/foodrag/data/hummus/recipes_df.csv')
print("recipes:", len(recipes_df))

recipe_filter = None
if os.path.exists(QA_PKL):
    qa = pickle.load(open(QA_PKL, 'rb'))
    recipe_filter = set()
    for it in qa:
        ans = it.get('answer') or []
        if answer_set_stratum(len(ans)) == 'tight':
            recipe_filter.update(ans)
    print("tight-stratum recipe_filter:", len(recipe_filter), "recipes")
else:
    # no QA pkl -> sample (smoke-sized); raise for a bigger run
    recipes_df = recipes_df.sample(3000, random_state=0)
    print("no QA pkl -> sampled", len(recipes_df), "recipes")
```

## Cell 6A — variant `pfoodreq` (faithful, GloVe 840B)

Downloads GloVe 840B.300d (~2 GB zip → ~5.6 GB txt) and runs the full pipeline on
the GPU. `build_all_data` is CPU/numpy (the slow stage); training uses CUDA.

```python
import os
if not os.path.exists('/content/glove.840B.300d.txt'):
    !wget -q https://nlp.stanford.edu/data/glove.840B.300d.zip -O /content/glove.zip
    !unzip -o -q /content/glove.zip -d /content/
print("glove ready:", os.path.getsize('/content/glove.840B.300d.txt') // 2**20, "MB")

from foodrag.baselines.pfoodreq import run_pfoodreq
res_glove = run_pfoodreq.run(
    recipes_df, '/content/work/pfoodreq',
    recipe_filter=recipe_filter,         # None -> use all of recipes_df
    max_dishes_per_tag=500,
    num_epochs=50,
    glove_path='/content/glove.840B.300d.txt',
    no_cuda=False,                       # use the GPU
)
print("VARIANT:", res_glove["variant"], "| KBQA:", res_glove["kbqa_counts"])
print("metrics:", {k: round(v, 4) for k, v in res_glove["metrics"].items() if isinstance(v, float)})
```

## Cell 6B — variant `pfoodreq_modernemb` (ablation, modern embedding)

Same pipeline; only the embedding matrix changes. Uses a local sentence-transformer
on the GPU as the embedder, so OOV is 0 by construction (§5.8.1).

```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('BAAI/bge-m3', device='cuda')

def embed_fn(words):
    # returns one vector per vocab word (static per-word table for BAMnet)
    return st.encode(list(words), batch_size=128, show_progress_bar=False)

from foodrag.baselines.pfoodreq import run_pfoodreq
res_modern = run_pfoodreq.run(
    recipes_df, '/content/work/pfoodreq_modernemb',
    recipe_filter=recipe_filter,
    max_dishes_per_tag=500,
    num_epochs=50,
    embed_fn=embed_fn,                   # modern embedding instead of glove_path
    no_cuda=False,
)
print("VARIANT:", res_modern["variant"], "| KBQA:", res_modern["kbqa_counts"])
print("metrics:", {k: round(v, 4) for k, v in res_modern["metrics"].items() if isinstance(v, float)})
```

## Cell 7 — save models + metrics back to Drive

```python
import json, shutil, os
os.makedirs('/content/drive/MyDrive/foodrag/pfoodreq', exist_ok=True)
out = {}
for name, work in [("pfoodreq", "/content/work/pfoodreq"),
                   ("pfoodreq_modernemb", "/content/work/pfoodreq_modernemb")]:
    model = os.path.join(work, "model", "bamnet.model")
    if os.path.exists(model):
        shutil.copy(model, f'/content/drive/MyDrive/foodrag/pfoodreq/{name}.model')
for name, res in [("pfoodreq", locals().get("res_glove")),
                  ("pfoodreq_modernemb", locals().get("res_modern"))]:
    if res:
        out[name] = {"kbqa_counts": res["kbqa_counts"], "metrics": res["metrics"]}
json.dump(out, open('/content/drive/MyDrive/foodrag/pfoodreq/metrics.json', 'w'), indent=2)
print("saved models + metrics.json to Drive:", out)
```

---

## Notes

- **What the numbers are:** `metrics` reports pFoodReq's *native* recall / precision
  / F1 on the held-out split of its **own HUMMUS-derived KBQA** (Path A, §5.8) —
  not the FoodRAG QA set. Report `pfoodreq` vs `pfoodreq_modernemb` side by side,
  and read both against the GloVe OOV number (`pfoodreq.glove_oov`) so a gap is
  attributable to *method* vs *embedding coverage*.
- **Faithful GloVe hit ratio:** Cell 6A prints `build_pretrained_w2v`'s
  "Pretrained word embeddings hit ratio" — this is the authoritative 840B coverage
  of the built vocab (the headline OOV figure for the paper).
- **Scope / runtime:** `build_all_data` is the slow stage (CPU/numpy; ~minutes and
  scales with KBQA size — bound it with `recipe_filter`/`max_dishes_per_tag`).
  Training then runs on the GPU. Start with the tight stratum; widen only if time
  allows. The free Colab session caps ~12 h and disconnects when idle.
- **Determinism / faithfulness:** the vendored BAMnet is unchanged except three
  documented torch-2.x / PyYAML / rapidfuzz compat patches (see
  `third_party/pfoodreq/NOTICE.md`). The embedding difference (GloVe vs modern) is
  a deliberate between-method variable, not a confound.
- **Local equivalent (no GPU):** the same call runs locally on CPU —
  `run_pfoodreq.run(recipes_df, work_dir, glove_path=..., no_cuda=True)` — just
  slowly. Everything except training/eval (data build, KB, vectorize) is CPU-only
  regardless.
- **Ablation embedder:** any model works as `embed_fn` as long as it returns one
  fixed-length vector per word; `BAAI/bge-m3` (1024-d) matches FoodRAG's own
  embedding family, making the contrast against GloVe most meaningful.
```
