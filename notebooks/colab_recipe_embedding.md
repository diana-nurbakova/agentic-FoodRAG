# Recipe embedding on Google Colab (free GPU)

Embed the HUMMUS recipes with **bge-m3 on a free Colab GPU** instead of the
DeepInfra API — minutes-to-~1h and free, vs ~5–8h / ~$2 on the API. Because it's
the *same model* (`BAAI/bge-m3`, 1024-d), the vectors are interchangeable with
the food_product embeddings already produced via DeepInfra, and they load into
pgvector locally with `foodrag init-vector-store` exactly as usual.

This file reproduces a Colab notebook: copy each fenced code block into its own
Colab cell (in order). Cells marked **[local]** run on your own machine, not Colab.

---

## Prerequisites

1. **Push the repo to GitHub** so Colab can `pip install` it:
   ```bash
   # [local]
   git push                      # pushes the refactor/uv-loader branch
   ```
2. **Put the annotated `recipes_df.csv` on Google Drive.** After the meal-type
   annotation finishes (it rewrites `data/hummus/recipes_df.csv`), upload that
   file to e.g. `MyDrive/foodrag/recipes_df.csv` (~1.1 GB). Embedding before the
   annotation is fine too — the docs just won't include `meal_types`.
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

---

## Cell 1 — confirm the GPU

```python
import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
# Expected: CUDA: True | Tesla T4   (if False, set Runtime -> T4 GPU)
```

## Cell 2 — install foodrag

Installs the package + its deps (llama-index, sentence-transformers, …). Takes a
few minutes; Colab's preinstalled torch is reused.

```python
!pip install -q "git+https://github.com/diana-nurbakova/agentic-FoodRAG.git@refactor/uv-loader"
# sanity check
import foodrag, llama_index  # noqa
print("foodrag installed")
```

## Cell 3 — mount Drive and stage the recipes

The embed step reads `data/hummus/recipes_df.csv` relative to the working dir.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
os.makedirs('/content/work/data/hummus', exist_ok=True)
os.chdir('/content/work')
shutil.copy('/content/drive/MyDrive/foodrag/recipes_df.csv', 'data/hummus/recipes_df.csv')
print("staged:", os.path.getsize('data/hummus/recipes_df.csv') // 2**20, "MB")
```

## Cell 4 — load `recipes_df` (only this, not the full hummus cache)

We load just `recipes_df` (~1.1 GB) instead of the whole hummus dataset, and
re-apply the same list/dict deserialization `get_current_hummus` does.

```python
from foodrag.utils.load_util import load_file, serialize

recipes_df = load_file("data/hummus/recipes_df.csv")
for col in ["tags", "directions", "ingredient_names", "ingredients", "meal_types"]:
    if col in recipes_df.columns:
        recipes_df[col] = recipes_df[col].apply(serialize)
print("recipes:", len(recipes_df), "| meal_types present:", "meal_types" in recipes_df.columns)
```

## Cell 5 — embed on the GPU

Builds a **local** bge-m3 on CUDA and runs the standard embed pipeline, writing
the usual batch files to `data/embeddings/BAAI_bge-m3/recipe_documents/`.

```python
import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from foodrag.pipeline import embed

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cuda", embed_batch_size=64)

t = time.time()
out = embed.run(
    doc_types=["recipe"],
    hummus_data={"recipes_df": recipes_df},   # only recipes_df, no full hummus load
    embed_model=embed_model,
    batch_size=2000,
)
print(f"done in {(time.time()-t)/60:.1f} min ->", out["counts"], "@", out["embedding_path"])
```

> Tip: to also embed users/reviews/ingredients/food_on/usda here, add them to
> `doc_types` — but those need the corresponding dataframes in `hummus_data`
> (e.g. `users_df`, `usda_df`), so stage those CSVs too. For just the vector
> store, `recipe` is what you need.

## Cell 6 — save the batches back to Drive

```python
import shutil
shutil.make_archive('/content/drive/MyDrive/foodrag/recipe_documents',
                    'zip', 'data/embeddings/BAAI_bge-m3/recipe_documents')
print("saved recipe_documents.zip to Drive")
```

---

## Cell 7 — **[local]** load the embeddings into pgvector

Download `recipe_documents.zip` from Drive, then on your machine:

```bash
# [local] — from the repo root, with Docker up (foodrag-postgres)
mkdir -p data/embeddings/BAAI_bge-m3/recipe_documents
unzip -o ~/Downloads/recipe_documents.zip -d data/embeddings/BAAI_bge-m3/recipe_documents

uv run foodrag init-vector-store --table hummus_recipes
```

`init-vector-store` reads those batches and builds the pgvector HNSW index — same
as if they'd been produced by the DeepInfra API path.

---

## Notes

- **Same vectors:** bge-m3 is deterministic; the Colab (local fp16 GPU) and
  DeepInfra embeddings live in the same space, so recipe + food_product vectors
  are compatible. No need to re-embed food_product.
- **No GPU? No problem:** the DeepInfra path (`uv run foodrag embed --doc-types recipe`)
  needs no GPU at all — this notebook is only the free/fast alternative.
- **Equivalent local CLI:** on a machine with a GPU you can skip Colab entirely:
  `uv run foodrag embed --doc-types recipe --embed-provider huggingface` (keeps
  the same `BAAI/bge-m3` model id and output dir).
- **Colab limits:** free sessions cap at ~12h and disconnect when idle — fine for
  a ~1h embed, but keep the tab active. If it disconnects mid-run, the saved
  batches are reusable; `embedding_helper.load_existing_batches` + a re-run
  resumes from completed batches.
- **Session RAM:** loading only `recipes_df` (not the full hummus cache) keeps
  Colab well within its ~12 GB RAM.
