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

> **Why a clone + `uv`, not `pip install git+…`:** two reasons. (1) The pFoodReq
> code uses the vendored `third_party/pfoodreq/` tree (the BAMnet model +
> `data_builder`), which is *not* part of the installed `foodrag` wheel — so the
> files must be on disk and `settings.repo_root()` must resolve to the clone.
> (2) The project locks to **Python 3.11** (`uv.lock` → `requires-python == 3.11.*`),
> but Colab now ships **Python 3.12**, and its 0.12-era llama-index pins don't
> resolve there — a plain `pip install -e .` fails with `ResolutionImpossible`.
> Colab has no kernel switcher, so we use **`uv`** to provision a standalone 3.11
> interpreter, build the exact locked environment, and run the pipeline through it
> via `uv run` (a small CLI, `scripts/run_pfoodreq_colab.py`) instead of importing
> `foodrag` into Colab's 3.12 kernel.

---

## Prerequisites

1. **Push the repo to GitHub** so Colab can clone it:
   ```bash
   # [local]
   git push                      # pushes the refactor/uv-loader branch
   ```
2. **Copy the project `data/` folder to Google Drive** (e.g.
   `MyDrive/foodrag/data/`). The clone has no `data/` (it's gitignored), so Cell 4
   symlinks the clone's `data/` at this Drive copy — no per-session re-copy.
   pFoodReq only needs `data/hummus/recipes_df.csv` (it uses
   tags/ingredients/nutrients/title).
3. *(Optional)* The **tight stratum** (1–50 valid answers, the protocol's
   recommended scope) is derived from `data/qa/calib/recipe_question_answer_set.pkl`,
   already inside the copied `data/` — no separate upload needed.
4. In Colab: **Runtime → Change runtime type → T4 GPU**.

---

## Cell 1 — confirm the GPU

```python
import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
# Expected: CUDA: True | Tesla T4   (if False, set Runtime -> T4 GPU)
```

## Cell 2 — clone the repo and build the locked 3.11 env (uv)

`uv sync --python 3.11 --frozen` downloads a standalone CPython 3.11, then builds
the exact environment from `uv.lock` into `.venv` — no dependency-resolution
fighting, and CUDA-enabled torch still comes from PyPI so the GPU is used. Takes a
few minutes. Everything after this runs through `uv run` (which auto-selects
`.venv`), not Colab's own Python.

```python
import os
!git clone --branch refactor/uv-loader https://github.com/diana-nurbakova/agentic-FoodRAG.git /content/foodrag
os.chdir('/content/foodrag')
!pip install -q uv
!uv sync --python 3.11 --frozen
# sanity: the env is 3.11 and can import foodrag + torch sees the GPU
!uv run python -c "import sys, foodrag, torch; print('py', sys.version.split()[0], '| foodrag', foodrag.__file__, '| cuda', torch.cuda.is_available())"
```

## Cell 3 — NLTK tokenizer data (into the uv env)

`build_all_data` tokenizes with `nltk.word_tokenize` (the faithful path). Download
through `uv run` so the data lands where the 3.11 env's `nltk` looks for it.

```python
!uv run python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab')]; print('nltk ready')"
```

## Cell 4 — mount Drive and symlink `data/` + results at Drive

The clone has no top-level `data/` (it's gitignored), so we symlink it straight at
your Drive copy — no 1.1 GB re-copy each session. The runner only *reads* from
`data/`; all writes (checkpoints, model, metrics) go to the Drive `--work-dir` /
`--metrics-out` set in the variant cells, so results land on Drive directly with
nothing to copy back afterwards.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
DRIVE_DATA = '/content/drive/MyDrive/foodrag/data'   # <- adjust to your Drive layout
link = '/content/foodrag/data'
if os.path.islink(link):
    os.unlink(link)
elif os.path.isdir(link):
    shutil.rmtree(link)                # a leftover real dir from an earlier run
os.symlink(DRIVE_DATA, link)

# tight-stratum QA set (the protocol's recommended scope); calib is the small one
QA_PKL = '/content/foodrag/data/qa/calib/recipe_question_answer_set.pkl'
# where the run writes (kept on Drive so a disconnect loses nothing)
WORK = '/content/drive/MyDrive/foodrag/work'
os.makedirs(WORK, exist_ok=True)
assert os.path.exists('/content/foodrag/data/hummus/recipes_df.csv'), \
    "recipes_df.csv not visible via the symlink — check DRIVE_DATA"
print("data ->", os.path.realpath(link),
      "| recipes: OK | QA pkl:", os.path.exists(QA_PKL), "| WORK:", WORK)
```

## Recipe subset (how the fan-out is bounded)

The tag-rooted KB duplicates each recipe under every tag, so a full-corpus run is
infeasible (see `specs/pFoodReq_Adaptation.md`). The runner bounds it for you:

- pass **`--qa-pkl {QA_PKL}`** to restrict to the **tight stratum** (1–50 valid
  answers) when a QA pickle is on Drive, or
- omit it to fall back to a **`--sample`** of recipes (default 3000; raise for a
  bigger run).

Both variant cells below already wire this up.

## Cell 5A — variant `pfoodreq` (faithful, GloVe 840B)

Downloads GloVe 840B.300d (~2 GB zip → ~5.6 GB txt), then runs the full pipeline
through the 3.11 env. `build_all_data` is CPU/numpy (the slow stage); training
uses CUDA (`--cuda`).

```python
import os
if not os.path.exists('/content/glove.840B.300d.txt'):
    !wget -q https://nlp.stanford.edu/data/glove.840B.300d.zip -O /content/glove.zip
    !unzip -o -q /content/glove.zip -d /content/
print("glove ready:", os.path.getsize('/content/glove.840B.300d.txt') // 2**20, "MB")

!uv run python scripts/run_pfoodreq_colab.py \
    --recipes data/hummus/recipes_df.csv \
    --work-dir /content/drive/MyDrive/foodrag/work/pfoodreq \
    --variant glove --glove-path /content/glove.840B.300d.txt \
    --qa-pkl {QA_PKL} \
    --max-dishes-per-tag 500 --num-epochs 50 --cuda \
    --metrics-out /content/drive/MyDrive/foodrag/pfoodreq/pfoodreq.metrics.json
```

## Cell 5B — variant `pfoodreq_modernemb` (ablation, modern embedding)

Same pipeline; only the embedding matrix changes. The runner loads a local
sentence-transformer (`--embed-model`, default `BAAI/bge-m3`) on the GPU as the
embedder, so OOV is 0 by construction (§5.8.1).

```python
!uv run python scripts/run_pfoodreq_colab.py \
    --recipes data/hummus/recipes_df.csv \
    --work-dir /content/drive/MyDrive/foodrag/work/pfoodreq_modernemb \
    --variant modernemb --embed-model BAAI/bge-m3 \
    --qa-pkl {QA_PKL} \
    --max-dishes-per-tag 500 --num-epochs 50 --cuda \
    --metrics-out /content/drive/MyDrive/foodrag/pfoodreq/pfoodreq_modernemb.metrics.json
```

## Cell 6 — combined summary

Both variants already wrote their model (`.../work/<variant>/model/bamnet.model`)
and a metrics JSON to Drive (`--metrics-out` + Drive `--work-dir`), so nothing is
lost on a disconnect. This just merges the two metrics files for a side-by-side
read.

```python
import json, os
base = '/content/drive/MyDrive/foodrag/pfoodreq'
out = {}
for name in ("pfoodreq", "pfoodreq_modernemb"):
    path = os.path.join(base, f"{name}.metrics.json")
    if os.path.exists(path):
        out[name] = json.load(open(path))
json.dump(out, open(os.path.join(base, "metrics.json"), "w"), indent=2)
for name, res in out.items():
    m = {k: round(v, 4) for k, v in res.get("metrics", {}).items() if isinstance(v, float)}
    print(name, "| KBQA:", res.get("kbqa_counts"), "| metrics:", m)
```

---

## Notes

- **What the numbers are:** `metrics` reports pFoodReq's *native* recall / precision
  / F1 on the held-out split of its **own HUMMUS-derived KBQA** (Path A, §5.8) —
  not the FoodRAG QA set. Report `pfoodreq` vs `pfoodreq_modernemb` side by side,
  and read both against the GloVe OOV number (`pfoodreq.glove_oov`) so a gap is
  attributable to *method* vs *embedding coverage*.
- **Faithful GloVe hit ratio:** Cell 5A prints `build_pretrained_w2v`'s
  "Pretrained word embeddings hit ratio" (forwarded from the `uv run` subprocess) —
  this is the authoritative 840B coverage of the built vocab (the headline OOV
  figure for the paper).
- **Scope / runtime:** `build_all_data` is the slow stage (CPU/numpy; ~minutes and
  scales with KBQA size — bound it with `--qa-pkl`/`--max-dishes-per-tag`).
  Training then runs on the GPU. Start with the tight stratum; widen only if time
  allows. The free Colab session caps ~12 h and disconnects when idle.
- **Resumable (re-run-safe):** the runner passes `resume=True` — it is re-entrant.
  If the session drops, just re-run the same cell against the same `--work-dir`.
  Completed stages are skipped: KBQA generation and vectorisation are reused when
  their outputs exist, and the modern-embedding file is built word-by-word and
  flushed per batch, so a re-run **only embeds the remaining vocab** rather than
  starting over. (To force a clean rebuild, delete the work dir or add
  `--no-resume`.) The `--work-dir` above is on Drive, so checkpoints survive a
  disconnect.
- **Determinism / faithfulness:** the vendored BAMnet is unchanged except three
  documented torch-2.x / PyYAML / rapidfuzz compat patches (see
  `third_party/pfoodreq/NOTICE.md`). The embedding difference (GloVe vs modern) is
  a deliberate between-method variable, not a confound.
- **Local equivalent (no GPU):** the same runner works locally on CPU — drop
  `--cuda` (CPU is the default) — just slowly. Everything except training/eval
  (data build, KB, vectorize) is CPU-only regardless. Locally you can also call
  `run_pfoodreq.run(recipes_df, work_dir, glove_path=..., no_cuda=True)` directly.
- **Ablation embedder:** any model works as `--embed-model` as long as it returns
  one fixed-length vector per word; `BAAI/bge-m3` (1024-d) matches FoodRAG's own
  embedding family, making the contrast against GloVe most meaningful.
```
