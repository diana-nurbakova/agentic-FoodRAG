"""pFoodReq / BAMnet runner (Evaluation Protocol §5.8, rung 4) — CLI entrypoint.

Folds Colab cells 5/6/7 of ``notebooks/colab_pfoodreq.md`` into one argparse
command so the pipeline can run under a ``uv``-provisioned Python 3.11
environment (the version the project's ``uv.lock`` is pinned to). Colab now ships
Python 3.12 and has no kernel switcher, so the interactive-import flow no longer
installs; ``uv run --python 3.11`` sidesteps that entirely.

Two faithful variants (§5.8):

    # GloVe 840B (faithful reproduction)
    uv run --python 3.11 python scripts/run_pfoodreq_colab.py \
        --recipes data/hummus/recipes_df.csv --work-dir work/pfoodreq \
        --variant glove --glove-path /content/glove.840B.300d.txt \
        --qa-pkl /content/drive/MyDrive/foodrag/recipe_question_answer_set.pkl \
        --cuda --num-epochs 50 --metrics-out work/pfoodreq/metrics.json

    # modern embedding (§5.8.1 ablation)
    uv run --python 3.11 python scripts/run_pfoodreq_colab.py \
        --recipes data/hummus/recipes_df.csv --work-dir work/pfoodreq_modernemb \
        --variant modernemb --embed-model BAAI/bge-m3 \
        --qa-pkl /content/drive/MyDrive/foodrag/recipe_question_answer_set.pkl \
        --cuda --num-epochs 50 --metrics-out work/pfoodreq_modernemb/metrics.json

``--qa-pkl`` bounds the run to the tight stratum (1-50 valid answers, the
protocol's recommended scope); without it the recipes are sampled
(``--sample``). ``run_pfoodreq.run`` is re-entrant (``resume=True``), so a
re-run against the same ``--work-dir`` only does the remaining work — keep the
work dir on Drive so checkpoints survive a Colab disconnect.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _build_recipe_filter(qa_pkl: Path) -> set[str]:
    """Collect the tight-stratum (1-50 answers) recipe ids from a QA pickle."""
    from foodrag.baselines.base import answer_set_stratum

    with open(qa_pkl, "rb") as fh:
        qa = pickle.load(fh)
    recipe_filter: set[str] = set()
    for item in qa:
        ans = item.get("answer") or []
        if answer_set_stratum(len(ans)) == "tight":
            recipe_filter.update(ans)
    return recipe_filter


def _make_embed_fn(model_name: str, device: str):
    """A per-word static embedder for the §5.8.1 modern-embedding ablation.

    Returns one fixed-length vector per vocab word (OOV is 0 by construction),
    matching FoodRAG's own embedding family when ``model_name`` is bge-m3.
    """
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device=device)

    def embed_fn(words):
        return st.encode(list(words), batch_size=128, show_progress_bar=False)

    return embed_fn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipes", required=True, type=Path,
                   help="path to recipes_df.csv")
    p.add_argument("--work-dir", required=True, type=Path,
                   help="pipeline work dir (kbqa/vec/model live here; keep on Drive)")
    p.add_argument("--variant", required=True, choices=("glove", "modernemb"),
                   help="glove = faithful 840B; modernemb = §5.8.1 ablation")
    p.add_argument("--glove-path", type=Path, default=None,
                   help="glove.840B.300d.txt (required for --variant glove)")
    p.add_argument("--embed-model", default="BAAI/bge-m3",
                   help="sentence-transformer for --variant modernemb")
    p.add_argument("--qa-pkl", type=Path, default=None,
                   help="QA pickle to bound the run to the tight stratum")
    p.add_argument("--sample", type=int, default=3000,
                   help="recipe sample size when no --qa-pkl is given")
    p.add_argument("--seed", type=int, default=0,
                   help="random_state for the no-QA-pkl sample")
    p.add_argument("--max-dishes-per-tag", type=int, default=500,
                   help="cap the tag-rooted KB fan-out (0 -> no cap)")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--eval-split", default="valid", choices=("valid", "test"))
    p.add_argument("--cuda", dest="no_cuda", action="store_false", default=True,
                   help="train/eval on the GPU (default: CPU)")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="force a clean rebuild instead of reusing checkpoints")
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="also write kbqa_counts + metrics to this JSON file")
    args = p.parse_args(argv)

    if args.variant == "glove" and args.glove_path is None:
        p.error("--variant glove requires --glove-path")

    from foodrag.baselines.pfoodreq import run_pfoodreq
    from foodrag.baselines.runner import load_recipes_df

    recipes_df = load_recipes_df(args.recipes)
    print(f"recipes: {len(recipes_df)}", flush=True)

    recipe_filter = None
    if args.qa_pkl is not None and args.qa_pkl.exists():
        recipe_filter = _build_recipe_filter(args.qa_pkl)
        print(f"tight-stratum recipe_filter: {len(recipe_filter)} recipes", flush=True)
    else:
        if args.qa_pkl is not None:
            print(f"QA pkl not found at {args.qa_pkl} -> sampling", flush=True)
        recipes_df = recipes_df.sample(args.sample, random_state=args.seed)
        print(f"no QA pkl -> sampled {len(recipes_df)} recipes", flush=True)

    embed_fn = None
    glove_path = None
    if args.variant == "glove":
        glove_path = str(args.glove_path)
    else:
        device = "cpu" if args.no_cuda else "cuda"
        embed_fn = _make_embed_fn(args.embed_model, device)

    max_dishes = args.max_dishes_per_tag or None
    res = run_pfoodreq.run(
        recipes_df, args.work_dir,
        recipe_filter=recipe_filter,
        max_dishes_per_tag=max_dishes,
        num_epochs=args.num_epochs,
        glove_path=glove_path,
        embed_fn=embed_fn,
        eval_split=args.eval_split,
        no_cuda=args.no_cuda,
        resume=args.resume,
    )

    print(f"VARIANT: {res['variant']} | KBQA: {res['kbqa_counts']}", flush=True)
    metrics = {k: round(v, 4) for k, v in res["metrics"].items() if isinstance(v, float)}
    print(f"metrics: {metrics}", flush=True)

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"variant": res["variant"], "kbqa_counts": res["kbqa_counts"],
                   "metrics": res["metrics"]}
        with open(args.metrics_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote metrics -> {args.metrics_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
