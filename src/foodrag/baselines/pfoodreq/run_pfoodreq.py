"""End-to-end pFoodReq orchestrator on HUMMUS (baseline ladder rung 4, §5.8).

Chains the whole faithful-retrain (Path A) pipeline:

    recipes_df → recipe_kg.json → HUMMUS-derived KBQA  (generate_qa)
              → build_all_data.py  (vectorise → *_vec.json, *2id.json)
              → [optional] build_pretrained_w2v  (GloVe 840B *or* modern-emb ablation)
              → train BAMnet  (train_bamnet.train)
              → evaluate  (recall / precision / F1 on the held-out split)

Every stage is already proven individually; this ties them together with the
fan-out bounding (``recipe_filter`` / ``max_dishes_per_tag``) so a run stays on
the tight stratum. Heavy stages (`build_all_data`, training) shell out / run on
CPU, so this is a driver for real runs rather than a unit test.

Two embedding variants (§5.8.1): pass ``glove_path`` for the faithful GloVe run,
or ``embed_fn`` for the modern-embedding ablation (`pfoodreq_modernemb`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from foodrag.baselines.pfoodreq.build_modern_w2v import vocab_from_vocab2id, write_glove_format
from foodrag.baselines.pfoodreq.generate_qa import generate_kbqa
from foodrag.baselines.pfoodreq.train_bamnet import evaluate, train


def _bamnet_src() -> Path:
    from foodrag.config.settings import repo_root
    return repo_root() / "third_party" / "pfoodreq" / "BAMnet" / "src"


def _vec_done(vec_dir: Path) -> bool:
    return all((vec_dir / f).exists() for f in ("vocab2id.json", "train_vec.json", "valid_vec.json"))


def build_all_data(kbqa_dir: str | Path, kb_path: str | Path, vec_dir: str | Path,
                   *, no_query_expansion: bool = False, no_kg_augmentation: bool = False,
                   resume: bool = True) -> Path:
    """Run the vendored ``build_all_data.py`` (vectorise KBQA → vec_dir).

    Re-entrant: with ``resume=True`` the (slow) vectorisation is skipped when its
    outputs already exist, so a re-run after an interruption doesn't redo it.
    """
    vec_dir = Path(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)
    if resume and _vec_done(vec_dir):
        return vec_dir
    args = [sys.executable, "build_all_data.py", "-data_dir", str(Path(kbqa_dir).resolve()),
            "-kb_path", str(Path(kb_path).resolve()), "-out_dir", str(vec_dir.resolve())]
    if no_query_expansion:
        args.append("--no_query_expansion")
    if no_kg_augmentation:
        args.append("--no_kg_augmentation")
    proc = subprocess.run(args, cwd=str(_bamnet_src()), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("build_all_data.py failed:\n" + proc.stderr[-2000:])
    return vec_dir


def build_embeddings(vec_dir: str | Path, out_npy: str | Path, *,
                     glove_path: Optional[str | Path] = None,
                     embed_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None) -> dict:
    """Build BAMnet's pretrained-embedding matrix for the built ``vocab2id``.

    Faithful run: pass ``glove_path`` (e.g. ``glove.840B.300d.txt``). Ablation
    (§5.8.1): pass ``embed_fn`` to synthesise a GloVe-format file from a modern
    embedder over the vocab, then build the matrix from that. Returns the
    embedding dim so the caller can set ``vocab_embed_size`` accordingly.
    """
    vec_dir = Path(vec_dir)
    out_npy = Path(out_npy)
    if (glove_path is None) == (embed_fn is None):
        raise ValueError("pass exactly one of glove_path (faithful) or embed_fn (ablation)")

    dim = None
    if embed_fn is not None:
        glove_path = out_npy.with_suffix(".modern_emb.txt")
        stats = write_glove_format(vocab_from_vocab2id(vec_dir / "vocab2id.json"), embed_fn, glove_path,
                                   skip_tokens={"PAD", "UNK"})
        dim = stats["dim"]

    proc = subprocess.run(
        [sys.executable, "build_pretrained_w2v.py", "-emb", str(Path(glove_path).resolve()),
         "-data_dir", str(vec_dir.resolve()), "-out", str(out_npy.resolve())],
        cwd=str(_bamnet_src()), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("build_pretrained_w2v.py failed:\n" + proc.stderr[-2000:])
    return {"out_npy": str(out_npy), "dim": dim, "stdout": proc.stdout.strip()}


def run(recipes_df: pd.DataFrame, work_dir: str | Path, *,
        recipe_filter: Optional[set[str]] = None, max_dishes_per_tag: Optional[int] = 500,
        num_epochs: int = 50, glove_path: Optional[str | Path] = None,
        embed_fn: Optional[Callable] = None, eval_split: str = "valid",
        no_cuda: bool = True, resume: bool = True) -> dict:
    """Run the full Path-A pipeline; return KBQA counts, train stats and metrics.

    Set ``no_cuda=False`` to train/eval on a GPU (e.g. on Colab) — BAMnet still
    falls back to CPU if CUDA is unavailable.

    ``resume=True`` (default) makes the pipeline re-entrant: KBQA generation and
    vectorisation are skipped when their outputs exist, and the modern-embedding
    file is appended word-by-word — so a re-run after an interrupted Colab session
    only does the remaining work.
    """
    work_dir = Path(work_dir)
    kbqa_dir = work_dir / "kbqa"
    vec_dir = work_dir / "vec"
    model_file = work_dir / "model" / "bamnet.model"

    kb_path = kbqa_dir / "recipe_kg.json"
    qa_files = [kbqa_dir / f"{s}_qas.json" for s in ("train", "valid", "test")]
    if resume and kb_path.exists() and all(f.exists() for f in qa_files):
        gen = {"kb_path": str(kb_path),
               "qa_counts": {s: sum(1 for _ in open(kbqa_dir / f"{s}_qas.json", encoding="utf-8"))
                             for s in ("train", "valid", "test")}}
    else:
        gen = generate_kbqa(recipes_df, kbqa_dir, recipe_filter=recipe_filter,
                            max_dishes_per_tag=max_dishes_per_tag)
    build_all_data(kbqa_dir, gen["kb_path"], vec_dir, resume=resume)

    pre_word2vec, overrides = "", {}
    if glove_path is not None or embed_fn is not None:
        emb = build_embeddings(vec_dir, work_dir / "emb.npy", glove_path=glove_path, embed_fn=embed_fn)
        pre_word2vec = emb["out_npy"]
        if emb["dim"]:
            overrides["vocab_embed_size"] = emb["dim"]

    train_stats = train(vec_dir, model_file=model_file, pre_word2vec=pre_word2vec,
                        num_epochs=num_epochs, no_cuda=no_cuda, overrides=overrides)
    metrics = evaluate(vec_dir, model_file=model_file, split=eval_split,
                       no_cuda=no_cuda, overrides=overrides)
    metrics.pop("predictions", None)
    metrics.pop("gold", None)
    variant = "pfoodreq_modernemb" if embed_fn is not None else "pfoodreq"
    return {"variant": variant, "kbqa_counts": gen["qa_counts"], "train": train_stats, "metrics": metrics}
