"""Modern-embedding ablation for pFoodReq (§5.8).

The faithful pFoodReq run uses its original **GloVe 840B.300d** (a deliberate
between-method difference vs FoodRAG, not a confound). For thoroughness we also
run pFoodReq with a **modern embedding** as an ablation, changing exactly one
variable: the per-word embedding matrix BAMnet loads.

pFoodReq's ``build_pretrained_w2v.py`` builds that matrix by scanning a
GloVe-format text file (``word v1 v2 … vd`` per line) and keeping vectors whose
word is in ``vocab2id``. So the cleanest, most faithful ablation is to emit a
GloVe-format file from a modern embedder over the *same* vocabulary — then point
the pFoodReq config's ``-emb`` at it and set ``word_emb_dim`` to the new dim.
Everything else in BAMnet is unchanged.

A subword/transformer embedder embeds any token, so this variant also has **0%
OOV** by construction — which is precisely why pairing it with the GloVe OOV
report (``glove_oov.py``) lets us attribute any gap to *method* vs *vocabulary
coverage*.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def vocab_from_vocab2id(path: str | Path) -> list[str]:
    """Read the word list from a pFoodReq ``vocab2id.json`` (built by build_all_data)."""
    with open(path, "r", encoding="utf-8") as fh:
        return list(json.load(fh).keys())


def write_glove_format(vocab: Iterable[str], embed_fn: EmbedFn, out_path: str | Path,
                       *, batch_size: int = 256, skip_tokens: Optional[set[str]] = None) -> dict:
    """Embed each vocab word and write a GloVe-format file consumable by pFoodReq.

    :param embed_fn: maps a batch of words → a batch of equal-length float vectors.
    :param skip_tokens: reserved tokens to omit (e.g. ``PAD``/``UNK``); pFoodReq
        initialises those randomly anyway.
    :returns: ``{n_words, dim, out_path}``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip_tokens = skip_tokens or set()
    # GloVe text format is space-delimited, so a token containing whitespace can't
    # be represented (and would be OOV in real GloVe too — left to random init).
    words = [w for w in vocab if w and w not in skip_tokens and not any(c.isspace() for c in w)]

    dim: Optional[int] = None
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for start in range(0, len(words), batch_size):
            batch = words[start:start + batch_size]
            vectors = embed_fn(batch)
            for word, vec in zip(batch, vectors):
                if dim is None:
                    dim = len(vec)
                # GloVe text format: word then space-separated floats, one per line.
                fh.write(word + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
                n_written += 1

    return {"n_words": n_written, "dim": dim or 0, "out_path": str(out_path)}


def project_embedder(model: Optional[object] = None) -> EmbedFn:
    """Adapt FoodRAG's configured embedding model to an :data:`EmbedFn`.

    Uses ``settings.build_embedding()`` by default — the same modern model family
    FoodRAG uses — so the ablation contrasts GloVe against FoodRAG's embedding
    space directly. Embeds each vocab word as a standalone string (BAMnet needs a
    static per-word table, not contextual vectors).
    """
    if model is None:
        from foodrag.config.settings import build_embedding
        model = build_embedding()

    def _embed(batch: Sequence[str]) -> list[list[float]]:
        if hasattr(model, "get_text_embedding_batch"):
            return [list(v) for v in model.get_text_embedding_batch(list(batch))]
        return [list(model.get_text_embedding(w)) for w in batch]

    return _embed
