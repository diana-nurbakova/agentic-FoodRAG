"""GloVe out-of-vocabulary analysis over HUMMUS food vocabulary.

pFoodReq is kept on its original **GloVe 840B.300d** embeddings as part of the
faithful-reproduction commitment (Path A, §5.8). To keep the comparison honest we
must be able to separate *"pFoodReq's method underperforms"* from *"GloVe simply
couldn't represent HUMMUS's food vocabulary."* This module measures the latter.

It reproduces pFoodReq's own vocabulary + lookup:

* tokenization — ``word_tokenize`` after replacing ``\\ - /`` with spaces
  (``BAMnet/src/core/utils/utils.py``); a regex fallback is used when nltk's
  ``punkt`` data is unavailable (documented approximation).
* vocabulary — content tokens from entity *names* (ingredients, tags, recipe
  titles), lowercased, kept at ``min_freq`` (build_all_data uses 2).
* hit/OOV — a token is covered iff its lowercase form appears in the GloVe vocab,
  matching ``generic_utils.load_embeddings`` ("hit ratio").

The authoritative in-pipeline number is what ``build_pretrained_w2v.py`` prints
on the real 840B file over the built ``vocab2id.json``; this tool gives the same
quantity standalone (and supports a clearly-labelled smaller-GloVe proxy).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

_SEP_RE = re.compile(r"[\\\-/]")
_WORD_RE = re.compile(r"[a-z0-9%]+")

DEFAULT_SOURCES = ("ingredient_names", "tags", "title")


def _resolve_word_tokenize():
    """Resolve nltk's ``word_tokenize`` once (None if nltk/punkt unavailable)."""
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("warm up")  # force punkt load / fail fast
        return word_tokenize
    except Exception:
        return None


_WORD_TOKENIZE = _resolve_word_tokenize()


def tokenize(text: str, *, use_nltk: bool = True) -> list[str]:
    """Lowercase + pFoodReq-style tokenization (nltk if available, else regex).

    The regex backend closely approximates nltk's ``word_tokenize`` for this
    food vocabulary and is ~100× faster; the authoritative reproduction number is
    ``build_pretrained_w2v.py``'s hit ratio regardless. Resolution is cached
    module-level so there is no per-call import/except overhead.
    """
    text = _SEP_RE.sub(" ", str(text).lower())
    if use_nltk and _WORD_TOKENIZE is not None:
        return [t for t in _WORD_TOKENIZE(text) if _WORD_RE.fullmatch(t)]
    return _WORD_RE.findall(text)


def extract_food_vocab(recipes_df: pd.DataFrame, *, sources: Iterable[str] = DEFAULT_SOURCES,
                       min_freq: int = 2) -> dict[str, Counter]:
    """Per-source token frequency Counters over the named columns.

    List columns (``ingredient_names``, ``tags``) are flattened; scalar columns
    (``title``) are tokenized directly. Tokens below ``min_freq`` are dropped to
    mirror build_all_data's vocab construction.
    """
    counters: dict[str, Counter] = {}
    token_cache: dict[str, list[str]] = {}

    def _toks(item: str) -> list[str]:
        cached = token_cache.get(item)
        if cached is None:
            cached = tokenize(item)
            token_cache[item] = cached
        return cached

    for col in sources:
        if col not in recipes_df.columns:
            continue
        # Count distinct field values first — ingredient/tag strings repeat across
        # the 270K rows, so tokenizing uniques (cached) is ~orders faster than
        # per-row while giving identical token frequencies.
        value_counts: Counter = Counter()
        for value in recipes_df[col]:
            items = value if isinstance(value, (list, tuple)) else [value]
            value_counts.update(str(i) for i in items)

        counter: Counter = Counter()
        for value, n in value_counts.items():
            for tok in _toks(value):
                counter[tok] += n
        if min_freq > 1:
            counter = Counter({w: c for w, c in counter.items() if c >= min_freq})
        counters[col] = counter
    return counters


def load_glove_vocab(glove_path: str | Path) -> set[str]:
    """Stream the vocabulary (first column, lowercased) from a GloVe-format file."""
    vocab: set[str] = set()
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            sp = line.find(" ")
            if sp > 0:
                vocab.add(line[:sp].lower())
    return vocab


def compute_oov(counters: dict[str, Counter], glove_vocab: set[str],
                *, top_n: int = 30) -> dict:
    """Type- and token-weighted OOV against ``glove_vocab``, overall + per source."""
    overall: Counter = Counter()
    for c in counters.values():
        overall.update(c)

    def _report(counter: Counter) -> dict:
        n_types = len(counter)
        n_tokens = sum(counter.values())
        oov_types = {w: c for w, c in counter.items() if w not in glove_vocab}
        n_oov_tokens = sum(oov_types.values())
        return {
            "n_types": n_types,
            "n_tokens": n_tokens,
            "type_oov_rate": (len(oov_types) / n_types) if n_types else 0.0,
            "token_oov_rate": (n_oov_tokens / n_tokens) if n_tokens else 0.0,
            "type_hit_ratio": (1 - len(oov_types) / n_types) if n_types else 0.0,
        }

    report = _report(overall)
    report["per_source"] = {src: _report(c) for src, c in counters.items()}
    report["top_oov"] = Counter({w: c for w, c in overall.items()
                                 if w not in glove_vocab}).most_common(top_n)
    return report


def analyze(recipes_df: pd.DataFrame, glove_path: str | Path, *,
            sources: Iterable[str] = DEFAULT_SOURCES, min_freq: int = 2,
            top_n: int = 30) -> dict:
    """Convenience: extract HUMMUS food vocab and report OOV vs a GloVe file."""
    counters = extract_food_vocab(recipes_df, sources=sources, min_freq=min_freq)
    glove_vocab = load_glove_vocab(glove_path)
    report = compute_oov(counters, glove_vocab, top_n=top_n)
    report["glove_path"] = str(glove_path)
    report["glove_vocab_size"] = len(glove_vocab)
    return report


def analyze_with_gensim(recipes_df: pd.DataFrame, model_name: str = "glove-wiki-gigaword-300",
                        *, sources: Iterable[str] = DEFAULT_SOURCES, min_freq: int = 2,
                        top_n: int = 30) -> dict:
    """OOV against a gensim-downloaded GloVe as a **proxy** for the real 840B file.

    ``glove-wiki-gigaword-300`` is GloVe 6B, a different (smaller) corpus than the
    faithful ``glove.840B.300d``; the result is a labelled lower bound on coverage,
    not the reproduction number. Reported under ``proxy_model``.
    """
    import gensim.downloader as api

    kv = api.load(model_name)
    glove_vocab = {w.lower() for w in kv.key_to_index}
    counters = extract_food_vocab(recipes_df, sources=sources, min_freq=min_freq)
    report = compute_oov(counters, glove_vocab, top_n=top_n)
    report["proxy_model"] = model_name
    report["glove_vocab_size"] = len(glove_vocab)
    return report
