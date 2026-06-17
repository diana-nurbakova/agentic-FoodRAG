"""Benchmark embedding models on a retrieval task, then lock the winner.

Each candidate is embedded into its OWN pgvector table (dims differ across
models, so there is no shared schema). Works on any corpus + labelled queries:
a tiny synthetic set for a smoke test, or the real HUMMUS QA set later.

    from foodrag.experiments.bench_embeddings import benchmark_embeddings
    results = benchmark_embeddings(corpus, queries)   # corpus/queries: see below
    # corpus:  [{"id": "r1", "text": "..."}, ...]
    # queries: [{"query": "...", "relevant_ids": ["r1", "r3"]}, ...]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from foodrag.config import settings
from foodrag.config.settings import ModelsConfig
from foodrag.stores import get_vector_store_provider


@dataclass
class CandidateResult:
    name: str
    model: str
    dim: Optional[int] = None
    hit_rate: float = 0.0
    mrr: float = 0.0
    recall: float = 0.0
    n_queries: int = 0
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def _safe_table(name: str) -> str:
    return "bench_" + re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def _metrics(retrieved: list[str], relevant: set[str], top_k: int) -> tuple[float, float, float]:
    """hit@k, reciprocal rank, recall@k for a single query."""
    hit = 0.0
    rr = 0.0
    for rank, rid in enumerate(retrieved[:top_k], start=1):
        if rid in relevant:
            hit = 1.0
            rr = 1.0 / rank
            break
    found = sum(1 for rid in retrieved[:top_k] if rid in relevant)
    recall = found / len(relevant) if relevant else 0.0
    return hit, rr, recall


def benchmark_embeddings(
    corpus: list[dict[str, str]],
    queries: list[dict[str, Any]],
    candidates: Optional[list[dict[str, Any]]] = None,
    top_k: int = 10,
    config: Optional[ModelsConfig] = None,
    vector_store: str = "pgvector",
    verbose: bool = True,
) -> list[CandidateResult]:
    """Embed the corpus with each candidate, run the labelled queries, score."""
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    config = config or settings.load_models_config()
    candidates = candidates or config.embedding_candidates
    provider = get_vector_store_provider(vector_store)
    provider.reset_database()  # one clean slate; each candidate gets its own table
    results: list[CandidateResult] = []

    for cand in candidates:
        res = CandidateResult(name=cand["name"], model=cand["model"], dim=cand.get("dim"))
        if verbose:
            print(f"\n[bench] {cand['name']} ({cand['model']})")
        try:
            embed = settings.build_embedding(spec=cand, config=config)
            corpus_vecs = embed.get_text_embedding_batch(
                [d["text"] for d in corpus], show_progress=False
            )
            res.dim = len(corpus_vecs[0])
            vs = provider.connect(table_name=_safe_table(cand["name"]), embed_dim=res.dim)
            vs.add([
                TextNode(text=d["text"], embedding=v, id_=str(d["id"]))
                for d, v in zip(corpus, corpus_vecs)
            ])

            hits = rrs = recalls = 0.0
            for q in queries:
                qv = embed.get_query_embedding(q["query"])
                out = vs.query(VectorStoreQuery(query_embedding=qv, similarity_top_k=top_k))
                retrieved = list(out.ids or [])
                h, rr, rc = _metrics(retrieved, set(map(str, q["relevant_ids"])), top_k)
                hits += h
                rrs += rr
                recalls += rc
            n = max(len(queries), 1)
            res.hit_rate = round(hits / n, 4)
            res.mrr = round(rrs / n, 4)
            res.recall = round(recalls / n, 4)
            res.n_queries = len(queries)
            if verbose:
                print(f"  dim={res.dim} hit@{top_k}={res.hit_rate} mrr={res.mrr} recall@{top_k}={res.recall}")
        except Exception as exc:  # noqa: BLE001 - one bad candidate shouldn't abort the sweep
            res.error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  SKIP: {res.error[:160]}")
        results.append(res)

    return results


def pick_winner(results: list[CandidateResult], metric: str = "mrr") -> Optional[CandidateResult]:
    ok = [r for r in results if r.error is None and r.n_queries > 0]
    if not ok:
        return None
    return max(ok, key=lambda r: getattr(r, metric))
