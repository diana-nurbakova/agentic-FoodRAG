"""foodrag command-line entrypoint.

Reproducible pipeline commands. Heavy data steps (init-hummus, embed, gen-qa,
eval) are driven from the notebooks/scripts; the commands here are the ones
that are safe and cheap to run directly.

    foodrag init-data [--prepare] [--verify-checksums] [--require-optional]
    foodrag models-check
    foodrag bench-embeddings [--smoke] [--top-k N] [--metric mrr|hit_rate|recall]
"""
from __future__ import annotations

import argparse
import sys


# --- init-data ------------------------------------------------------------
def _cmd_init_data(args) -> int:
    from foodrag.data.loader import DataLoader

    loader = DataLoader.from_manifest()
    if args.prepare:
        loader.prepare()
    report = loader.validate(
        require_optional=args.require_optional, verify_checksums=args.verify_checksums
    )
    print(report.format())
    return 0 if report.ok else 1


# --- models-check ---------------------------------------------------------
def _cmd_models_check(args) -> int:
    from foodrag.config import settings

    cfg = settings.load_models_config()
    print(f"LLM:       {cfg.llm['model']} ({cfg.llm['provider']})")
    print(f"Embedding: {cfg.embedding['model']} dim={cfg.embedding['dim']}")
    print(f"Judges:    {[m['name'] for m in cfg.judge_models]} (default {cfg.judges.get('default')})")

    ok = True
    try:
        llm = settings.build_llm()
        resp = str(llm.complete("Reply with exactly one word: pong")).strip()
        print(f"  LLM call OK -> {resp[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  LLM call FAILED -> {type(exc).__name__}: {exc}")
    try:
        emb = settings.build_embedding()
        v = emb.get_text_embedding("smoke test")
        print(f"  Embedding OK -> dim {len(v)}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  Embedding FAILED -> {type(exc).__name__}: {exc}")
    return 0 if ok else 1


# --- bench-embeddings -----------------------------------------------------
_SMOKE_CORPUS = [
    {"id": "oatmeal", "text": "Hearty oatmeal porridge with banana and honey for breakfast."},
    {"id": "pancakes", "text": "Fluffy buttermilk pancakes with maple syrup."},
    {"id": "eggs", "text": "Scrambled eggs with butter and chives."},
    {"id": "salmon", "text": "Grilled salmon fillet with lemon, dill and asparagus."},
    {"id": "stew", "text": "Slow-cooked beef stew with carrots and potatoes."},
    {"id": "salad", "text": "Caesar salad with romaine lettuce, croutons and parmesan."},
    {"id": "cake", "text": "Rich chocolate cake with chocolate ganache frosting."},
    {"id": "stirfry", "text": "Vegetable stir fry with tofu, broccoli and soy sauce."},
    {"id": "curry", "text": "Spicy chicken curry with coconut milk and rice."},
    {"id": "soup", "text": "Creamy roasted tomato soup with basil."},
    {"id": "yogurt", "text": "Greek yogurt with berries and granola."},
    {"id": "smoothie", "text": "Banana and spinach smoothie with almond milk."},
]
_SMOKE_QUERIES = [
    {"query": "a healthy light breakfast", "relevant_ids": ["oatmeal", "yogurt", "smoothie"]},
    {"query": "fish for dinner", "relevant_ids": ["salmon"]},
    {"query": "vegetarian main dish", "relevant_ids": ["stirfry", "soup", "salad"]},
    {"query": "sweet dessert", "relevant_ids": ["cake"]},
]


def _cmd_bench_embeddings(args) -> int:
    from foodrag.config import settings
    from foodrag.experiments.bench_embeddings import benchmark_embeddings, pick_winner

    cfg = settings.load_models_config()
    candidates = cfg.embedding_candidates
    if args.candidates:
        wanted = set(args.candidates.split(","))
        candidates = [c for c in candidates if c["name"] in wanted]

    if not args.smoke:
        print("Non-smoke benchmark needs the HUMMUS QA set (corpus + labelled queries),")
        print("which is produced by gen-qa. Re-run with --smoke for the synthetic check.")
        return 2

    results = benchmark_embeddings(
        _SMOKE_CORPUS, _SMOKE_QUERIES, candidates=candidates, top_k=args.top_k, config=cfg
    )
    print("\n=== bench-embeddings (smoke) ===")
    print(f"{'candidate':<14}{'dim':>6}{'hit@k':>8}{'mrr':>8}{'recall':>8}  note")
    for r in results:
        note = r.error[:40] if r.error else ""
        print(f"{r.name:<14}{(r.dim or 0):>6}{r.hit_rate:>8}{r.mrr:>8}{r.recall:>8}  {note}")
    winner = pick_winner(results, metric=args.metric)
    if winner:
        print(f"\nWinner by {args.metric}: {winner.name} ({winner.model})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foodrag")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init-data", help="validate/prepare external datasets")
    p.add_argument("--prepare", action="store_true", help="unzip local archives + download public files")
    p.add_argument("--verify-checksums", action="store_true")
    p.add_argument("--require-optional", action="store_true")
    p.set_defaults(func=_cmd_init_data)

    p = sub.add_parser("models-check", help="smoke-test the configured LLM + embedding")
    p.set_defaults(func=_cmd_models_check)

    p = sub.add_parser("bench-embeddings", help="compare embedding candidates on a retrieval task")
    p.add_argument("--smoke", action="store_true", help="use the built-in synthetic corpus")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--metric", choices=["mrr", "hit_rate", "recall"], default="mrr")
    p.add_argument("--candidates", help="comma-separated candidate names to include")
    p.set_defaults(func=_cmd_bench_embeddings)

    return parser


# Pipeline stages delegate to their module's own argparse-based main().
_STAGE_MODULES = {
    "init-hummus": "foodrag.pipeline.init_hummus",
    "embed": "foodrag.pipeline.embed",
    "init-vector-store": "foodrag.pipeline.vector_store",
    "init-graph-stores": "foodrag.pipeline.graph_stores",
    "gen-qa": "foodrag.pipeline.gen_qa",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # delegate pipeline-stage commands to their module main()
    if argv and argv[0] in _STAGE_MODULES:
        import importlib

        module = importlib.import_module(_STAGE_MODULES[argv[0]])
        return module.main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        print("Commands: init-data, models-check, bench-embeddings, "
              + ", ".join(_STAGE_MODULES))
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
