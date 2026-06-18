"""Stage 1b: resumable LLM meal-type annotation of the recipes.

The legacy annotation in get_hummus is a single sequential loop over all ~270k
recipes with no checkpointing, so any interruption loses everything. This
runner instead:

- checkpoints every recipe by id to a RunStore JSONL (resume re-runs only the
  not-yet-done recipes);
- runs a small worker pool (default 2, matching Pagoda's in-flight cap);
- merges the result into recipes_df['meal_types'] and re-saves at the end.

Default LLM is Pagoda Ollama (free). Safe to Ctrl-C and re-run.

    foodrag annotate [--data-path data] [--limit N] [--workers 2]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional


# Only the fields relevant to meal-type classification are sent to the LLM.
# The legacy path sent the entire recipe dict (directions, nutrition, URLs,
# ...), bloating the prompt (~3k chars) for no benefit.
ANNOTATION_FIELDS = ("title", "ingredient_names", "tags", "description")


def recipe_payload(row_dict: dict) -> dict:
    payload = {k: row_dict.get(k) for k in ANNOTATION_FIELDS if k in row_dict}
    return payload or row_dict  # fall back to the full row if none of the fields exist


def annotate_one(row_dict: dict, llm: Any) -> list[str]:
    """Annotate a single recipe's meal types via the LLM (same prompt/parse as the legacy path)."""
    from foodrag.utils.hummus_util import MEAL_TYPE_ANNOTATE_TEMPLATE, try_parse

    prompt = MEAL_TYPE_ANNOTATE_TEMPLATE.format(recipe=str(recipe_payload(row_dict)))
    text = llm.complete(prompt).text
    try:
        json_str = text.split("```json")[1].split("```")[0].strip()
    except Exception:
        json_str = text
    try:
        parsed = try_parse(json_str)
    except SyntaxError:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    if isinstance(parsed, str):
        return [parsed]
    return []


def store_path_for(data_path: str) -> Path:
    # In a subdir so get_current_hummus's top-level file scan doesn't try to load it.
    return Path(data_path, "hummus", "annotations", "meal_type_annotations.jsonl")


def run(
    data_path: str = "data",
    *,
    limit: Optional[int] = None,
    workers: int = 2,
    llm: Any = None,
    hummus_data: Any = None,
    save: bool = True,
    chunk_size: int = 2000,
    verbose: bool = True,
) -> dict:
    """Annotate meal types resumably; returns {annotated, total, store}.

    ``llm`` and ``hummus_data`` are injectable for testing.
    """
    from foodrag.config import settings
    from foodrag.llm.runstore import RunStore
    from foodrag.utils import load_util

    llm = llm or settings.build_annotation_llm()
    if hummus_data is None:
        hummus_data = load_util.get_current_hummus(data_path)
    recipes_df = hummus_data["recipes_df"]
    if limit:
        recipes_df = recipes_df.head(limit)

    id_col = "recipe_id"
    store = RunStore(store_path_for(data_path))
    todo = [row for _, row in recipes_df.iterrows() if not store.is_done(str(row[id_col]))]
    if verbose:
        print(f"Annotating {len(todo)} of {len(recipes_df)} recipes ({len(store.results())} already done), "
              f"{workers} workers -> {store.path}")

    def work(row) -> None:
        rid = str(row[id_col])
        try:
            store.mark_ok(rid, annotate_one(row.to_dict(), llm))
        except Exception as exc:  # noqa: BLE001 - persist + continue
            store.mark_error(rid, f"{type(exc).__name__}: {exc}")

    # Process in chunks so we never hold 270k futures at once.
    done = 0
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, chunk))
        done += len(chunk)
        if verbose:
            print(f"  [{done}/{len(todo)}] checkpointed ({store.summary()})")

    if save:
        results = store.results()
        recipes_df = hummus_data["recipes_df"]
        recipes_df["meal_types"] = (
            recipes_df[id_col].astype(str).map(results).apply(lambda x: x if isinstance(x, list) else [])
        )
        load_util.store_file(recipes_df, Path(data_path, "hummus", "recipes_df"))

    return {"annotated": len(store.results()), "total": len(recipes_df), "store": str(store.path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable LLM meal-type annotation of recipes.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--limit", type=int, help="annotate only the first N recipes (calibration)")
    p.add_argument("--workers", type=int, default=2, help="concurrent workers (Pagoda allows 2)")
    p.add_argument("--no-save", action="store_true", help="checkpoint only; don't rewrite recipes_df.csv")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = run(data_path=args.data_path, limit=args.limit, workers=args.workers, save=not args.no_save)
    print(f"Annotated {out['annotated']}/{out['total']} recipes -> {out['store']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
