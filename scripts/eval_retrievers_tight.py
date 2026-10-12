"""EXP-ret-tight — Text-to-Pandas retriever evaluation over the tight QA set.

Rebuilt driver. The script that produced the Jun/Jul 2026 ``ret_text2pandas@*``
artefacts is **not in the repository** (no script, no notebook, not in the
deleted-file history), so this is a reimplementation against
``foodrag.baselines.runner``, not a recovered original.

Three things it does that the original demonstrably did not
(see ``specs/checks-T-Q-H-findings.md``):

1. **Both backbones in one session, one ``out_dir``.** The old llama70b number
   lived in ``eval_results_llama/`` and gpt4o-mini in ``eval_results/``, produced
   a day apart, which is what made check H.2 unresolvable.

2. **Every LLM interaction is persisted per question-run** to a JSONL trace:
   the generated pandas expression, the row count, the error, and a classified
   ``empty_class``. The old artefacts carried no error field and left no entry in
   ``data/logs/llm_calls-*.jsonl``, so check Q.2's empty-class partition could
   not be reconstructed after the fact. ``empty_class`` is also written into the
   per-question parquet as a first-class column (check SCH), so the published
   artefact schema is health-aware on its own, without the sidecar.

3. **A per-run health gate.** The old gpt4o-mini job averaged two runs that were
   100% empty into a published headline (0.427096) without complaint. Here a run
   whose empty rate exceeds ``--max-empty-rate`` aborts the job.

Usage::

    foodrag-eval-ret --smoke                    # 20 questions x 1 run, both backbones
    foodrag-eval-ret --backbone llama70b        # one backbone, full
    foodrag-eval-ret                            # full: both backbones x 5 runs

Nothing here writes to ``eval_results/``; the default target is
``eval_results_ret_rerun/`` so the published artefacts stay untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ── empty-result classification (check Q.2's partition) ──────────────────────
#
# The four classes the spec asked for, plus "ok". The distinction that matters
# is CODE_ERROR/NO_CODE/LLM_ERROR (failures) vs ZERO_ROWS (correct behaviour for
# an unsatisfiable persona). Conflating them is what made the old 55.02% rate
# uninterpretable.
OK = "ok"                    # rows returned
LLM_ERROR = "llm_error"      # the LLM call itself raised (after retries)
NO_CODE = "no_code"          # response contained nothing parseable as code
CODE_ERROR = "code_error"    # code produced, but it raised / would not eval
ZERO_ROWS = "zero_rows"      # code produced, ran clean, matched nothing


@dataclass
class Trace:
    """Per-question-run record of what the LLM actually did."""
    path: Path
    rows: list[dict] = field(default_factory=list)

    def add(self, **kw: Any) -> None:
        self.rows.append(kw)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        self.rows.clear()


def _instrumented_output_processor(output: str, df: pd.DataFrame, verbose: bool = False):
    """``pandas_retriever.default_output_processor`` with the failure reason kept.

    Mirrors ``src/foodrag/food_rag/retriever/pandas_retriever.py:72-134`` exactly
    in what it *returns* (an empty frame with the original columns on any
    failure), but also reports which stage failed so the empty can be classified.

    Returns ``(result_df, empty_class, error_str)``.
    """
    import ast

    import numpy as np
    from llama_index.core.output_parsers.utils import parse_code_markdown
    from llama_index.experimental.exec_utils import safe_eval, safe_exec

    empty = pd.DataFrame(columns=df.columns)

    try:
        parsed = parse_code_markdown(output, only_last=True)
        code = parsed[0] if parsed else ""
    except Exception as exc:  # noqa: BLE001
        return empty, NO_CODE, f"{type(exc).__name__}: {exc}"
    if not str(code).strip():
        return empty, NO_CODE, "parse_code_markdown produced no code"

    local_vars = {"df": df, "pd": pd}
    global_vars = {"np": np}

    try:
        tree = ast.parse(code)
        if not tree.body:
            return empty, NO_CODE, "empty AST"
        module = ast.Module(tree.body[:-1], type_ignores=[])
        safe_exec(ast.unparse(module), {}, local_vars)
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)
        if module_end_str.strip("'\"") != module_end_str:
            module_end_str = safe_eval(module_end_str, global_vars, local_vars)
    except Exception as exc:  # noqa: BLE001 — matches the library's bare except
        return empty, CODE_ERROR, f"{type(exc).__name__}: {exc}"

    # The library tries safe_eval, then falls back to eval / pd.eval. Same order.
    out_df = None
    err: Optional[str] = None
    for fn in (
        lambda: safe_eval(module_end_str, global_vars, local_vars),
        lambda: eval(module_end_str, global_vars, local_vars),  # noqa: S307 — as upstream
        lambda: pd.eval(module_end_str, local_dict=local_vars, global_dict=global_vars),
    ):
        try:
            out_df = fn()
            err = None
            break
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
    if out_df is None:
        return empty, CODE_ERROR, err

    # Mirror PandasInstructionParser.parse: project back onto the source frame by
    # index, so a partial-column result still resolves to whole recipes.
    try:
        result_df = df[df.index.isin(getattr(out_df, "index", []))]
    except Exception as exc:  # noqa: BLE001
        return empty, CODE_ERROR, f"{type(exc).__name__}: {exc}"

    return result_df, (OK if len(result_df) else ZERO_ROWS), None


# ── the baseline ─────────────────────────────────────────────────────────────

def build_cleaned_df(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """The frame the Text-to-Pandas retriever sees.

    Mirrors ``scripts/eval_1.py:169-178``. Note it DROPS ``recipe_url`` — ids are
    recovered through the preserved index, not from a column, so the index of
    ``recipes_df`` is load-bearing here.
    """
    from foodrag.food_rag.utils import hummus_docify
    from foodrag.utils import hummus_util

    df = recipes_df.copy()
    df = hummus_docify.stringify_nested_lists(df)
    df = hummus_docify.stringify_nested_dicts(df)
    df = df.rename(columns={
        "duration": "duration [min]",
        "ingredients": "ingredient_quantities",
        "ingredient_names": "ingredients",
        "direction_size": "number_of_directions",
        "ingredients_sizes": "number_of_ingredients",
    })
    df = df.drop(columns=[c for c in ("last_changed_date", "food_kg_locator",
                                      "recipe_url", "normalization_comment")
                          if c in df.columns])
    return hummus_util.clean_serves_to_int(df)


class Text2PandasBaseline:
    """``ret_text2pandas@<backbone>`` — one LLM-generated pandas query per question.

    Implements the ``foodrag.baselines.base.Baseline`` interface. Every call
    appends one row to ``trace``, whether it succeeded or not.
    """

    rung = "1"

    def __init__(self, cleaned_df: pd.DataFrame, url_by_index: pd.Series,
                 llm: Any, backbone: str, trace: Trace, k: int = 10):
        from foodrag.food_rag.retriever import pandas_retriever

        self.config_id = f"ret_text2pandas@{backbone}"
        self._df = cleaned_df
        self._urls = url_by_index
        self._trace = trace
        self._k = k
        self.run = 1
        self._engine = pandas_retriever.PandasQueryRetriever(
            df=cleaned_df, llm=llm, similarity_top_k=k, verbose=False,
        )

    def recommend(self, qa_item: dict, k: int) -> list[str]:
        question = qa_item.get("question") or ""
        qid = qa_item.get("_qid")
        t0 = time.perf_counter()
        code = None
        klass = None
        err = None
        ids: list[str] = []

        try:
            # Go through query() rather than retrieve() so the generated
            # expression in .metadata survives; retrieve() drops it.
            resp = self._engine.query(question)
            meta = getattr(resp, "metadata", None) or {}
            code = meta.get("pandas_instruction_str")
            raw = meta.get("raw_pandas_output")

            if code is None:
                klass, err = NO_CODE, "no pandas_instruction_str in response metadata"
                result = self._df.iloc[:0]
            else:
                result, klass, err = _instrumented_output_processor(code, self._df)
                # Trust the engine's own output for scoring; our re-execution is
                # for classification only. Fall back to it when the engine gave
                # us nothing usable.
                if isinstance(raw, pd.DataFrame) and len(raw):
                    result = self._df[self._df.index.isin(raw.index)]
                    klass = OK if len(result) else klass

            result = result.head(k)
            ids = [u for u in self._urls.reindex(result.index).tolist()
                   if isinstance(u, str)]
        except Exception as exc:  # noqa: BLE001 — an empty must never be silent
            klass = LLM_ERROR
            err = f"{type(exc).__name__}: {exc}"
            if self._trace is not None:
                err += " | " + traceback.format_exc(limit=3).replace("\n", " ⏎ ")

        if not ids and klass == OK:
            klass = ZERO_ROWS

        self._trace.add(
            config_id=self.config_id, run=self.run, question_id=qid,
            question=question, generated_code=code, n_retrieved=len(ids),
            retrieved_ids=ids, empty_class=klass, error=err,
            latency_s=round(time.perf_counter() - t0, 4),
        )
        return ids


# ── empty_class as a first-class per-question field (check SCH) ──────────────

def attach_empty_class(records: list[dict], rows: list[dict]) -> list[dict]:
    """Copy ``empty_class`` from this run's trace rows onto the scored records.

    The trace and the per-question records are two views of the same
    ``(config_id, run, question_id)`` population — the trace is written by
    ``Text2PandasBaseline.recommend`` and the records by
    ``runner.run_baseline`` — so the join is total in both directions. It is
    asserted rather than assumed: a silent left-join miss here would put a
    ``None`` into the published schema, which is precisely the health-blindness
    this field exists to remove.

    Mutates and returns ``records`` so ``runner.write_results`` persists
    ``empty_class`` into the per-question parquet rather than leaving it in the
    sidecar (E-1'.4).
    """
    by_key = {(r["config_id"], r["run"], r["question_id"]): r.get("empty_class")
              for r in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"trace keys are not unique: {len(rows)} rows, "
                         f"{len(by_key)} distinct (config_id, run, question_id)")

    missing = []
    for rec in records:
        key = (rec["config_id"], rec["run"], rec["question_id"])
        klass = by_key.pop(key, _MISSING)
        if klass is _MISSING:
            missing.append(key)
        else:
            rec["empty_class"] = klass
    if missing or by_key:
        raise ValueError(
            f"trace/record join is not total: {len(missing)} record(s) with no "
            f"trace row (e.g. {missing[:3]}), {len(by_key)} trace row(s) with no "
            f"record (e.g. {list(by_key)[:3]})")

    # The class must agree with the artefact it annotates, or it is decoration.
    bad = [rec["question_id"] for rec in records
           if bool(rec["retrieved_ids"]) != (rec["empty_class"] == OK)]
    if bad:
        raise ValueError(f"{len(bad)} record(s) whose empty_class disagrees with "
                         f"retrieved_ids (e.g. {bad[:3]})")
    return records


_MISSING = object()


# ── run health ───────────────────────────────────────────────────────────────

def run_health(records: list[dict]) -> dict:
    """Empty rate + class histogram for one run."""
    n = len(records)
    empty = sum(1 for r in records if not r["retrieved_ids"])
    return {"n": n, "empty": empty, "empty_rate": (empty / n) if n else 0.0}


def trace_health(rows: list[dict]) -> dict:
    hist: dict[str, int] = {}
    for r in rows:
        hist[r.get("empty_class") or "?"] = hist.get(r.get("empty_class") or "?", 0) + 1
    return hist


# ── driver ───────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default="data")
    p.add_argument("--qa-set", default="data/qa/main_10k_tight/recipe_question_answer_set.pkl")
    p.add_argument("--out-dir", default="eval_results_ret_rerun")
    p.add_argument("--backbone", action="append",
                   help="backbone name from models.yml retriever_backbones "
                        "(repeatable; default: all of them)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, help="only the first N questions (smoke)")
    p.add_argument("--workers", type=int, default=8, help="concurrent LLM calls")
    p.add_argument("--max-empty-rate", type=float, default=0.50,
                   help="abort if a run's empty rate exceeds this (0-1)")
    p.add_argument("--allow-degraded", action="store_true",
                   help="warn instead of aborting when the health gate trips")
    p.add_argument("--smoke", action="store_true",
                   help="shorthand for --limit 20 --runs 1")
    args = p.parse_args(argv)

    if args.smoke:
        args.limit, args.runs = args.limit or 20, 1

    from foodrag.baselines import runner
    from foodrag.config import settings
    from foodrag.llm import call_log
    from foodrag.utils import load_util

    settings.load_env()
    config = settings.load_models_config()
    names = args.backbone or [b["name"] for b in config.retriever_backbones]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep this job's LLM call log separate from the day's shared log.
    call_log.configure(out_dir / "llm_calls.jsonl")

    print(f"[setup] loading recipes from {args.data_path} ...", flush=True)
    recipes_df = runner.load_recipes_df(Path(args.data_path, "hummus", "recipes_df.csv"))
    url_by_index = recipes_df["recipe_url"]
    cleaned = build_cleaned_df(recipes_df)
    print(f"[setup] recipes: {len(recipes_df):,}  columns given to the LLM: {len(cleaned.columns)}",
          flush=True)

    qa_set = load_util.load_file(Path(args.qa_set))
    if args.limit:
        qa_set = qa_set[:args.limit]
    print(f"[setup] questions: {len(qa_set):,}   backbones: {names}   runs: {args.runs}",
          flush=True)

    health: list[dict] = []
    failed = False

    for name in names:
        spec = config.backbone(name)
        llm = settings.build_llm(spec=spec, config=config)
        config_id = f"ret_text2pandas@{name}"
        trace = Trace(out_dir / f"{config_id}_trace.jsonl")
        # question_id must match the prefix convention in the published artefacts.
        for qi, item in enumerate(qa_set):
            item["_qid"] = f"{config_id}_{qi:05d}"

        all_records: list[dict] = []
        for run in range(1, args.runs + 1):
            baseline = Text2PandasBaseline(cleaned, url_by_index, llm, name, trace, k=args.k)
            baseline.run = run
            t0 = time.perf_counter()
            print(f"\n[{config_id}] run {run}/{args.runs} ...", flush=True)
            records = runner.run_baseline(
                baseline, qa_set, recipes_df, k=args.k,
                experiment_id="EXP-ret-tight", run=run,
                id_prefix=config_id, max_workers=args.workers,
            )
            rows = list(trace.rows)
            trace.flush()
            # empty_class is a persisted per-question field, not a sidecar-only
            # annotation (check SCH, route A).
            attach_empty_class(records, rows)
            all_records.extend(records)

            h = run_health(records) | {"config_id": config_id, "run": run,
                                       "classes": trace_health(rows),
                                       "wall_s": round(time.perf_counter() - t0, 1)}
            health.append(h)
            print(f"[{config_id}] run {run}: empty {h['empty']}/{h['n']} "
                  f"({h['empty_rate']*100:.2f}%)  classes={h['classes']}  "
                  f"{h['wall_s']}s", flush=True)

            # ── the gate the old job did not have ──
            if h["empty_rate"] > args.max_empty_rate:
                msg = (f"[HEALTH] {config_id} run {run} empty rate "
                       f"{h['empty_rate']*100:.2f}% exceeds "
                       f"--max-empty-rate {args.max_empty_rate*100:.0f}%. "
                       f"This is what produced the discarded 0.427096.")
                if args.allow_degraded:
                    print(msg + " (continuing: --allow-degraded)", file=sys.stderr, flush=True)
                else:
                    print(msg + " Aborting.", file=sys.stderr, flush=True)
                    failed = True
                    break

        if all_records:
            agg = runner.aggregate(all_records, experiment_id="EXP-ret-tight",
                                   config_id=config_id)
            paths = runner.write_results(all_records, agg, out_dir, config_id=config_id)
            hit = agg[(agg.stratum == "aggregate") & (agg.metric == f"hit_at_{args.k}")]
            if len(hit):
                r = hit.iloc[0]
                print(f"[{config_id}] hit@{args.k} = {r['mean']:.6f} "
                      f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]  "
                      f"n={r['n_questions']} runs={r['n_runs']}", flush=True)
            print(f"[{config_id}] wrote {paths['per_question']}", flush=True)
        if failed:
            break

    (out_dir / "run_health.json").write_text(
        json.dumps(health, indent=2), encoding="utf-8")
    print(f"\n[done] health summary -> {out_dir/'run_health.json'}")
    print(f"[done] llm call log   -> {out_dir/'llm_calls.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
