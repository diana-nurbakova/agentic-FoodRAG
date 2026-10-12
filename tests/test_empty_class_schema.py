"""Check SCH — ``empty_class`` is a first-class per-question column.

The E-1' re-run persisted the four-way empty-result partition to the
``*_trace.jsonl`` sidecar only, leaving the published parquet health-blind
(E-1'.4). These tests pin the fix at both ends: the driver's trace-to-record
join, and the fact that the column survives ``runner.write_results`` into the
parquet a future run would publish.

Offline end-to-end: the LLM step is stubbed, everything downstream of it —
``Trace``, ``attach_empty_class``, ``aggregate``, ``write_results`` — is the
real code path the full job takes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from foodrag.baselines import runner

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "eval_retrievers_tight",
    Path(__file__).resolve().parents[1] / "scripts" / "eval_retrievers_tight.py")
ert = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_retrievers_tight"] = ert
_SPEC.loader.exec_module(ert)


CONFIG_ID = "ret_text2pandas@stub"


@pytest.fixture
def recipes_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipe_id": i, "recipe_url": f"r{i}", "title": f"Recipe {i}",
         "tags": ["vegan"], "ingredient_names": ["rice"], "servingSize [g]": 100,
         "sugars [g]": 5, "who_score": 0.5, "fsa_score": 0.5, "nutri_score": 0.5,
         "average_rating": 4.0, "number_of_ratings": 10}
        for i in range(1, 5)
    ])


@pytest.fixture
def qa_set() -> list[dict]:
    """Four questions, one per outcome the partition must distinguish."""
    return [
        {"question": "ok", "persona": {}, "answer": ["r1"]},
        {"question": "zero rows", "persona": {}, "answer": ["r2"]},
        {"question": "code error", "persona": {}, "answer": ["r3"]},
        {"question": "llm error", "persona": {}, "answer": ["r4"]},
    ]


class StubBaseline:
    """Drives ``Trace`` exactly as ``Text2PandasBaseline`` does, without an LLM.

    The outcome per question is scripted so all four classes appear; the trace
    write is the real one, so the join under test is the real join.
    """

    rung = "1"
    config_id = CONFIG_ID

    def __init__(self, trace, plan):
        self._trace, self._plan, self.run = trace, plan, 1

    def recommend(self, qa_item, k):
        klass, ids = self._plan[qa_item["question"]]
        self._trace.add(
            config_id=self.config_id, run=self.run, question_id=qa_item["_qid"],
            question=qa_item["question"], generated_code="df.head()",
            n_retrieved=len(ids), retrieved_ids=ids, empty_class=klass,
            error=None if klass == ert.OK else "stubbed", latency_s=0.1)
        return ids


@pytest.fixture
def plan() -> dict:
    return {
        "ok": (ert.OK, ["r1"]),
        "zero rows": (ert.ZERO_ROWS, []),
        "code error": (ert.CODE_ERROR, []),
        "llm error": (ert.LLM_ERROR, []),
    }


def _run(tmp_path, recipes_df, qa_set, plan, runs=1):
    trace = ert.Trace(tmp_path / f"{CONFIG_ID}_trace.jsonl")
    for qi, item in enumerate(qa_set):
        item["_qid"] = f"{CONFIG_ID}_{qi:05d}"
    all_records = []
    for run in range(1, runs + 1):
        baseline = StubBaseline(trace, plan)
        baseline.run = run
        records = runner.run_baseline(baseline, qa_set, recipes_df, k=10,
                                      experiment_id="EXP-ret-tight", run=run,
                                      id_prefix=CONFIG_ID)
        rows = list(trace.rows)
        trace.flush()
        ert.attach_empty_class(records, rows)
        all_records.extend(records)
    return all_records


def test_column_reaches_the_parquet(tmp_path, recipes_df, qa_set, plan):
    """SCH.2: the driver writes empty_class natively into the per-question parquet."""
    records = _run(tmp_path, recipes_df, qa_set, plan, runs=2)
    agg = runner.aggregate(records, experiment_id="EXP-ret-tight", config_id=CONFIG_ID)
    paths = runner.write_results(records, agg, tmp_path, config_id=CONFIG_ID)

    per_q = pd.read_parquet(paths["per_question"])
    assert "empty_class" in per_q.columns
    assert len(per_q) == 8                      # 4 questions x 2 runs
    assert per_q["empty_class"].isna().sum() == 0
    assert set(per_q["empty_class"]) == {ert.OK, ert.ZERO_ROWS,
                                         ert.CODE_ERROR, ert.LLM_ERROR}
    # Both runs carry their own classes, not run 1's broadcast over both.
    assert per_q.groupby("run")["empty_class"].nunique().tolist() == [4, 4]


def test_column_agrees_with_the_sidecar(tmp_path, recipes_df, qa_set, plan):
    """The persisted column must equal the trace, row for row — the SCH.2 criterion."""
    records = _run(tmp_path, recipes_df, qa_set, plan, runs=2)
    agg = runner.aggregate(records, experiment_id="EXP-ret-tight", config_id=CONFIG_ID)
    paths = runner.write_results(records, agg, tmp_path, config_id=CONFIG_ID)

    per_q = pd.read_parquet(paths["per_question"])
    trace = pd.read_json(tmp_path / f"{CONFIG_ID}_trace.jsonl", lines=True)
    key = ["config_id", "run", "question_id"]
    merged = per_q.merge(trace[key + ["empty_class"]], on=key, suffixes=("", "_tr"))
    assert len(merged) == len(per_q)
    assert (merged["empty_class"] == merged["empty_class_tr"]).all()


def test_the_partition_separates_failure_from_an_unanswerable_question(
        tmp_path, recipes_df, qa_set, plan):
    """C2: three distinct empties that the twelve-column schema renders identical."""
    records = _run(tmp_path, recipes_df, qa_set, plan)
    empties = [r for r in records if not r["retrieved_ids"]]
    assert len(empties) == 3
    assert {r["empty_class"] for r in empties} == {ert.ZERO_ROWS, ert.CODE_ERROR,
                                                  ert.LLM_ERROR}
    # Without the column they are indistinguishable: same ids, same zero score.
    assert all(r["metrics"]["hit_at_10"] == 0.0 for r in empties)


def test_a_missing_trace_row_raises_rather_than_writing_a_null(
        tmp_path, recipes_df, qa_set, plan):
    """A silent left-join miss would put a null into the field's own schema."""
    trace = ert.Trace(tmp_path / "t.jsonl")
    for qi, item in enumerate(qa_set):
        item["_qid"] = f"{CONFIG_ID}_{qi:05d}"
    baseline = StubBaseline(trace, plan)
    records = runner.run_baseline(baseline, qa_set, recipes_df, k=10,
                                  experiment_id="EXP-ret-tight", run=1,
                                  id_prefix=CONFIG_ID)
    with pytest.raises(ValueError, match="not total"):
        ert.attach_empty_class(records, list(trace.rows)[:-1])


def test_a_class_contradicting_retrieved_ids_raises(tmp_path, recipes_df, qa_set, plan):
    """An annotation that disagrees with the artefact it annotates is decoration."""
    trace = ert.Trace(tmp_path / "t.jsonl")
    for qi, item in enumerate(qa_set):
        item["_qid"] = f"{CONFIG_ID}_{qi:05d}"
    baseline = StubBaseline(trace, plan)
    records = runner.run_baseline(baseline, qa_set, recipes_df, k=10,
                                  experiment_id="EXP-ret-tight", run=1,
                                  id_prefix=CONFIG_ID)
    rows = list(trace.rows)
    rows[1]["empty_class"] = ert.OK          # zero_rows mislabelled as ok
    with pytest.raises(ValueError, match="disagrees with"):
        ert.attach_empty_class(records, rows)
