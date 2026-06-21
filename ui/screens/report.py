"""Flow C: Report tab (Evaluation Protocol §11).

Interactive view over pre-computed evaluation results (§11.2 — read-only, no live
computation). Panels render conditionally on available data (§12.3 minimum =
EXP-1 + EXP-3 + Safety). The aggregate ↔ stratified toggle is the key
interaction (§3.4 skew).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import eval_results
from ..state import goto
from ..theme import C, F, FP, html

_STRATA = ["aggregate", "tight", "medium", "broad"]
_METRIC_LABEL = {
    "hit_at_10": "Hit@10", "mrr_at_10": "MRR@10", "ndcg_at_10": "nDCG@10",
    "avg_nutri_at_10": "Avg NutriScore@10", "avg_fsa_at_10": "Avg FSA@10",
    "allergen_violation": "Allergen violation ↓", "hard_constraint_violation": "Hard-constraint violation ↓",
    "constraint_satisfaction": "Constraint satisfaction", "correctness": "Correctness",
    "faithfulness": "Faithfulness", "relevancy": "Relevancy", "structural_validity": "Structural validity",
    "explanation_quality": "Explanation quality", "intra_plan_diversity": "Intra-plan diversity",
    "calorie_adherence": "Calorie adherence", "llm_usage": "LLM calls", "latency_s": "Latency (s)",
}
_CONFIG_LABEL = {
    "dense": "Dense", "sparse": "Sparse (BM25)", "hybrid": "Hybrid",
    "text-to-pandas": "Text→Pandas", "fusion": "Fusion (3-way)",
    "rag": "Plain RAG", "heuristic_1": "Heuristic 1 (full agentic)", "heuristic_2": "Heuristic 2 (recommend only)",
}


def _label(mapping, key):
    return mapping.get(key, key)


def _pivot(df: pd.DataFrame, index: str, columns: str, value: str = "mean") -> pd.DataFrame:
    p = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    p.index = [_label(_CONFIG_LABEL, i) if index == "config_id" else i for i in p.index]
    if columns == "metric":
        p.columns = [_label(_METRIC_LABEL, c) for c in p.columns]
    return p


# ── panels ─────────────────────────────────────────────────────────────────
def _overview(agg: pd.DataFrame) -> None:
    exp = st.selectbox("Experiment", sorted(agg["experiment_id"].unique()), key="rep_ov_exp")
    sub = agg[(agg["experiment_id"] == exp) & (agg["stratum"] == "aggregate")]
    if sub.empty:
        st.info("No aggregate-stratum rows for this experiment.")
        return
    st.caption("Headline metrics per configuration (aggregate stratum).")
    table = _pivot(sub, "config_id", "metric").round(3)
    st.dataframe(table, use_container_width=True)


def _retriever_comparison(agg: pd.DataFrame) -> None:
    exp1 = agg[agg["experiment_id"] == "EXP-1"]
    if exp1.empty:
        st.info("EXP-1 (retriever comparison) results not available.")
        return
    metrics = ["hit_at_10", "mrr_at_10", "ndcg_at_10", "avg_nutri_at_10", "avg_fsa_at_10"]
    metrics = [m for m in metrics if m in set(exp1["metric"])]
    chosen = st.multiselect("Metrics", metrics, default=metrics[:3],
                            format_func=lambda m: _METRIC_LABEL.get(m, m), key="rep_rc_metrics")
    view = st.radio("View", ["Aggregate", "Stratified"], horizontal=True, key="rep_rc_view")
    if not chosen:
        st.caption("Pick at least one metric.")
        return
    if view == "Aggregate":
        sub = exp1[(exp1["metric"].isin(chosen)) & (exp1["stratum"] == "aggregate")]
        st.bar_chart(_pivot(sub, "config_id", "metric"))
    else:
        metric = st.selectbox("Metric", chosen, format_func=lambda m: _METRIC_LABEL.get(m, m),
                              key="rep_rc_strat_metric")
        sub = exp1[(exp1["metric"] == metric) & (exp1["stratum"] != "aggregate")]
        chart = _pivot(sub, "config_id", "stratum")
        chart = chart[[s for s in _STRATA if s in chart.columns]]
        st.bar_chart(chart)
        st.caption("Tight = 1–50 valid answers (where health-aware recommendation matters most, §9.1).")


def _stratified(agg: pd.DataFrame) -> None:
    exp = st.selectbox("Experiment", sorted(agg["experiment_id"].unique()), key="rep_st_exp")
    sub0 = agg[agg["experiment_id"] == exp]
    metric = st.selectbox("Metric", sorted(sub0["metric"].unique()),
                          format_func=lambda m: _METRIC_LABEL.get(m, m), key="rep_st_metric")
    sub = sub0[(sub0["metric"] == metric) & (sub0["stratum"] != "aggregate")]
    if sub.empty:
        st.info("No stratified rows.")
        return
    chart = _pivot(sub, "config_id", "stratum")
    chart = chart[[s for s in _STRATA if s in chart.columns]]
    st.bar_chart(chart)
    st.caption("Aggregates are dominated by broad personas — always read alongside the tight stratum (§3.4).")


def _safety(agg: pd.DataFrame) -> None:
    st.caption("Deterministic, rule-based safety metrics (target = 0). Allergen detection uses the "
               "FoodOn hierarchy, not string matching (§4.3).")
    safety_metrics = ["allergen_violation", "hard_constraint_violation"]
    stratum = st.radio("Stratum", ["aggregate", "tight"], horizontal=True, key="rep_safety_stratum",
                       help="Tight ≈ allergen/constraint-heavy personas.")
    sub = agg[(agg["metric"].isin(safety_metrics)) & (agg["stratum"] == stratum)]
    if sub.empty:
        st.info("Safety metrics not available.")
        return
    for metric in safety_metrics:
        msub = sub[sub["metric"] == metric]
        if msub.empty:
            continue
        html(f"<div style=\"font-family:{F};font-weight:700;font-size:12px;color:{C['text']};margin:8px 0 4px\">"
             f"{_METRIC_LABEL[metric]}</div>")
        cols = st.columns(len(msub))
        for col, (_, row) in zip(cols, msub.sort_values("config_id").iterrows()):
            val = row["mean"]
            color = C["nutA"] if val <= 0.01 else (C["nutC"] if val < 0.1 else C["nutE"])
            col.markdown(
                f"<div style=\"text-align:center;border:1px solid {C['border']};border-radius:8px;padding:8px\">"
                f"<div style=\"font-family:{F};font-size:10px;color:{C['textMuted']}\">{_label(_CONFIG_LABEL, row['config_id'])}</div>"
                f"<div style=\"font-family:{F};font-size:20px;font-weight:800;color:{color}\">{val:.0%}</div></div>",
                unsafe_allow_html=True,
            )


def _agentic(agg: pd.DataFrame, pw: pd.DataFrame | None) -> None:
    exp3 = agg[agg["experiment_id"] == "EXP-3"]
    if exp3.empty:
        st.info("EXP-3 (meal-plan generators) results not available.")
        return
    metric = st.selectbox("Metric", sorted(exp3["metric"].unique()),
                          format_func=lambda m: _METRIC_LABEL.get(m, m), key="rep_ag_metric")
    stratum = st.radio("Stratum", _STRATA, horizontal=True, key="rep_ag_stratum")
    sub = exp3[(exp3["metric"] == metric) & (exp3["stratum"] == stratum)].sort_values("config_id")
    if sub.empty:
        st.info("No rows for this selection.")
        return
    chart = pd.DataFrame({"mean": sub["mean"].values},
                         index=[_label(_CONFIG_LABEL, c) for c in sub["config_id"]])
    st.bar_chart(chart)
    # Mean ± 95% CI table
    show = sub[["config_id", "mean", "ci_low", "ci_high", "per_question_variance"]].copy()
    show["config_id"] = show["config_id"].map(lambda c: _label(_CONFIG_LABEL, c))
    st.dataframe(show.round(3).rename(columns={"config_id": "config"}), use_container_width=True, hide_index=True)
    if pw is not None:
        sig = pw[(pw["metric"] == metric) & (pw["stratum"] == stratum)]
        for _, r in sig.iterrows():
            mark = "✅ significant" if r["significant"] else "ns"
            st.caption(f"{_label(_CONFIG_LABEL, r['config_a'])} vs {_label(_CONFIG_LABEL, r['config_b'])}: "
                       f"Δ={r['mean_diff']:+.3f}, {r['test']}, p={r['p_value']} "
                       f"(corrected {r['p_value_corrected']}) — {mark}")


def _cost_quality(agg: pd.DataFrame) -> None:
    exp3 = agg[(agg["experiment_id"] == "EXP-3") & (agg["stratum"] == "aggregate")]
    needed = {"llm_usage", "latency_s", "correctness"}
    if exp3.empty or not needed <= set(exp3["metric"]):
        st.info("Cost/quality metrics not available.")
        return
    x_axis = st.radio("Cost axis", ["llm_usage", "latency_s"], horizontal=True,
                      format_func=lambda m: _METRIC_LABEL[m], key="rep_cq_x")
    wide = exp3.pivot_table(index="config_id", columns="metric", values="mean")
    df = pd.DataFrame({
        "config": [_label(_CONFIG_LABEL, c) for c in wide.index],
        _METRIC_LABEL[x_axis]: wide[x_axis].values,
        "Correctness": wide["correctness"].values,
    })
    st.scatter_chart(df, x=_METRIC_LABEL[x_axis], y="Correctness", color="config")
    st.caption("Upper-left = better (more correct, cheaper). The agentic pipeline trades LLM cost for correctness "
               "and safety (EXP-3/§4).")


# ── entry ────────────────────────────────────────────────────────────────
def render() -> None:
    head_l, head_r = st.columns([4, 1])
    with head_l:
        html(f"<span class=\"foodrag-brand\" style=\"font-size:20px\">FoodRAG</span>"
             f"<span style=\"font-family:{F};font-size:12px;color:{C['textMuted']};margin-left:8px\">"
             f"Evaluation Report</span>")
    with head_r:
        if st.button("← Back", use_container_width=True):
            goto("welcome")

    html(f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};margin:4px 0 10px;line-height:1.6\">"
         f"Offline evaluation results (Evaluation Protocol §11). Pre-computed and read-only.</p>")

    if not eval_results.available():
        st.warning("No evaluation results found. Place `aggregate.csv` (+ optional `pairwise.csv`) in "
                   "`eval_results/` (or set `FOODRAG_EVAL_RESULTS`). See Evaluation Protocol §13 for the schema.")
        return

    if eval_results.is_sample():
        st.warning("⚠ Showing **bundled sample results (synthetic placeholders)** so the panels are explorable. "
                   "Replace `eval_results/*.csv` with real evaluation outputs to see real numbers.")

    agg = eval_results.load_aggregate()
    pw = eval_results.load_pairwise()

    tabs = st.tabs(["Overview", "Retriever comparison", "Stratified", "Safety", "Agentic", "Cost vs quality"])
    with tabs[0]:
        _overview(agg)
    with tabs[1]:
        _retriever_comparison(agg)
    with tabs[2]:
        _stratified(agg)
    with tabs[3]:
        _safety(agg)
    with tabs[4]:
        _agentic(agg, pw)
    with tabs[5]:
        _cost_quality(agg)
