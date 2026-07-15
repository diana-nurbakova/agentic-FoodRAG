"""Flow C: Report tab (Evaluation Protocol §11; spec: FoodRAG_Report_Tab_Modifications).

Interactive view over the *real* pre-computed evaluation results (§11.2 — read-only,
no live computation). The tab leads with the headline findings and organises panels
into three tiers (spec §4):

  ① Headline    Overview (marquee) · Retriever comparison · Safety (before→after)
  ② Attribution Agentic comparison · Error taxonomy · Retrieval vs full pipeline
  ③ Depth       Pareto · Explainability · Judge reliability · Plan quality (gated) ·
                Guideline adherence (gated) · Time-of-day · Case browser · Volatility ·
                Stratified (demoted)

Real experiment ids: EXP-ret-tight (retrievers), EXP-7 (system + baselines),
EXP-10k* (baseline ladders), EXP-5 (KG). Descriptive titles, descriptions and the
config/metric label maps live in ``ui/report_copy.json`` (editable without code).
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from .. import eval_results
from ..components.pareto_chart import render_pareto
from ..services import report_data as rd
from ..state import goto
from ..theme import C, F, NUTRI_C, html

_STRATA = ["aggregate", "tight", "medium", "broad"]

# Experiments that anchor the repointed panels.
_EXP_RETRIEVER = "EXP-ret-tight"
_EXP_SYSTEM = "EXP-7"


# ── shared helpers ─────────────────────────────────────────────────────────
def _clabel(cid: str) -> str:
    return rd.config_label(cid)


def _mlabel(m: str) -> str:
    return rd.metric_label(m)


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt(v, fmt: str = "ratio") -> str:
    if _isnan(v):
        return "—"
    if fmt == "pct":
        return f"{v * 100:.1f}%"
    if fmt == "num":
        return f"{v:.1f}"
    return f"{v:.3f}"


def _pivot(df: pd.DataFrame, index: str, columns: str, value: str = "mean") -> pd.DataFrame:
    p = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    if index == "config_id":
        p.index = [_clabel(i) for i in p.index]
    if columns == "metric":
        p.columns = [_mlabel(c) for c in p.columns]
    return p


_SERIES_COLORS = [C["primary"], C["primaryLight"], C["accent"], C["nutC"], C["sub"]]


def _hbar(df: pd.DataFrame, grouped: bool = False, value_title: str = "") -> None:
    """Horizontal bar chart with **un-truncated** category labels.

    ``df`` is indexed by category label; each column is a data series. Native
    ``st.bar_chart`` truncates long y-axis labels at ~180px, so we render an
    Altair bar chart with the axis ``labelLimit`` lifted. With one column it's a
    simple bar; with several, bars are grouped side-by-side (``grouped``) or
    stacked (default)."""
    d = df.copy()
    d.index = [str(x) for x in d.index]
    order = list(d.index)
    d.index.name = "category"
    long = d.reset_index().melt(id_vars="category", var_name="series", value_name="value")
    n = d.shape[1]
    # Enough height for every band (Altair truncates to a fixed height otherwise).
    height = max(120, len(order) * (26 if n == 1 else 20 + 12 * n))
    y = alt.Y("category:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=1000, labelFontSize=11))
    x = alt.X("value:Q", title=value_title or None)
    enc = {"y": y, "x": x}
    if n > 1:
        enc["color"] = alt.Color("series:N", title=None,
                                 scale=alt.Scale(range=_SERIES_COLORS[:n]),
                                 sort=list(d.columns))
        if grouped:
            enc["yOffset"] = alt.YOffset("series:N", sort=list(d.columns))
    else:
        enc["color"] = alt.value(C["primary"])
    chart = alt.Chart(long).mark_bar().encode(**enc).properties(height=height)
    st.altair_chart(chart, use_container_width=True)


# ── Nutri-Score: normalized [0,1] ⇄ categorical A–E ────────────────────────
# The pipeline scores each recipe's Nutri-Score letter, then normalizes it to
# [0,1] as letter→{A:4,B:3,C:2,D:1,E:0} ÷ 4 (see src/foodrag/utils/score_util.py).
# So A=1.00, B=0.75, C=0.50, D=0.25, E=0.00; ``avg_nutri_at_10`` is the mean of
# those normalized values over the top-10 recipes.
_NUTRI_INT = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
_INT_NUTRI = {v: k for k, v in _NUTRI_INT.items()}
_NUTRI_METRICS = {"avg_nutri_at_10"}
# Decision bands for mapping a normalized mean back to a letter: the grade is the
# one whose anchor (A=1.0 … E=0.0) is closest, i.e. round(value×4). The cutoffs are
# the midpoints between adjacent anchors — 0.125 / 0.375 / 0.625 / 0.875.
_NUTRI_BANDS = [("E", 0.0, 0.125), ("D", 0.125, 0.375), ("C", 0.375, 0.625),
                ("B", 0.625, 0.875), ("A", 0.875, 1.0)]
_NUTRI_RULE = ("Nutri-Score is originally categorical (**A** best … **E** worst). It is normalised "
               "to [0,1] as A=1.00, B=0.75, C=0.50, D=0.25, E=0.00 (letter→{4,3,2,1,0} ÷ 4). The "
               "bars above are the mean of those normalised values over the top-10 recipes; the "
               "grade plot maps each mean to the **nearest** band — cutoffs at 0.125 / 0.375 / "
               "0.625 / 0.875 (the midpoints between grades), shown as the coloured background.")


def _nutri_letter(value) -> str | None:
    """Nearest Nutri-Score band (A–E) for a normalized [0,1] mean (None if NaN)."""
    if _isnan(value):
        return None
    idx = int(round(max(0.0, min(1.0, float(value))) * 4))
    return _INT_NUTRI.get(idx)


def _nutri_grade_chart(labels, values) -> None:
    """Companion plot: each config's mean NutriScore over a coloured A–E background
    (official Nutri-Score palette). The band a bar reaches into is its grade, also
    labelled at the bar end."""
    rows = [{"category": str(lab), "value": float(val), "letter": _nutri_letter(val)}
            for lab, val in zip(labels, values) if not _isnan(val)]
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = list(df["category"])
    domain = list("ABCDE")
    color_scale = alt.Scale(domain=domain, range=[NUTRI_C[l] for l in domain])

    # Background decision bands (full height, coloured by grade zone).
    bands = pd.DataFrame([{"letter": l, "lo": lo, "hi": hi} for l, lo, hi in _NUTRI_BANDS])
    band_layer = alt.Chart(bands).mark_rect(opacity=0.22).encode(
        x=alt.X("lo:Q", title="Avg NutriScore (0–1)", scale=alt.Scale(domain=[0, 1], nice=False)),
        x2="hi:Q",
        color=alt.Color("letter:N", scale=color_scale, sort=domain,
                        legend=alt.Legend(title="Nutri-Score band")),
    )
    base = alt.Chart(df).encode(
        y=alt.Y("category:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=1000, labelFontSize=11)),
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1], nice=False)),
    )
    bars = base.mark_bar(color=C["primary"], size=14)
    text = base.mark_text(align="left", dx=4, fontWeight="bold", color=C["text"]).encode(text="letter:N")
    chart = (band_layer + bars + text).properties(height=max(120, len(order) * 30))
    st.altair_chart(chart, use_container_width=True)


def _maybe_nutri_grades(metric: str, labels, values) -> None:
    """If ``metric`` is a Nutri-Score metric, render the A–E grade companion plot."""
    if metric not in _NUTRI_METRICS:
        return
    _section_label("As Nutri-Score grades (A–E)")
    _nutri_grade_chart(labels, values)
    st.caption(_NUTRI_RULE)


def _section_label(text: str) -> None:
    """Small bold sub-heading used above an individual chart."""
    html(f"<div style=\"font-family:{F};font-weight:700;font-size:12px;color:{C['text']};"
         f"margin:8px 0 4px\">{text}</div>")


def _stat_card(col, label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    color = {"good": C["nutA"], "bad": C["nutE"], "warn": C["warn"]}.get(tone, C["primary"])
    col.markdown(
        f"<div style=\"border:1px solid {C['border']};border-radius:10px;padding:12px 14px;"
        f"background:{C['card']};height:100%\">"
        f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};margin-bottom:2px\">{label}</div>"
        f"<div style=\"font-family:{F};font-size:26px;font-weight:800;color:{color};line-height:1.1\">{value}</div>"
        f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};margin-top:4px;line-height:1.4\">{note}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _distinct_strata(df: pd.DataFrame, metric: str) -> list[str]:
    """Strata (in canonical order) whose per-config means for ``metric`` actually
    differ. Collapses strata that are verbatim duplicates of one already kept —
    e.g. EXP-7's ``aggregate`` is a copy of ``tight`` — so a stratum toggle is
    only shown when it changes something."""
    present = [s for s in _STRATA if s in set(df["stratum"])]
    kept: list[str] = []
    seen: list[pd.Series] = []
    for s in present:
        vec = (df[(df["metric"] == metric) & (df["stratum"] == s)]
               .set_index("config_id")["mean"].sort_index())
        if any(vec.equals(prev) for prev in seen):
            continue
        kept.append(s)
        seen.append(vec)
    return kept or present[:1]


def _exp_select(agg: pd.DataFrame, key: str, only=None) -> str:
    """Descriptive experiment selector + description block (spec §3)."""
    exps = sorted(agg["experiment_id"].unique())
    if only:
        exps = [e for e in only if e in set(exps)] or exps
    exp = st.selectbox("Experiment", exps, format_func=lambda e: rd.exp_meta(e)["title"], key=key)
    meta = rd.exp_meta(exp)
    html(f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};"
         f"letter-spacing:.04em;margin:-4px 0 4px\">{meta.get('sub', exp)}</div>")
    if meta.get("desc"):
        st.caption(meta["desc"])
    return exp


def _ceiling_captions(configs) -> None:
    """Reference-ceiling / cost caveats for any flagged config in view (spec §7)."""
    for cid in configs:
        note = rd.ceiling_note(cid)
        if note:
            st.caption(f"⚑ **{_clabel(cid)}** — {note}")


# ═══ TIER 1 — HEADLINE ══════════════════════════════════════════════════════
def _overview(agg: pd.DataFrame) -> None:
    st.caption("Offline evaluation of FoodRAG against retrieval and baseline ladders "
               "(tight stratum, n=1,646; generator Llama-3.3-70B; dual-judge Gemma-3-27B / "
               "DeepSeek-V3). The marquee findings:")
    cards = rd.marquee_values(agg)
    cols = st.columns(len(cards)) if cards else []
    for col, card in zip(cols, cards):
        _stat_card(col, card["label"], _fmt(card["value"], card.get("fmt", "ratio")),
                   card.get("note", ""), card.get("tone", "neutral"))
    st.divider()
    st.markdown("**What's in this report**")
    for e in ("EXP-ret-tight", "EXP-7", "EXP-10k", "EXP-10k-tight", "EXP-10k-tog", "EXP-5"):
        if e in set(agg["experiment_id"]):
            m = rd.exp_meta(e)
            html(f"<div style=\"font-family:{F};font-size:12px;color:{C['text']};margin:2px 0\">"
                 f"<b>{m['title']}</b> "
                 f"<span style=\"color:{C['textMuted']}\">· {m.get('sub','')}</span></div>")


def _retriever_comparison(agg: pd.DataFrame) -> None:
    exp1 = agg[agg["experiment_id"] == _EXP_RETRIEVER]
    if exp1.empty:
        st.info("Retriever comparison results not available.")
        return
    m = rd.exp_meta(_EXP_RETRIEVER)
    html(f"<div style=\"font-family:{F};font-size:11px;color:{C['textMuted']};margin-bottom:4px\">"
         f"{m.get('sub','')}</div>")
    st.caption(m.get("desc", ""))
    metrics = ["hit_at_10", "mrr_at_10", "ndcg_at_10", "avg_nutri_at_10", "avg_fsa_at_10"]
    metrics = [x for x in metrics if x in set(exp1["metric"])]
    chosen = st.multiselect("Metrics", metrics, default=metrics[:3],
                            format_func=_mlabel, key="rep_rc_metrics")
    if not chosen:
        st.caption("Pick at least one metric.")
        return
    sub = exp1[(exp1["metric"].isin(chosen)) & (exp1["stratum"] == "aggregate")]
    # One plot per selected metric — horizontal bars so the (long) retriever
    # labels are never truncated.
    for met in chosen:
        msub = sub[sub["metric"] == met].sort_values("mean", ascending=False)
        if msub.empty:
            continue
        _section_label(_mlabel(met))
        chart = pd.DataFrame({_mlabel(met): msub["mean"].values},
                             index=[_clabel(c) for c in msub["config_id"]])
        _hbar(chart)
        _maybe_nutri_grades(met, [_clabel(c) for c in msub["config_id"]], msub["mean"].values)
    st.caption("Lightweight symbolic retrieval (Text→Pandas) dominates dense/sparse embeddings — "
               "hit@10 0.43 (GPT-4o-mini) vs 0.055 (dense). Tight = 1–50 valid answers, where "
               "health-aware recommendation matters most.")
    with st.expander("Full metric table"):
        table = _pivot(exp1[exp1["stratum"] == "aggregate"], "config_id", "metric").round(3)
        st.dataframe(table, use_container_width=True)
    _ceiling_captions(exp1["config_id"].unique())


def _safety(agg: pd.DataFrame) -> None:
    st.caption("Deterministic, rule-based safety (target = 0). Allergen detection uses the FoodOn "
               "hierarchy, not string matching. Safety is a hard feasibility gate — never "
               "slot-normalized, never a Pareto axis.")

    def allergen(exp, cfg):
        return rd.cell(agg, exp, cfg, "allergen_violation", "aggregate")

    # Before → after arc.
    st.markdown("**Allergen violation**")
    c1, c2, c3 = st.columns(3)
    _stat_card(c1, "Unconstrained LLM-only", _fmt(allergen("EXP-10k-tight", "llm_only"), "pct"),
               "invents recipes with no retrieval — breaks a hard allergen constraint ~9× in 10", "bad")
    naive = allergen("EXP-10k-tight", "naive_rag")
    _stat_card(c2, "Grounded baseline (naive-RAG)", _fmt(naive, "pct"),
               "grounding the answer collapses the violation rate below 1%", "warn")
    foodrag_max = agg[(agg["experiment_id"] == _EXP_SYSTEM)
                      & (agg["metric"] == "allergen_violation")
                      & (agg["stratum"] == "aggregate")]["mean"].max()
    _stat_card(c3, "FoodRAG (post-filter)", _fmt(foodrag_max, "pct"),
               "deterministic post-filter → zero across every FoodRAG config", "good")

    st.divider()
    st.markdown("**Per-config safety (aggregate stratum)**")
    for metric in ("allergen_violation", "hard_constraint_violation"):
        sub = agg[(agg["metric"] == metric) & (agg["stratum"] == "aggregate")]
        if sub.empty:
            continue
        _section_label(_mlabel(metric))
        sub = sub.sort_values("mean")
        chart = pd.DataFrame({_mlabel(metric): sub["mean"].values},
                             index=[_clabel(c) for c in sub["config_id"]])
        _hbar(chart)


# ═══ TIER 2 — ATTRIBUTION ═══════════════════════════════════════════════════
def _agentic(agg: pd.DataFrame, pw: pd.DataFrame | None) -> None:
    sysm = agg[agg["experiment_id"] == _EXP_SYSTEM]
    if sysm.empty:
        st.info("Full system evaluation results not available.")
        return
    st.caption("H1 = agentic (Recommend + Correct + Enhance + Explain); H2 = recommend-only. "
               "Compare on a fixed retriever (dense / Text→Pandas). The agentic pipeline helps "
               "most when retrieval is strong — on t2p it recovers dropped-gold recipes and lifts "
               "correctness; on weak (dense) retrieval it has little to correct.")
    metric = st.selectbox("Metric", sorted(sysm["metric"].unique()),
                          format_func=_mlabel, key="rep_ag_metric")
    desc = rd.metric_description(metric)
    if desc:
        st.caption(f"**{_mlabel(metric)}** — {desc}")

    # The full system matrix (EXP-7) was run on the tight stratum only, so the
    # "aggregate" rows are a verbatim copy of "tight" (identical means and n).
    # Only offer strata that actually carry distinct numbers, so the control
    # never looks like it's silently doing nothing.
    strata = _distinct_strata(sysm, metric)
    if len(strata) > 1:
        stratum = st.radio("Stratum", strata, horizontal=True, key="rep_ag_stratum")
    else:
        stratum = strata[0]
        st.caption("Tight stratum only (n=1,646) — the full system evaluation (FoodRAG "
                   "configurations vs the baseline ladder) was not stratified further, so there is "
                   "no separate aggregate view.")
    sub = sysm[(sysm["metric"] == metric) & (sysm["stratum"] == stratum)].sort_values("config_id")
    if sub.empty:
        st.info("No rows for this selection.")
        return
    chart = pd.DataFrame({_mlabel(metric): sub["mean"].values},
                         index=[_clabel(c) for c in sub["config_id"]])
    _hbar(chart)
    _maybe_nutri_grades(metric, [_clabel(c) for c in sub["config_id"]], sub["mean"].values)
    show = sub[["config_id", "mean", "ci_low", "ci_high", "per_question_variance"]].copy()
    show["config_id"] = show["config_id"].map(_clabel)
    st.dataframe(
        show.round(3).rename(columns={
            "config_id": "config", "ci_low": "95% CI low", "ci_high": "95% CI high",
            "per_question_variance": "per-question variance"}),
        use_container_width=True, hide_index=True)
    st.caption("Confidence intervals are 95% bootstrap intervals (10,000 resamples over "
               "per-question means).")

    # Error decomposition callout (retriever-miss vs pipeline-dropped-gold).
    et = rd.error_taxonomy()
    if et is not None and {"foodrag_h2_t2p", "foodrag_h1_t2p"}.issubset(et.index):
        h2 = et.loc["foodrag_h2_t2p", "empty_plan"]
        h1 = et.loc["foodrag_h1_t2p", "empty_plan"]
        st.caption(f"Error decomposition (t2p): empty-plan / dropped-gold rate {h2:.1%} (H2) → "
                   f"{h1:.1%} (H1) — the Correct stage recovers plans H2 would have left empty.")

    if pw is not None and not pw.empty:
        sig = pw[(pw["metric"] == metric) & (pw["stratum"] == stratum)]
        for _, r in sig.iterrows():
            mark = "✅ significant" if r["significant"] else "ns"
            st.caption(f"{_clabel(r['config_a'])} vs {_clabel(r['config_b'])}: "
                       f"Δ={r['mean_diff']:+.3f}, {r['test']}, p={r['p_value']} "
                       f"(corrected {r['p_value_corrected']}) — {mark}")


def _error_taxonomy(agg: pd.DataFrame) -> None:
    et = rd.error_taxonomy()
    if et is None:
        st.info("Per-question parquets not available — error taxonomy cannot be computed.")
        return
    st.caption("Per-config failure rates (fraction of per-question records; higher = worse). "
               "The **dense↔t2p** split is the dominant error axis: dense configs miss retrieval "
               "~97% of the time, so everything downstream fails.")
    buckets = ["retrieval_miss", "empty_plan", "hard_violation", "low_correctness",
               "unfaithful", "structural_invalid"]
    labels = {
        "retrieval_miss": "Retrieval miss", "empty_plan": "Empty plan",
        "hard_violation": "Hard-constraint viol.", "low_correctness": "Low correctness (<0.5)",
        "unfaithful": "Unfaithful (<0.5)", "structural_invalid": "Structural-invalid *",
    }
    definitions = [
        ("Retrieval miss", "no constraint-valid gold recipe reached the top-10 (hit@10 = 0), so "
                           "the pipeline has nothing correct to build on."),
        ("Empty plan", "the planner filled no meal slot — an honest no-recommendation rather than "
                       "a hallucinated plan."),
        ("Hard-constraint viol.", "the plan broke at least one hard dietary/health constraint "
                                  "(e.g. an allergen)."),
        ("Low correctness (<0.5)", "the recommendation scored below 0.5 on judged correctness "
                                   "(Judge A · Gemma-3-27B)."),
        ("Unfaithful (<0.5)", "the answer scored below 0.5 on judged faithfulness — it drifted "
                              "from the retrieved evidence (Judge A · Gemma-3-27B)."),
        ("Structural-invalid *", "the full-day structural-validity check returned 0 — an artifact "
                                 "of the full-day denominator (see note below), not a real failure."),
    ]
    with st.expander("What each error type means"):
        for name, meaning in definitions:
            html(f"<div style=\"font-family:{F};font-size:12px;color:{C['text']};margin:2px 0\">"
                 f"<b>{name}</b> — <span style=\"color:{C['textMuted']}\">{meaning}</span></div>")
    disp = et[buckets].rename(columns=labels).copy()
    disp.index = [_clabel(i) for i in disp.index]
    st.dataframe(
        disp.style.format("{:.1%}").background_gradient(cmap="Reds", vmin=0, vmax=1),
        use_container_width=True,
    )
    st.caption("\\* **Structural-invalid is an artifact** for the deployed (partial-day) configs, "
               "not a real failure: it uses a full-day (4-slot) denominator, but the time-aware "
               "planner returns partial-day plans (mean 2.32 of 4 slots). The dedicated "
               "**…full-day** rows above force all four slots and show structural-invalid collapsing "
               "to ~0 — confirming it was a denominator artifact. Read the slot-normalized values in "
               "the **Plan quality** panel.")


def _retrieval_vs_pipeline(agg: pd.DataFrame) -> None:
    st.caption("Dual-use decomposition: retrieval-only (Use-1) vs the full meal-plan pipeline "
               "(Use-2) hit@10, per retrieval approach. Shows what the end-to-end system adds on "
               "top of raw retrieval.")

    def hit(exp, cfg):
        return rd.cell(agg, exp, cfg, "hit_at_10", "aggregate")

    rows = [
        {"Approach": "Dense (embeddings)",
         "Use-1 retrieval": hit(_EXP_RETRIEVER, "ret_dense"),
         "Use-2 H2 (rec-only)": hit(_EXP_SYSTEM, "foodrag_h2"),
         "Use-2 H1 (agentic)": hit(_EXP_SYSTEM, "foodrag_h1_dense")},
        {"Approach": "Text→Pandas",
         "Use-1 retrieval": hit(_EXP_RETRIEVER, "ret_text2pandas@gpt4o-mini"),
         "Use-2 H2 (rec-only)": hit(_EXP_SYSTEM, "foodrag_h2_t2p"),
         "Use-2 H1 (agentic)": hit(_EXP_SYSTEM, "foodrag_h1_t2p")},
    ]
    df = pd.DataFrame(rows).set_index("Approach")
    st.dataframe(df.round(3), use_container_width=True)
    st.caption("External baselines (Use-1 vs Use-2 where available):")
    brows = []
    for exp, cfg in [("EXP-10k-tight", "llm_only"), ("EXP-10k-tight", "naive_rag"),
                     ("EXP-10k-tog", "tog"), ("EXP-10k", "popularity_raw"),
                     ("EXP-10k", "popularity_health_aware")]:
        brows.append({"Baseline": _clabel(cfg), "hit@10": hit(exp, cfg),
                      "allergen": rd.cell(agg, exp, cfg, "allergen_violation", "aggregate")})
    st.dataframe(pd.DataFrame(brows).round(3), use_container_width=True, hide_index=True)
    _ceiling_captions(["popularity_health_aware", "tog"])


# ═══ TIER 3 — DEPTH ═════════════════════════════════════════════════════════
def _pareto_df(result, x_axis: str, y_axis: str) -> pd.DataFrame | None:
    recs = result.as_records()
    if not recs:
        return None
    rows = []
    for r in recs:
        rows.append({"config_id": r["config_id"], "config": _clabel(r["config_id"]),
                     x_axis: r.get(x_axis), y_axis: r.get(y_axis), "frontier": bool(r["frontier"])})
    return pd.DataFrame(rows)


def _domination_captions(result) -> None:
    lines = []
    for cfg in result.frontier:
        beaten = result.dominates.get(cfg, [])
        if beaten:
            lines.append(f"**{_clabel(cfg)}** dominates " + ", ".join(_clabel(b) for b in beaten))
    if lines:
        st.caption("Domination: " + "; ".join(lines) + ".")
    if result.excluded:
        st.caption("Excluded (missing metric): "
                   + ", ".join(f"{_clabel(c)} ({why})" for c, why in result.excluded.items()) + ".")


def _pareto(agg: pd.DataFrame) -> None:
    from foodrag.analysis import pareto as pmod

    st.caption("Pareto-optimal (non-dominated) configs are highlighted and connected by the "
               "frontier; dominated configs are greyed. An **analysis lens** over feasible outputs "
               "— safety is **never** a Pareto axis (hard constraints gate feasibility).")
    avail = set(agg["experiment_id"])

    st.markdown("**Health vs Relevance**")
    exps_hr = [e for e in (_EXP_RETRIEVER, "EXP-10k", _EXP_SYSTEM) if e in avail]
    if not exps_hr:
        st.info("No experiment with health/relevance metrics available.")
    else:
        c1, c2 = st.columns(2)
        exp_hr = c1.selectbox("Experiment", exps_hr, format_func=lambda e: rd.exp_meta(e)["title"],
                              key="rep_par_hr_exp")
        health = c2.radio("Health axis", ["avg_nutri_at_10", "avg_fsa_at_10"], horizontal=True,
                          format_func=_mlabel, key="rep_par_hr_health")
        res = pmod.health_vs_relevance(agg, experiment_id=exp_hr, health_metric=health)
        df = _pareto_df(res, "relevance", "health")
        if df is None:
            st.info("Health/relevance metrics not available for this experiment.")
        else:
            rel_lbl = _mlabel(res.axis_labels.get("relevance", "relevance"))
            render_pareto(df, "relevance", "health", f"{rel_lbl} (relevance)",
                          f"{_mlabel(health)} (health)", x_dir="max", y_dir="max", flag_col="frontier",
                          ideal="Ideal: top-right — relevant *and* healthy. Popularity/rating-sorted "
                                "baselines sit high-relevance / low-health (popularity bias).")
            _domination_captions(res)
            _ceiling_captions(df["config_id"].unique())

    st.divider()
    st.markdown("**Quality vs Cost**")
    exps_qc = [e for e in (_EXP_SYSTEM,) if e in avail]
    if not exps_qc:
        st.info("No experiment with quality/cost metrics available.")
    else:
        c1, c2 = st.columns(2)
        exp_qc = c1.selectbox("Experiment", exps_qc, format_func=lambda e: rd.exp_meta(e)["title"],
                              key="rep_par_qc_exp")
        cost = c2.radio("Cost axis", ["llm_usage", "latency_s"], horizontal=True,
                        format_func=_mlabel, key="rep_par_qc_cost")
        res = pmod.quality_vs_cost(agg, experiment_id=exp_qc, cost_metric=cost)
        df = _pareto_df(res, "cost", "quality")
        if df is None:
            st.info("Quality/cost metrics not available for this experiment.")
        else:
            render_pareto(df, "cost", "quality", f"{_mlabel(cost)} (cost ↓)",
                          "Correctness (quality)", x_dir="min", y_dir="max", flag_col="frontier",
                          ideal="Ideal: top-left — high quality at low cost. Frontier is the genuine "
                                "trade-off {H2-t2p (cheap), H1-t2p (best quality)}.")
            _domination_captions(res)

    st.divider()
    st.markdown("**Safety vs Coverage**")
    exps_sc = [e for e in ("EXP-10k", "EXP-10k-tog", _EXP_SYSTEM) if e in avail]
    if not exps_sc:
        st.info("No experiment with a coverage proxy available.")
    else:
        exp_sc = st.selectbox("Experiment", exps_sc, format_func=lambda e: rd.exp_meta(e)["title"],
                              key="rep_par_sc_exp")
        res = pmod.safety_vs_coverage(agg, experiment_id=exp_sc)
        df = _pareto_df(res, "strictness", "coverage")
        if df is None or df.empty:
            st.info("Coverage proxy not available across strata for this experiment.")
        else:
            df = df.copy()
            df["config"] = df["config_id"]  # points are strata, not configs
            cov_lbl = _mlabel(res.axis_labels.get("coverage", "coverage"))
            render_pareto(df, "strictness", "coverage", "Strictness (answer-set stratum, proxy)",
                          f"Coverage — {cov_lbl}", x_dir="max", y_dir="max", flag_col="frontier",
                          ideal="Relaxation level (strictness) vs fraction of personas with a full "
                                "feasible plan. Strictness uses answer-set-size strata as a "
                                "documented PROXY (tight → high strictness).")
            st.caption("⚠ Documented proxy: answer-set-size strata stand in for a per-config "
                       "relaxation axis. Safety is never an axis — hard constraints gate "
                       "feasibility upstream.")
            _domination_captions(res)


def _explainability(agg: pd.DataFrame) -> None:
    st.caption("Two explanation channels. **Intrinsic KG paths** are read from HUMMUS (always "
               "valid); the **NL explanation layer** is LLM-generated and can drift. The intrinsic "
               "layer is the more reliable channel.")
    cfg = st.selectbox("Config", ["foodrag_h1_t2p", "foodrag_h1_dense"], format_func=_clabel,
                       key="rep_expl_cfg")
    c1, c2, c3, c4 = st.columns(4)
    _stat_card(c1, "KG path validity", _fmt(rd.cell(agg, _EXP_SYSTEM, cfg, "path_validity"), "pct"),
               "intrinsic paths read from the graph", "good")
    _stat_card(c2, "KG path coverage", _fmt(rd.cell(agg, _EXP_SYSTEM, cfg, "path_coverage"), "pct"),
               "share of plan items with a KG path", "neutral")
    _stat_card(c3, "NL claim grounding", _fmt(rd.cell(agg, _EXP_SYSTEM, cfg, "claim_grounding_rate"), "pct"),
               "NL claims supported by evidence", "neutral")
    _stat_card(c4, "Unsupported claims", _fmt(rd.cell(agg, _EXP_SYSTEM, cfg, "unsupported_claim_rate"), "pct"),
               "NL claims with no support ↓", "warn")
    st.caption("The stronger retriever (t2p) yields better-grounded NL explanations (0.71 vs 0.55 "
               "on dense); intrinsic KG paths stay 100% valid regardless.")


def _judge_reliability(agg: pd.DataFrame) -> None:
    ja = rd.judge_agreement()
    if ja is None:
        st.info("Per-question judge scores not available — cross-judge agreement cannot be computed.")
        return
    st.caption("Cross-judge agreement between **Judge A (Gemma-3-27B)** and **Judge B "
               "(DeepSeek-V3)**, per config × metric (Pearson over per-question scores). "
               "Judge–human calibration is **future work**; only cross-judge agreement is shown.")
    disp = ja.copy()
    disp["config"] = disp["config_id"].map(_clabel)
    disp["metric"] = disp["metric"].map(lambda m: _mlabel(m))
    disp = disp[["config", "metric", "pearson", "mean_a", "mean_b", "mean_abs_diff", "n"]]
    st.dataframe(disp.round(3).rename(columns={
        "pearson": "Pearson r", "mean_a": "mean (Gemma)", "mean_b": "mean (DeepSeek)",
        "mean_abs_diff": "mean |Δ|"}), use_container_width=True, hide_index=True)
    st.caption("Agreement is only moderate (correctness r ≈ 0.35–0.76) — the headline uses Judge A "
               "(Gemma, cross-lineage) and reports Judge B alongside rather than averaging them.")


def _plan_quality(agg: pd.DataFrame) -> None:
    sysm = agg[agg["experiment_id"] == _EXP_SYSTEM]
    configs = list(sysm["config_id"].unique())
    fullday = [c for c in configs if c.endswith("_fullday")]
    partial = [c for c in configs if not c.endswith("_fullday")]

    st.info("**Two honest readings of plan quality.** The *deployed* planner is time-aware and "
            "returns partial-day plans (mean 2.32 of 4 slots), so full-day-denominated metrics "
            "understate those configs — read them on the **slot-normalized** denominator. Separately, "
            "dedicated **full-day runs** (…full-day) force all four slots; their raw full-day metrics "
            "are the honest full-day reading, and they **confirm the artifact** — structural validity "
            "rises from ~0.17 (partial-day, full denominator) to ~1.0 on the real full-day run. "
            "See the **Time-of-day** panel for why partial-day plans are a feature.")

    # Artifact-resolution contrast (H1-t2p): partial-day full-denominator → real full-day run.
    sv_p = rd.cell(agg, _EXP_SYSTEM, "foodrag_h1_t2p", "structural_validity_full")
    sv_f = rd.cell(agg, _EXP_SYSTEM, "foodrag_h1_t2p_fullday", "structural_validity_full")
    cm_p = rd.cell(agg, _EXP_SYSTEM, "foodrag_h1_t2p", "meal_slot_completeness_full")
    cm_f = rd.cell(agg, _EXP_SYSTEM, "foodrag_h1_t2p_fullday", "meal_slot_completeness_full")
    if not _isnan(sv_f) or not _isnan(cm_f):
        cA, cB = st.columns(2)
        _stat_card(cA, "Structural validity (H1-t2p)", f"{_fmt(sv_p)} → {_fmt(sv_f)}",
                   "partial-day (full denom.) → dedicated full-day run", "good")
        _stat_card(cB, "Meal-slot completeness (H1-t2p)", f"{_fmt(cm_p)} → {_fmt(cm_f)}",
                   "partial-day → dedicated full-day run", "good")

    if partial:
        _section_label("Deployed (partial-day) configs — slot-normalized reading")
        norm = ["meal_slot_completeness_norm", "calorie_adherence_norm",
                "intra_plan_diversity_norm", "structural_validity_norm"]
        norm = [m for m in norm if m in set(sysm["metric"])]
        psub = sysm[(sysm["config_id"].isin(partial)) & (sysm["metric"].isin(norm))
                    & (sysm["stratum"] == "aggregate")]
        if not psub.empty:
            st.dataframe(_pivot(psub, "config_id", "metric").round(3), use_container_width=True)

    if fullday:
        _section_label("Dedicated full-day runs — real full-day metrics")
        full = ["meal_slot_completeness_full", "calorie_adherence_full",
                "structural_validity_full", "guideline_adherence_full"]
        full = [m for m in full if m in set(sysm["metric"])]
        fsub = sysm[(sysm["config_id"].isin(fullday)) & (sysm["metric"].isin(full))
                    & (sysm["stratum"] == "aggregate")]
        if not fsub.empty:
            st.dataframe(_pivot(fsub, "config_id", "metric").round(3), use_container_width=True)
            st.caption("Genuine full-day plans (all four slots requested), so the full-day "
                       "denominator is appropriate here — these are real values, not artifacts.")

    if not partial and not fullday:
        st.info("Plan-quality metrics not available.")


def _guideline_adherence(agg: pd.DataFrame) -> None:
    st.info("Per-nutrient guideline adherence on the **slot-normalized** denominator for the "
            "deployed (partial-day) configs; the dedicated **…full-day** runs are a real full-day "
            "reading. Low energy/carb bands on partial-day configs reflect the partial-day confound, "
            "not a planner failure — see the **Time-of-day** and **Plan quality** panels.")
    sysm = agg[agg["experiment_id"] == _EXP_SYSTEM]
    cfg = st.selectbox("Config", sorted(sysm["config_id"].unique()), format_func=_clabel,
                       key="rep_gl_cfg")
    is_full = cfg.endswith("_fullday")
    # Per-nutrient bands are stored only on the slot-normalized denominator; for the
    # dedicated full-day runs the normalized and full denominators coincide (all four
    # slots requested), so these are the real full-day bands for those configs.
    nutrients = ["energy", "protein", "carb", "fat", "satfat", "fibre", "sodium", "sugar"]
    col_label = "adherence (full-day)" if is_full else "adherence (slot-norm)"
    rows = []
    for n in nutrients:
        v = rd.cell(agg, _EXP_SYSTEM, cfg, f"guideline_adherence_norm_{n}")
        if not _isnan(v):
            rows.append({"nutrient": n, col_label: v})
    if not rows:
        st.info("Per-nutrient guideline metrics not available for this config.")
        return
    df = pd.DataFrame(rows).set_index("nutrient")
    _hbar(df)
    overall = rd.cell(agg, _EXP_SYSTEM, cfg,
                      "guideline_adherence_full" if is_full else "guideline_adherence_norm")
    if is_full:
        st.caption(f"Overall (full-day): {_fmt(overall)}. Dedicated full-day run — all four slots "
                   "requested, so these bands are a real full-day reading.")
    else:
        st.caption(f"Overall (slot-norm): {_fmt(overall)}. Bands are per-nutrient; low energy/carb "
                   "reflect the partial-day confound, not a planner failure.")


def _time_of_day() -> None:
    tod = rd.time_of_day()
    if tod is None:
        st.info("mealplan_time_of_day.csv not available.")
        return
    st.caption("Deployed behaviour: the meal-plan tool selects meal types by time of day (it plans "
               "'the remaining day'), so plans are **partial-day**. This is the quantitative reason "
               "the full-day plan-level metrics read low — a feature, not a bug (no breakfast at 9 pm).")
    meals = tod.set_index("hour")[["breakfast", "lunch", "dinner", "snack"]]
    st.markdown("**Meal-type selection probability by hour**")
    st.line_chart(meals)
    st.markdown("**Mean meals per plan by hour**")
    st.bar_chart(tod.set_index("hour")[["mean_meals"]])
    st.caption("Mornings → breakfast-dominant; evenings → dinner + snack. Mean 2.32 meals/plan "
               "overall (not 4), which is why full-day completeness reads ~0.48.")


def _case_browser() -> None:
    cases = rd.cases()
    if not cases:
        st.info("Per-question parquet not available — case browser is empty.")
        return
    st.caption("Illustrative per-question cases from the H1-t2p run.")
    for c in cases:
        with st.container(border=True):
            html(f"<div style=\"font-family:{F};font-weight:700;color:{C['primary']}\">{c['kind']}</div>"
                 f"<div style=\"font-family:{F};font-size:12px;color:{C['textMuted']};margin-bottom:6px\">"
                 f"{c['why']}</div>")
            if c["plan"]:
                st.markdown("**Plan:** " + ", ".join(f"{k} → `{str(v).split('/')[-1]}`"
                                                      for k, v in c["plan"].items()))
            else:
                st.markdown("**Plan:** _(empty — honest no-recommendation)_")
            if c["explanation"]:
                st.caption("Explanation: " + c["explanation"])
            cols = st.columns(4)
            for col, (k, lbl) in zip(cols, [("hit_at_10", "hit@10"),
                                            ("correctness_judgeA", "corr. (Gemma)"),
                                            ("correctness_judgeB", "corr. (DeepSeek)"),
                                            ("allergen_violation", "allergen")]):
                col.metric(lbl, _fmt(c.get(k)))


def _volatility(agg: pd.DataFrame) -> None:
    vol = rd.volatility(agg, experiment=_EXP_SYSTEM)
    if vol is None:
        st.info("No per-question variance recorded for the full system evaluation.")
        return
    st.caption("Per-question variance by config (full system evaluation). Higher variance = noisier "
               "per-question behaviour for that metric.")
    metric = st.selectbox("Metric", sorted(vol["metric"].unique()), format_func=_mlabel,
                          key="rep_vol_metric")
    sub = vol[vol["metric"] == metric].sort_values("per_question_variance", ascending=True)
    chart = pd.DataFrame({"variance": sub["per_question_variance"].values},
                         index=[_clabel(c) for c in sub["config_id"]])
    _hbar(chart)


def _stratified(agg: pd.DataFrame) -> None:
    st.caption("Demoted: the headline results are all on the tight stratum (n=1,646) by design, so "
               "cross-stratum views are thin. Kept for completeness.")
    exp = _exp_select(agg, "rep_st_exp")
    sub0 = agg[agg["experiment_id"] == exp]
    metric = st.selectbox("Metric", sorted(sub0["metric"].unique()), format_func=_mlabel,
                          key="rep_st_metric")
    sub = sub0[(sub0["metric"] == metric) & (sub0["stratum"] != "aggregate")]
    if sub.empty:
        st.info("No stratified rows for this experiment (tight-only).")
        return
    chart = _pivot(sub, "config_id", "stratum")
    chart = chart[[s for s in _STRATA if s in chart.columns]]
    # Annotate each stratum (legend) with the number of items (personas) it covers
    # for this metric — the stratum's answer-set size.
    nmap = sub.groupby("stratum")["n_questions"].max()
    chart = chart.rename(columns={s: f"{s} (n={int(nmap[s]):,})"
                                  for s in chart.columns if s in nmap.index and pd.notna(nmap[s])})
    # Grouped (not stacked) horizontal bars: one bar per stratum, side by side,
    # so per-stratum values are read directly and config labels are never cut off.
    _hbar(chart, grouped=True)
    st.caption("Aggregates are dominated by broad personas — read alongside the tight stratum.")


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

    html(f"<p style=\"font-family:{F};font-size:13px;color:{C['textMuted']};margin:4px 0 10px;"
         f"line-height:1.6\">Offline evaluation results. Pre-computed and read-only.</p>")

    if not eval_results.available():
        st.warning("No evaluation results found. Place `aggregate.csv` (+ optional `pairwise.csv`) "
                   "in `eval_results/` (or set `FOODRAG_EVAL_RESULTS`).")
        return

    if eval_results.is_sample():
        st.warning("⚠ Showing **bundled sample results (synthetic placeholders)**. Replace "
                   "`eval_results/*.csv` with real evaluation outputs to see real numbers.")

    agg = rd.load_aggregate()
    pw = rd.load_pairwise()

    tier1, tier2, tier3 = st.tabs(["① Headline", "② Attribution", "③ Depth"])
    with tier1:
        t = st.tabs(["Overview", "Retriever comparison", "Safety"])
        with t[0]:
            _overview(agg)
        with t[1]:
            _retriever_comparison(agg)
        with t[2]:
            _safety(agg)
    with tier2:
        t = st.tabs(["Agentic comparison", "Error taxonomy", "Retrieval vs full pipeline"])
        with t[0]:
            _agentic(agg, pw)
        with t[1]:
            _error_taxonomy(agg)
        with t[2]:
            _retrieval_vs_pipeline(agg)
    with tier3:
        t = st.tabs(["Pareto", "Explainability", "Judge reliability", "Plan quality",
                     "Guideline adherence", "Time-of-day", "Case browser", "Volatility", "Stratified"])
        with t[0]:
            _pareto(agg)
        with t[1]:
            _explainability(agg)
        with t[2]:
            _judge_reliability(agg)
        with t[3]:
            _plan_quality(agg)
        with t[4]:
            _guideline_adherence(agg)
        with t[5]:
            _time_of_day()
        with t[6]:
            _case_browser()
        with t[7]:
            _volatility(agg)
        with t[8]:
            _stratified(agg)
