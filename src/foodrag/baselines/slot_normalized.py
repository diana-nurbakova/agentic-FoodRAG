"""Slot-normalized plan-level metrics (spec: FoodRAG_Slot_Normalized_Metrics.md).

The deployed meal-planner is time-driven: it plans only the *remaining* meals of the
day, producing partial-day plans (mean ~2.3 of 4 slots). The full-day plan metrics in
:mod:`meal_plan_metrics` / :mod:`guideline_adherence` score such a plan against a fixed
4-slot day, so a well-formed 2-meal plan looks like a 2-slot failure. This module scores
each plan against the meals it was *asked* to produce (``requested_meal_types``), and
scales full-day nutrient targets by the day-energy share those meals represent.

Everything here is deterministic and recomputable OFFLINE from an existing §13.1 record
plus ``recipes_df`` (for nutrients / meal_types) and ``guideline_reference_tables.json``
(bands) — NO pipeline re-run, no LLM. The persona's guideline class is read from the
record's stored ``guideline_class`` (the shipped QA set is always ``general_adult``), so
the QA pickle is not needed.

Design (spec §3):
- **Slot-set normalization** (completeness / structural): denominator is ``|Q|`` (requested
  slots), not 4.
- **Energy-share normalization** (guideline / calorie): full-day bands/ceilings scaled by
  ``rho(Q) = sum(delta_m for m in Q)`` — the requested slots' nominal share of daily energy.

Two ``Q`` sources are supported:
- ``"persisted"``  — ``record["requested_meal_types"]`` is present (only ``foodrag_h1_t2p``).
- ``"approx_from_plan"`` — absent (h2 / h2_t2p / h1_dense ran pre-persist) → ``Q`` = the
  plan's *filled* slots. Completeness/structural are then ~trivial (F == Q by construction);
  the energy-share-scaled guideline/calorie metrics stay meaningful (scaled by the filled
  slots' day-share). Callers should surface ``requested_slots_source`` to flag this.

IMPORTANT asymmetry (spec §3.4): scaling applies to the guideline *adherence-quality* score
only. The hard *safety* ceiling (feeding ``hard_constraint_violation``) is NEVER scaled and
is not touched here — a plan over the real sodium UL is unsafe regardless of slot count.
"""
from __future__ import annotations

from typing import Mapping, Optional

from foodrag.baselines import guideline_adherence as _ga
from foodrag.baselines import meal_plan_metrics as _mpm

Plan = Mapping[str, Optional[dict]]

# §4.5 daily calorie-distribution weights (Protocol §4.5 / meal_plan_metrics.CALORIE_TARGETS).
DELTA = dict(_mpm.CALORIE_TARGETS)  # breakfast .22 / lunch .32 / dinner .32 / snack .14

# spec §3.3 nutrient split. Band-target: "reach [l,h]" → scale BOTH edges by rho.
# Ceiling: "stay under kappa" → scale the (adherence-scoring) ceiling by rho.
BAND_TARGET_NUTRIENTS = {"energy_kcal", "carbohydrate_g", "total_fat_g", "protein_g", "fibre_g"}
CEILING_NUTRIENTS = {"saturated_fat_g", "sugars_g", "sodium_g"}

NAN = float("nan")


# ── Q (requested slots) + rho (requested-day energy share) ────────────────────

def requested_slots(record: dict, plan_rows: Plan) -> tuple[list[str], str]:
    """``(Q, source)``. Prefer persisted ``requested_meal_types``; else the plan's filled slots.

    ``Q`` is lower-cased and restricted to the four canonical meal slots. ``source`` is
    ``"persisted"`` or ``"approx_from_plan"`` (documented approximation for pre-persist runs).
    """
    rmt = record.get("requested_meal_types")
    if rmt:
        Q = [str(m).strip().lower() for m in rmt]
        Q = [m for m in Q if m in DELTA]
        if Q:
            return Q, "persisted"
    # fallback: the meals the plan actually produced
    Q = [str(s).strip().lower() for s, r in plan_rows.items() if r and str(s).strip().lower() in DELTA]
    return Q, "approx_from_plan"


def rho(Q) -> float:
    """Requested-day energy share ``sum(delta_m for m in Q)`` ∈ (0, 1] (0 for empty Q)."""
    return sum(DELTA[m] for m in Q if m in DELTA)


# ── slot-set metrics (spec §3.1, §3.2) ────────────────────────────────────────

def filled_requested(plan_rows: Plan, Q) -> list[str]:
    """Requested slots that are filled with a resolved recipe: ``F = {m in Q : P_m != ∅}``."""
    return [m for m in Q if _mpm._filled(plan_rows.get(m))]


def meal_slot_completeness_norm(plan_rows: Plan, Q) -> float:
    """``|F| / |Q|`` — fraction of *requested* slots filled. NaN when nothing was requested."""
    if not Q:
        return NAN
    return len(filled_requested(plan_rows, Q)) / len(Q)


def structural_validity_norm(plan_rows: Plan, Q, *, parseable: bool = True) -> float:
    """1.0 iff parseable AND ``F == Q`` AND every filled recipe admits its slot.

    "Admits its slot" (spec §3.2, ``m ∈ μ(P_m)``): the slot label appears in the recipe's
    ``meal_types``. No full-4-slot requirement. NaN when ``Q`` is empty (undefined).
    """
    if not Q:
        return NAN
    F = filled_requested(plan_rows, Q)
    if not parseable or set(F) != set(Q):
        return 0.0
    for slot in F:
        if slot not in _mpm._meal_types(plan_rows.get(slot)):
            return 0.0
    return 1.0


# ── energy-share-scaled guideline adherence (spec §3.3, §3.4) ─────────────────

def guideline_adherence_norm(totals: dict, bands: dict, rho_q: float, ref: dict) -> tuple[dict, float]:
    """Per-nutrient 0/1 (short keys) + overall mean, with bands/ceilings scaled by ``rho_q``.

    Band-target nutrients (energy/carb/fat/protein/fibre): in-band iff
    ``rho*l <= total <= rho*h``. Ceiling nutrients (satfat/sugar/sodium): adherent iff
    ``total <= rho*kappa`` where ``kappa`` is the band's upper edge (the *adherence-scoring*
    ceiling — the safety hard-ceiling is untouched, spec §3.4). Returns ``({}, NaN)`` when the
    plan has no nutrient totals (no resolved recipe).
    """
    if not totals:
        return {}, NAN
    per: dict = {}
    for nutrient in ref["metric_computation"]["tracked_nutrients"]:
        short = _ga.SHORT_KEYS[nutrient]
        band = bands.get(nutrient)
        total = totals.get(nutrient)
        if band is None or total is None:
            per[short] = 0
            continue
        if nutrient in CEILING_NUTRIENTS:
            kappa = band.get("high")
            per[short] = 1 if (kappa is None or total <= rho_q * kappa) else 0
        else:  # band-target: scale both edges
            low = band["low"] if band.get("band_type") != "upper_only" else 0
            low = low if low is not None else 0
            high = band.get("high")
            lo = rho_q * low
            if high is None:
                per[short] = 1 if total >= lo else 0
            else:
                per[short] = 1 if lo <= total <= rho_q * high else 0
    overall = sum(per.values()) / len(per) if per else NAN
    return per, overall


# ── renormalized calorie-distribution adherence (spec §3.5) ───────────────────

def calorie_adherence_norm(plan_rows: Plan, Q, rho_q: float) -> float:
    """``1 - mean_{m in Q} |eshare_m(P) - delta_m/rho(Q)|``.

    The target split is renormalized over the requested slots so they sum to 1 (e.g.
    lunch/dinner/snack .32/.32/.14 → .41/.41/.18). ``eshare_m`` is meal m's share of the
    plan's energy (0 for an unfilled requested slot). NaN when ``Q`` is empty or the plan has
    no usable calorie value.
    """
    if not Q or rho_q <= 0:
        return NAN
    cals = {}
    for slot, recipe in plan_rows.items():
        c = _mpm._total_calories(recipe)
        if c is not None and c > 0:
            cals[str(slot).strip().lower()] = c
    total = sum(cals.values())
    if total <= 0:
        return NAN
    devs = [abs(cals.get(m, 0.0) / total - DELTA[m] / rho_q) for m in Q]
    return max(0.0, 1.0 - sum(devs) / len(Q))


# ── helpers: plan rows + class bands ──────────────────────────────────────────

def plan_rows_from_record(record: dict, indexed) -> dict:
    """Resolve the record's stored ``plan`` (``{slot: recipe_url|None}``) to recipe-row dicts.

    ``indexed`` is ``recipes_df.set_index("recipe_url", drop=False)``. Unresolvable / empty
    slots map to ``None``.
    """
    plan = record.get("plan") or {}
    rows: dict = {}
    for slot, url in plan.items():
        key = str(slot)
        if url and url in indexed.index:
            row = indexed.loc[url]
            if getattr(row, "ndim", 1) == 2:
                row = row.iloc[0]
            rows[key] = row.to_dict()
        else:
            rows[key] = None
    return rows


def bands_for_class_label(label: Optional[str], ref: dict) -> dict:
    """Resolved per-nutrient bands for a stored ``guideline_class`` label (default general_adult).

    Handles the ``composite(a+b)`` label form (comorbidity → stricter band) by re-resolving
    across the component classes, reusing :func:`guideline_adherence._resolve_bands`.
    """
    label = label or _ga.DEFAULT_CLASS
    if label.startswith("composite(") and label.endswith(")"):
        classes = [c for c in label[len("composite("):-1].split("+") if c]
    else:
        classes = [label]
    valid = ref["reference_classes"]
    classes = [c for c in classes if c in valid] or [_ga.DEFAULT_CLASS]
    return _ga._resolve_bands(classes, ref)


def _class_label(record: dict) -> str:
    return (record.get("guideline_class")
            or record.get("metrics", {}).get("guideline_class")
            or _ga.DEFAULT_CLASS)


# ── top-level: compute all normalized metrics for one record ──────────────────

def compute_record_norm(record: dict, indexed, ref: dict) -> dict:
    """Compute the slot-normalized block for one §13.1 record (pure — no mutation).

    Returns a dict with the ``*_norm`` scalars, the per-nutrient norm vector, ``rho`` and
    ``requested_slots_source``. The caller merges these into the record (keeping the existing
    full-day values as ``*_full``).
    """
    plan_rows = plan_rows_from_record(record, indexed)
    Q, source = requested_slots(record, plan_rows)
    rho_q = rho(Q)

    totals = _ga.plan_daily_totals(plan_rows)
    bands = bands_for_class_label(_class_label(record), ref)
    per_nut, guide_overall = guideline_adherence_norm(totals, bands, rho_q, ref) if Q else ({}, NAN)

    return {
        "meal_slot_completeness_norm": meal_slot_completeness_norm(plan_rows, Q),
        "structural_validity_norm": structural_validity_norm(plan_rows, Q),
        "guideline_adherence_norm": guide_overall,
        "guideline_adherence_norm_per_nutrient": per_nut,
        "calorie_adherence_norm": calorie_adherence_norm(plan_rows, Q, rho_q),
        # spec §3.6: null (not 0) when < 2 filled slots — already the guard in this fn.
        "intra_plan_diversity_norm": _mpm.intra_plan_diversity(plan_rows),
        "requested_day_energy_share": rho_q if Q else NAN,
        "requested_slots_source": source,
        "n_requested_slots": len(Q),
    }
