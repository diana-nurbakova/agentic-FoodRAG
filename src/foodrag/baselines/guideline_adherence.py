"""Guideline Adherence — persona-class-relative nutrient-band scoring.

Scores an assembled meal plan's aggregate *daily* nutrient totals against reference
bands from ``config/guideline_reference_tables.json`` (EFSA DRV / WHO, class-based —
see that file's documented modeling assumptions and limitations). Each persona maps
to ONE reference class (default ``general_adult``; condition-specific overrides and a
comorbidity conflict-resolution rule are implemented for future personas — the shipped
QA set carries no condition field, so it always resolves to ``general_adult``).

For each of the eight tracked nutrients the plan total is scored 1 if it falls inside
the class band ``[low, high]`` (``upper_only`` bands treat ``low`` as 0), else 0.
``guideline_adherence_overall`` is the mean of those 0/1 flags;
``guideline_adherence_per_nutrient`` is a dict of the 0/1 flags under short per-nutrient record keys.

hard_ceiling interaction: a nutrient total exceeding its ``hard_ceiling`` (a UL
/ safety ceiling, e.g. sodium) is ALSO a safety violation — :func:`hard_ceiling_breach`
reports it so the caller can fold it into ``hard_constraint_violation`` for plan configs.

The reference tables are loaded lazily from the packaged config path but the loaded
dict is injectable (``reference_tables=...``) so tests need no file I/O.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

Plan = Mapping[str, Optional[dict]]

# Ref-nutrient key -> (recipes_df column, unit scale). sodium is mg in HUMMUS but the
# reference bands are in grams, so scale by 1/1000.
NUTRIENT_COLS = {
    "energy_kcal": ("calories [kcal]", 1.0),
    "protein_g": ("protein [g]", 1.0),
    "carbohydrate_g": ("totalCarbohydrate [g]", 1.0),
    "total_fat_g": ("totalFat [g]", 1.0),
    "saturated_fat_g": ("saturatedFat [g]", 1.0),
    "fibre_g": ("dietaryFiber [g]", 1.0),
    "sodium_g": ("sodium [mg]", 0.001),
    "sugars_g": ("sugars [g]", 1.0),
}

# Ref-nutrient key -> short per-nutrient record key.
SHORT_KEYS = {
    "energy_kcal": "energy",
    "protein_g": "protein",
    "carbohydrate_g": "carb",
    "total_fat_g": "fat",
    "saturated_fat_g": "satfat",
    "fibre_g": "fibre",
    "sodium_g": "sodium",
    "sugars_g": "sugar",
}

SERVING_COL = "servingSize [g]"
DEFAULT_CLASS = "general_adult"


def _default_ref_path() -> Path:
    # src/foodrag/baselines/guideline_adherence.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "config" / "guideline_reference_tables.json"


@lru_cache(maxsize=2)
def load_reference_tables(path: Optional[str] = None) -> dict:
    """Load the guideline reference-tables JSON (cached). ``path`` overrides the default."""
    p = Path(path) if path else _default_ref_path()
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _condition_text(persona: dict) -> str:
    """The persona's free-text condition/health signal, lowercased (empty if none).

    The shipped QA personas carry no such field (always → default class); this supports
    future personas that add ``condition`` or ``health``.
    """
    parts = []
    for key in ("condition", "health"):
        val = persona.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, (list, tuple)):
            parts.extend(str(v) for v in val)
    return " ".join(parts).lower()


def matched_classes(persona: dict, reference_tables: Optional[dict] = None) -> list[str]:
    """All reference classes a persona matches (first-match rules → usually one).

    Every rule whose ``if_condition_contains`` token appears in the persona's condition
    text matches; ``default`` applies when none match. Multiple matches (comorbidities)
    are resolved by :func:`_resolve_bands`.
    """
    ref = reference_tables or load_reference_tables()
    rules = ref["persona_to_class_mapping"]["rules"]
    text = _condition_text(persona)
    matched = []
    default_cls = DEFAULT_CLASS
    for rule in rules:
        if "default" in rule:
            default_cls = rule["default"]
            continue
        tokens = [t.lower() for t in rule.get("if_condition_contains", [])]
        if text and any(tok in text for tok in tokens):
            matched.append(rule["class"])
    return matched or [default_cls]


def _resolve_bands(classes: list[str], reference_tables: dict) -> dict:
    """Per-nutrient band dict, taking the STRICTER band across matched classes.

    Stricter = max of lower bounds, min of upper bounds (conflict_resolution). Preserves
    ``band_type`` and the strictest (lowest) ``hard_ceiling`` present.
    """
    ref_classes = reference_tables["reference_classes"]
    tracked = reference_tables["metric_computation"]["tracked_nutrients"]
    resolved: dict = {}
    for nutrient in tracked:
        low, high, band_type, ceiling = None, None, None, None
        for cls in classes:
            spec = ref_classes[cls]["nutrients"].get(nutrient)
            if spec is None:
                continue
            low = spec["low"] if low is None else max(low, spec["low"])
            high = spec["high"] if high is None else min(high, spec["high"])
            band_type = spec.get("band_type", band_type)
            if "hard_ceiling" in spec:
                ceiling = spec["hard_ceiling"] if ceiling is None else min(ceiling, spec["hard_ceiling"])
        resolved[nutrient] = {"low": low, "high": high, "band_type": band_type, "hard_ceiling": ceiling}
    return resolved


ENERGY_COL = "calories [kcal]"


def resolve_daily_energy(persona: dict, reference_tables: Optional[dict] = None) -> float:
    """The persona's daily energy budget in kcal — the anchor for per-slot caps.

    Resolution order: the persona's own stated cap when known, else the reference
    energy of their mapped guideline class (comorbidity → the strictest matched
    class), else the documented 2000 kcal default. The shipped personas carry no
    condition field, so they resolve to ``general_adult`` → 2000 unless they state
    a cap of their own.

    Anchoring the per-slot split here (rather than on an independent draw) is what
    keeps a meal-plan question self-consistent: the caps sum to the same number the
    question states as the daily total, and to the band this module scores against.
    """
    stated = (persona.get("total_nutrients") or {}).get(ENERGY_COL)
    try:
        stated = float(stated)
    except (TypeError, ValueError):
        stated = None
    if stated and stated > 0:
        return stated

    ref = reference_tables or load_reference_tables()
    classes = matched_classes(persona, ref)
    energies = [ref["reference_classes"][c].get("reference_energy_kcal")
                for c in classes if c in ref["reference_classes"]]
    energies = [float(e) for e in energies if e]
    if energies:
        return min(energies)  # strictest matched class, mirroring _resolve_bands
    return float(ref["modeling_assumptions"]["reference_energy_kcal_default"])


def map_persona_to_class(persona: dict, reference_tables: Optional[dict] = None):
    """Return ``(class_label, resolved_bands)`` for a persona.

    ``class_label`` is the single matched class, or a ``composite(...)`` label when a
    persona matches several conditions (comorbidity → stricter-band composite).
    """
    ref = reference_tables or load_reference_tables()
    classes = matched_classes(persona, ref)
    bands = _resolve_bands(classes, ref)
    label = classes[0] if len(classes) == 1 else "composite(" + "+".join(sorted(classes)) + ")"
    return label, bands


def _whole_serving(recipe: dict, col: str) -> Optional[float]:
    """Whole-serving amount of column ``col`` (HUMMUS values are per 100 g)."""
    if not recipe or col not in recipe:
        return None
    val = recipe.get(col)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    serving = recipe.get(SERVING_COL)
    try:
        serving = float(serving)
    except (TypeError, ValueError):
        serving = None
    return val / 100.0 * serving if serving else val


def plan_daily_totals(plan: Plan) -> dict:
    """Aggregate whole-serving nutrient totals over ALL filled slots of the plan.

    Returns ``{ref_nutrient: total}`` for the eight tracked nutrients, or ``{}`` if the
    plan has no resolved recipe. A missing nutrient column contributes nothing.
    """
    recipes = [r for r in plan.values() if r]
    if not recipes:
        return {}
    totals: dict = {}
    for nutrient, (col, scale) in NUTRIENT_COLS.items():
        vals = [_whole_serving(r, col) for r in recipes]
        vals = [v * scale for v in vals if v is not None]
        totals[nutrient] = sum(vals) if vals else None
    return totals


def _in_band(total: float, band: dict) -> int:
    low = band["low"] if band.get("band_type") != "upper_only" else 0
    low = low if low is not None else 0
    high = band["high"]
    if high is None:
        return 1 if total >= low else 0
    return 1 if low <= total <= high else 0


def _direction(total: float, band: dict) -> str:
    low = band["low"] if band.get("band_type") != "upper_only" else 0
    low = low if low is not None else 0
    high = band["high"]
    if high is not None and total > high:
        return "over"
    if total < low:
        return "under"
    return "within"


def hard_ceiling_breach(plan: Plan, persona: dict, reference_tables: Optional[dict] = None) -> bool:
    """True if any plan nutrient total exceeds its class ``hard_ceiling`` (a safety UL).

    Per the reference file's ``hard_ceiling_interaction``: a plan exceeding a UL
    is a SAFETY flag, folded into ``hard_constraint_violation`` for plan configs. False
    when the plan has no resolved recipes (nothing to breach).
    """
    ref = reference_tables or load_reference_tables()
    _, bands = map_persona_to_class(persona, ref)
    totals = plan_daily_totals(plan)
    for nutrient, band in bands.items():
        ceiling = band.get("hard_ceiling")
        total = totals.get(nutrient)
        if ceiling is not None and total is not None and total > ceiling:
            return True
    return False


def guideline_adherence(plan: Plan, persona: dict, reference_tables: Optional[dict] = None) -> dict:
    """The guideline-adherence block for a per-question record.

    Returns::

        {
          "guideline_adherence_overall": float,          # mean of 8 per-nutrient 0/1 (NaN if no recipes)
          "guideline_adherence_per_nutrient": {short_key: 0/1, ...},   # {} if no recipes
          "guideline_class": str,                        # resolved reference class label
          "guideline_direction": {short_key: "under"/"within"/"over"},  # for the report (not a record key)
          "guideline_hard_ceiling_breach": bool,         # feeds hard_constraint_violation
        }

    NaN / empty when the plan has no resolved recipe.
    """
    ref = reference_tables or load_reference_tables()
    label, bands = map_persona_to_class(persona, ref)
    totals = plan_daily_totals(plan)

    if not totals:
        return {
            "guideline_adherence_overall": float("nan"),
            "guideline_adherence_per_nutrient": {},
            "guideline_class": label,
            "guideline_direction": {},
            "guideline_hard_ceiling_breach": False,
        }

    per_nutrient: dict = {}
    direction: dict = {}
    breach = False
    for nutrient in ref["metric_computation"]["tracked_nutrients"]:
        short = SHORT_KEYS[nutrient]
        band = bands.get(nutrient)
        total = totals.get(nutrient)
        if band is None or total is None:
            # nutrient column absent for every recipe -> treat as out-of-band (0), unknown direction
            per_nutrient[short] = 0
            direction[short] = "unknown"
            continue
        per_nutrient[short] = _in_band(total, band)
        direction[short] = _direction(total, band)
        ceiling = band.get("hard_ceiling")
        if ceiling is not None and total > ceiling:
            breach = True

    overall = sum(per_nutrient.values()) / len(per_nutrient) if per_nutrient else float("nan")
    return {
        "guideline_adherence_overall": overall,
        "guideline_adherence_per_nutrient": per_nutrient,
        "guideline_class": label,
        "guideline_direction": direction,
        "guideline_hard_ceiling_breach": breach,
    }
