"""§8 Explainability metrics — graph-path coverage/validity + NL-claim grounding.

FoodRAG's explanations have two layers (Evaluation Protocol §8):

* an **intrinsic KG-path layer** (structural, extracted from HUMMUS recipe
  attributes), evaluated by :func:`path_coverage` / :func:`path_validity` (§8.1);
* a **natural-language rationale layer** (Enhancement Agent, user-facing), which
  can *hallucinate*, evaluated by :func:`extract_claims` + :func:`claim_grounding`
  (§8.2).

The contrast between the two layers is the finding: **the KG-path layer is valid
by construction** — every supporting path here is read directly off a real recipe
attribute (a real HUMMUS edge), so :func:`path_validity` is 1.0 whenever anything
is covered — whereas the NL layer's claims are re-derived by an LLM and must be
checked one-by-one, so its grounding rate can (and does) fall below 1.0.

Everything is deterministic + rule-based and memory-light: it needs only the
persona dict and the *recommended* recipe rows (mappings of HUMMUS columns), never
Neo4j or the 7GB HUMMUS graph. The "constraint-first, graph-second" framing of §8
is what makes this sound: because we operate only over the already-recommended
recipes (a tiny induced subgraph), the KG-path evidence is a plain attribute lookup.

── §8.1 requirement enumeration ────────────────────────────────────────────────
Each atomic persona requirement becomes one "requirement to explain":

* ``relative_nutrients[n]`` bound  → supported if some recipe's ``row[n]`` ∈ [lo,hi]
* ``nutrition_scores[s]`` minimum  → supported if some recipe's ``row[s]`` >= min
* ``total_nutrients[n]`` maximum   → supported if some recipe's whole-serving
                                     total (``constraints._total``) <= max
* ``tags.likes[t]``                → supported if some recipe has liked tag ``t``
* ``ingredients.likes[i]``         → supported if some recipe contains liked ``i``
* ``tags.dislikes[t]`` (exclusion) → supported if ALL recipes avoid tag ``t``
* ``ingredients.dislikes[i]`` (allergen/exclusion) → supported if ALL recipes
                                     avoid ingredient ``i``

An *exclusion* requirement is "supported" precisely when the exclusion holds and is
therefore explainable ("excludes X — none of the recommended recipes contain it").
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from foodrag.baselines import constraints

Row = Optional[dict]


# ── §8.1 graph-path explanations ─────────────────────────────────────────────

def _resolved(recipes: Sequence[Row]) -> list[dict]:
    return [r for r in recipes if r]


def supporting_paths(persona: dict, recipes: Sequence[Row],
                     expand: constraints.IngredientExpander = constraints.identity_expand) -> list[dict]:
    """Enumerate every stated requirement and whether the recommended recipes
    provide explicit supporting attribute evidence (a constructed recipe→requirement path).

    Returns one dict per stated requirement::

        {"kind": <str>, "target": <str>, "supported": <bool>}

    ``supported`` means an explicit supporting path was constructed from a real
    recipe attribute (see the module docstring for the per-kind rule). The result
    length is the number of stated requirements; the number of ``supported=True``
    entries is the covered count.
    """
    rows = _resolved(recipes)
    paths: list[dict] = []

    # inclusion-style nutrient / score requirements: supported if >=1 recipe provides it
    for nutrient, limits in persona.get("relative_nutrients", {}).items():
        lo = limits["limits"]["lower"]
        hi = limits["limits"]["upper"]
        ok = any(lo <= r[nutrient] <= hi for r in rows if nutrient in r)
        paths.append({"kind": "relative_nutrient", "target": nutrient, "supported": ok})

    for score_name, score_min in persona.get("nutrition_scores", {}).items():
        ok = any(r[score_name] >= score_min for r in rows if score_name in r)
        paths.append({"kind": "nutrition_score", "target": score_name, "supported": ok})

    for nutrient, total_max in persona.get("total_nutrients", {}).items():
        ok = any(constraints._total(r, nutrient) <= total_max
                 for r in rows if nutrient in r and constraints.SERVING_COL in r)
        paths.append({"kind": "total_nutrient", "target": nutrient, "supported": ok})

    for tag in persona.get("tags", {}).get("likes", []):
        ok = any(tag in (r.get("tags") or []) for r in rows)
        paths.append({"kind": "tag_like", "target": tag, "supported": ok})

    for ingr in expand(persona.get("ingredients", {}).get("likes", [])):
        ok = any(ingr in (r.get("ingredient_names") or []) for r in rows)
        paths.append({"kind": "ingredient_like", "target": ingr, "supported": ok})

    # exclusion-style requirements: supported iff ALL recommended recipes avoid it
    # (the exclusion holds and is therefore explainable). With no recipes, an
    # exclusion is vacuously held but there is nothing to explain -> unsupported.
    for tag in persona.get("tags", {}).get("dislikes", []):
        ok = bool(rows) and all(tag not in (r.get("tags") or []) for r in rows)
        paths.append({"kind": "tag_dislike", "target": tag, "supported": ok})

    for ingr in expand(persona.get("ingredients", {}).get("dislikes", [])):
        ok = bool(rows) and all(ingr not in (r.get("ingredient_names") or []) for r in rows)
        paths.append({"kind": "ingredient_dislike", "target": ingr, "supported": ok})

    return paths


def path_coverage(persona: dict, recipes: Sequence[Row],
                  expand: constraints.IngredientExpander = constraints.identity_expand) -> float:
    """§8.1 Path Coverage = (# requirements with a supporting path) / (# stated requirements).

    NaN when the persona states no requirements (nothing to cover).
    """
    paths = supporting_paths(persona, recipes, expand)
    if not paths:
        return float("nan")
    return sum(1 for p in paths if p["supported"]) / len(paths)


def path_validity(persona: dict, recipes: Sequence[Row],
                  expand: constraints.IngredientExpander = constraints.identity_expand) -> float:
    """§8.1 Path Validity = fraction of constructed supporting paths whose edges are
    real HUMMUS edges.

    **Valid by construction:** every supporting path this module constructs is read
    directly off a real recipe attribute (a real HUMMUS edge — the recipe *does*
    have that tag / that ingredient / that nutrient value), so validity is 1.0
    whenever >=1 requirement is covered. This is the deliberate contrast with the NL
    layer (:func:`claim_grounding`), whose claims are LLM-re-derived and can be
    unsupported. Returns 1.0 when there is >=1 covered requirement, NaN otherwise
    (no covered requirement → no path to validate).
    """
    paths = supporting_paths(persona, recipes, expand)
    covered = any(p["supported"] for p in paths)
    return 1.0 if covered else float("nan")


# ── §8.2 NL-explanation grounding ────────────────────────────────────────────
#
# The explanations are templated NL (see ``ExplanationTool.EXPLANATION_TEMPLATE``).
# ``extract_claims`` pulls typed, checkable factual claims out via rule-based regex;
# ``claim_grounding`` verifies each against the recommended recipes' attributes.
# Claim types (matched to the real template wording):
#
#   score_min      "FSA score of at least 0.06", "NutriScore of at least 0.7"
#   nutriscore_letter  "NutriScore ... B", "a NutriScore of B or better"
#   nutrient_range "within your requested range of 30 to 300 grams" (with a nearby
#                   nutrient word so we know which column) — also "fat content ... 30-300 g"
#   nutrient_max   "below your 356.06mg limit" / "stays below 356.06 mg" (with unit)
#   ingredient_include "includes spinach", "contains tofu"
#   ingredient_exclude "excludes peanut", "without peanut", "free of peanut"
#   tag_exclude    "exclude any ingredients with Californian tags", "excludes Californian tags"

_NUM = r"[-+]?\d+(?:\.\d+)?"

# nutrient phrase -> recipes_df column (for range/max claims). Matched case-insensitively.
_NUTRIENT_WORDS = {
    "calorie": "calories [kcal]", "calories": "calories [kcal]", "energy": "calories [kcal]",
    "protein": "protein [g]",
    "carb": "totalCarbohydrate [g]", "carbohydrate": "totalCarbohydrate [g]",
    "fat": "totalFat [g]", "total fat": "totalFat [g]",
    "saturated fat": "saturatedFat [g]", "satfat": "saturatedFat [g]",
    "fiber": "dietaryFiber [g]", "fibre": "dietaryFiber [g]",
    "sodium": "sodium [mg]", "salt": "sodium [mg]",
    "sugar": "sugars [g]", "sugars": "sugars [g]",
    "cholesterol": "cholesterol [mg]",
}
# score phrase -> recipes_df column (for score-min claims).
_SCORE_WORDS = {
    "fsa": "fsa_score", "fsa score": "fsa_score",
    "nutriscore": "nutri_score", "nutri score": "nutri_score", "nutri-score": "nutri_score",
}
# NutriScore grade letters -> normalised nutri_score value (A=5..E=1 over /5 -> 1.0..0.2).
_GRADE_TO_SCORE = {"a": 1.0, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}

_NUTRIENT_ALT = "|".join(sorted((re.escape(w) for w in _NUTRIENT_WORDS), key=len, reverse=True))
_SCORE_ALT = "|".join(sorted((re.escape(w) for w in _SCORE_WORDS), key=len, reverse=True))

# "FSA score of at least 0.06" / "an FSA score of 0.06 or higher"
_RE_SCORE_MIN = re.compile(
    rf"(?P<score>{_SCORE_ALT})[^.]*?(?:of\s+)?(?:at\s+least|minimum|min\.?|>=|≥|"
    rf"of|above)\s+(?P<val>{_NUM})",
    re.IGNORECASE)
# "NutriScore ... of B", "a NutriScore of B or better", "NutriScore B"
_RE_NUTRISCORE_LETTER = re.compile(
    r"nutri[\s-]?score\b[^.]*?\b(?:of\s+|:\s*)?(?P<grade>[A-E])\b(?![a-z])",
    re.IGNORECASE)
# "range of 30 to 300 grams" / "between 30 and 300 g" / "30-300 g", with a nutrient word nearby before it
_RE_NUTRIENT_RANGE = re.compile(
    rf"(?P<nutrient>{_NUTRIENT_ALT})\b(?P<mid>[^.,]*?)"
    rf"(?:range\s+of\s+|between\s+)?(?P<lo>{_NUM})\s*(?:to|-|–|and)\s*(?P<hi>{_NUM})\s*"
    rf"(?P<unit>g|grams?|mg|milligrams?|kcal)?",
    re.IGNORECASE)
# "cholesterol ... below your 356.06mg limit" / "sodium stays below 500 mg".
# The gap between the nutrient word and the bound keyword must stay within one clause
# ([^.,] — no comma / period) so the NEAREST preceding nutrient word wins (e.g. in
# "…fat range 30-300 g, and the cholesterol levels stay below 356 mg" the max binds to
# cholesterol, not fat).
_RE_NUTRIENT_MAX = re.compile(
    rf"(?P<nutrient>{_NUTRIENT_ALT})\b[^.,]*?(?:below|under|less\s+than|no\s+more\s+than|<=|≤|max(?:imum)?\s+(?:of\s+)?)\s*"
    rf"(?:your\s+)?(?P<val>{_NUM})\s*(?P<unit>g|grams?|mg|milligrams?|kcal)?",
    re.IGNORECASE)
# "excludes ... Californian tags" / "exclude any ingredients with Californian tags"
_RE_TAG_EXCLUDE = re.compile(
    r"(?:exclude[s]?|without|free\s+of|avoid[s]?|no)\b[^.]*?\b(?P<tag>[A-Za-z][\w-]*)\s+tags?\b",
    re.IGNORECASE)
# "excludes peanut" / "without peanuts" / "free of shellfish" (ingredient, no "tag" word)
_RE_INGREDIENT_EXCLUDE = re.compile(
    r"(?:exclude[s]?|without|free\s+of|contains\s+no|no)\s+(?P<ingredient>[a-z][\w-]*)\b(?!\s+tags?\b)",
    re.IGNORECASE)
# "includes spinach" / "contains tofu" / "features salmon"
_RE_INGREDIENT_INCLUDE = re.compile(
    r"(?:include[s]?|contain[s]?|feature[s]?|with)\s+(?P<ingredient>[a-z][\w-]*)\b(?!\s+tags?\b)",
    re.IGNORECASE)

# stop-words that are never real ingredients (avoid false "includes <word>" claims)
_INGREDIENT_STOP = {
    "a", "an", "the", "your", "any", "all", "each", "this", "that", "these", "those",
    "and", "or", "of", "no", "not", "some", "meals", "meal", "plan", "recipes",
    "recipe", "selection", "ingredients", "ingredient", "them", "it", "dishes",
    "better", "least", "most", "more", "less", "than", "you", "requested",
}


def _clean_token(tok: str) -> str:
    return tok.strip().strip(".,;:'\"").lower()


def extract_claims(explanation_text: str) -> list[dict]:
    """Rule-based extraction of typed, checkable factual claims from a templated NL
    explanation (§8.2). Returns a list of claim dicts, each with a ``type`` and the
    fields needed to verify it against a recipe row. Empty list if the text is empty
    or yields no recognisable claim.

    Claim schemas::

        {"type": "score_min",      "column": <col>, "value": <float>}
        {"type": "nutrient_range", "column": <col>, "low": <float>, "high": <float>}
        {"type": "nutrient_max",   "column": <col>, "value": <float>}
        {"type": "ingredient_include", "ingredient": <str>}
        {"type": "ingredient_exclude", "ingredient": <str>}
        {"type": "tag_exclude",        "tag": <str>}
    """
    if not explanation_text or not explanation_text.strip():
        return []
    text = explanation_text
    claims: list[dict] = []

    # score minimums (FSA / NutriScore numeric)
    for m in _RE_SCORE_MIN.finditer(text):
        col = _SCORE_WORDS[_clean_token(m.group("score"))]
        claims.append({"type": "score_min", "column": col, "value": float(m.group("val"))})

    # NutriScore letter grade -> normalised min score
    for m in _RE_NUTRISCORE_LETTER.finditer(text):
        grade = m.group("grade").lower()
        claims.append({"type": "score_min", "column": "nutri_score",
                       "value": _GRADE_TO_SCORE[grade], "grade": grade.upper()})

    # nutrient ranges (need both bounds)
    for m in _RE_NUTRIENT_RANGE.finditer(text):
        col = _NUTRIENT_WORDS[_clean_token(m.group("nutrient"))]
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if lo > hi:
            lo, hi = hi, lo
        claims.append({"type": "nutrient_range", "column": col, "low": lo, "high": hi})

    # nutrient maxima ("below X mg")
    for m in _RE_NUTRIENT_MAX.finditer(text):
        col = _NUTRIENT_WORDS[_clean_token(m.group("nutrient"))]
        claims.append({"type": "nutrient_max", "column": col, "value": float(m.group("val"))})

    # tag exclusions ("excludes ... Californian tags")
    for m in _RE_TAG_EXCLUDE.finditer(text):
        tag = _clean_token(m.group("tag"))
        if tag and tag not in _INGREDIENT_STOP:
            claims.append({"type": "tag_exclude", "tag": tag})

    # ingredient inclusions / exclusions — skip spans already consumed as a tag claim
    for m in _RE_INGREDIENT_EXCLUDE.finditer(text):
        ingr = _clean_token(m.group("ingredient"))
        if ingr and ingr not in _INGREDIENT_STOP:
            claims.append({"type": "ingredient_exclude", "ingredient": ingr})
    for m in _RE_INGREDIENT_INCLUDE.finditer(text):
        ingr = _clean_token(m.group("ingredient"))
        if ingr and ingr not in _INGREDIENT_STOP:
            claims.append({"type": "ingredient_include", "ingredient": ingr})

    return _dedupe_claims(claims)


def _dedupe_claims(claims: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in claims:
        key = tuple(sorted((k, str(v)) for k, v in c.items() if k != "grade"))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _claim_supported(claim: dict, rows: list[dict]) -> bool:
    """True if the claim is verifiable against >=1 recommended recipe (include /
    numeric claims) or against ALL of them (exclusion claims)."""
    t = claim["type"]
    if t == "score_min":
        col, val = claim["column"], claim["value"]
        return any(col in r and r[col] is not None and r[col] >= val for r in rows)
    if t == "nutrient_range":
        col, lo, hi = claim["column"], claim["low"], claim["high"]
        return any(col in r and r[col] is not None and lo <= r[col] <= hi for r in rows)
    if t == "nutrient_max":
        col, val = claim["column"], claim["value"]
        return any(col in r and r[col] is not None and r[col] <= val for r in rows)
    if t == "ingredient_include":
        ingr = claim["ingredient"]
        return any(_contains_ingredient(r, ingr) for r in rows)
    if t == "ingredient_exclude":
        ingr = claim["ingredient"]
        return bool(rows) and all(not _contains_ingredient(r, ingr) for r in rows)
    if t == "tag_exclude":
        tag = claim["tag"]
        return bool(rows) and all(not _has_tag(r, tag) for r in rows)
    return False


def _contains_ingredient(row: dict, ingr: str) -> bool:
    return any(ingr == str(i).lower() or ingr in str(i).lower()
               for i in (row.get("ingredient_names") or []))


def _has_tag(row: dict, tag: str) -> bool:
    return any(tag == str(t).lower() or tag in str(t).lower()
               for t in (row.get("tags") or []))


def claim_grounding(claims: Sequence[dict], recipes: Sequence[Row]) -> dict:
    """§8.2 Claim Grounding — verify extracted claims against the recommended recipes.

    Returns ``{"claim_grounding_rate": supported/total, "unsupported_claim_rate":
    unsupported/total}``. Both NaN when no claims were extracted (nothing to ground).
    Unresolved (``None``) recipes are dropped before checking (a hallucinated recipe
    provides no attribute evidence, so a claim relying on it is unsupported).
    """
    if not claims:
        return {"claim_grounding_rate": float("nan"), "unsupported_claim_rate": float("nan")}
    rows = _resolved(recipes)
    supported = sum(1 for c in claims if _claim_supported(c, rows))
    total = len(claims)
    return {
        "claim_grounding_rate": supported / total,
        "unsupported_claim_rate": (total - supported) / total,
    }


# ── one-shot §8 block ────────────────────────────────────────────────────────

def explainability_metrics(persona: dict, recipes: Sequence[Row], *,
                           explanation_text: str = "",
                           expand: constraints.IngredientExpander = constraints.identity_expand) -> dict:
    """The full §8 explainability block for a §13.1 record.

    ``path_coverage`` / ``path_validity`` (§8.1) are always computed from the
    recommended recipes. ``claim_grounding_rate`` / ``unsupported_claim_rate`` (§8.2)
    are computed only when ``explanation_text`` is non-empty (they need the NL layer);
    otherwise they are NaN — populating on the new-code configs that persist the text.
    """
    block = {
        "path_coverage": path_coverage(persona, recipes, expand),
        "path_validity": path_validity(persona, recipes, expand),
    }
    if explanation_text and explanation_text.strip():
        claims = extract_claims(explanation_text)
        block.update(claim_grounding(claims, recipes))
    else:
        block["claim_grounding_rate"] = float("nan")
        block["unsupported_claim_rate"] = float("nan")
    return block
