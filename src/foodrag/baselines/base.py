"""Baseline interface + stratification helpers for the §5.7 baseline ladder.

A ``Baseline`` turns one QA item (persona + question) into a ranked list of
``recipe_url`` ids. The runner (:mod:`foodrag.baselines.runner`) scores that list
with the deterministic metrics and tags each record with the strata of §9 so the
Report tab can switch between aggregate and stratified views.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

# §9.1 answer-set-size strata.
TIGHT_MAX = 50
MEDIUM_MAX = 5_000


class Baseline(ABC):
    """A baseline-ladder rung.

    Subclasses set :attr:`rung` (e.g. ``"0a"``) and :attr:`config_id` (the stable
    key used in result files, e.g. ``"popularity_raw"``) and implement
    :meth:`recommend`.
    """

    rung: str = ""
    config_id: str = ""

    @abstractmethod
    def recommend(self, qa_item: dict, k: int) -> list[str]:
        """Return up to ``k`` ranked ``recipe_url`` ids for the QA item."""
        raise NotImplementedError


# ── stratification (§9) ──────────────────────────────────────────────────────

def answer_set_stratum(answer_set_size: int) -> str:
    """Map a gold answer-set size to the §9.1 stratum label.

    Size 0 (no recipe satisfies the persona — an over-constrained/unsatisfiable
    question) is its own ``empty`` stratum: §9.1 defines ``tight`` as 1–50, so
    folding zeros into ``tight`` would swamp it with unsatisfiable personas.
    """
    if answer_set_size <= 0:
        return "empty"
    if answer_set_size <= TIGHT_MAX:
        return "tight"
    if answer_set_size <= MEDIUM_MAX:
        return "medium"
    return "broad"


def requirement_count(persona: dict) -> int:
    """Number of atomic stated requirements (§9.2): every liked/disliked tag &
    ingredient plus every nutrient/score bound counts once."""
    tags = persona.get("tags", {})
    ingredients = persona.get("ingredients", {})
    return (
        len(tags.get("likes", [])) + len(tags.get("dislikes", []))
        + len(ingredients.get("likes", [])) + len(ingredients.get("dislikes", []))
        + len(persona.get("relative_nutrients", {}))
        + len(persona.get("total_nutrients", {}))
        + len(persona.get("nutrition_scores", {}))
    )


def requirement_types(persona: dict) -> list[str]:
    """Requirement-type tags present in the persona (§9.3), ordered & de-duped.

    * ``allergen``       — disliked (blacklisted) ingredients (safety-critical)
    * ``ingredient``     — liked-ingredient preferences
    * ``tag``            — liked/disliked tags
    * ``nutrient_bound`` — relative- or total-nutrient numeric bounds
    * ``health``         — minimum nutrition-score thresholds (WHO/FSA/Nutri)
    """
    ingredients = persona.get("ingredients", {})
    tags = persona.get("tags", {})
    types: list[str] = []
    if ingredients.get("dislikes"):
        types.append("allergen")
    if ingredients.get("likes"):
        types.append("ingredient")
    if tags.get("likes") or tags.get("dislikes"):
        types.append("tag")
    if persona.get("relative_nutrients") or persona.get("total_nutrients"):
        types.append("nutrient_bound")
    if persona.get("nutrition_scores"):
        types.append("health")
    return types
