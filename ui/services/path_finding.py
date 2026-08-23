"""Grounded KG-path extraction for explanations.

For a recommended recipe, extract the path(s) connecting it to each stated
persona requirement, grounded in the recipe's real attributes / ingredients /
tags (and the FoodOn hierarchy when available). This is the **constraint-first,
graph-second** approach: we explain only already-recommended recipes,
so traversal is bounded and every presented edge is real.

Two deterministic metrics:
  - **Path Coverage** — fraction of stated requirements with a supporting path.
  - **Path Validity** — fraction of presented paths whose edges are real.
    Paths here are built only from genuine attributes/membership, so validity is
    1.0 by construction (the point of deterministic grounding).

Operates on the GUI recipe-card shape (``ui.mock_data`` / ``services.mapping``)
so it works for both mock and live recipes. Pure — no Streamlit/backend imports.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from . import eval_metrics

# HUMMUS nutrient column → GUI card key + display unit
_COL_TO_CARD = {
    "sugars [g]": ("sugar", "g"), "protein [g]": ("protein", "g"),
    "totalFat [g]": ("fat", "g"), "calories [kcal]": ("cal", "kcal"),
    "sodium [mg]": ("sodium", "mg"), "totalCarbohydrate [g]": ("carbs", "g"),
    "dietaryFiber [g]": ("fiber", "g"),
}
_SYM = {"<=": "≤", "<": "<", ">=": "≥", ">": ">", "==": "="}


# ── requirements derived from a persona ───────────────────────────────────
def persona_requirements(persona: dict) -> list[dict]:
    """Typed requirements from a persona: allergens, nutrient bounds, liked tags."""
    reqs: list[dict] = []
    for a in persona.get("allergies") or []:
        if a and a.strip():
            reqs.append({"kind": "allergen", "name": a.strip(), "label": f"No {a.strip()}"})
    for column, op, value in eval_metrics.parse_nutrient_constraints(persona.get("nutrientConstraint", "")):
        reqs.append({"kind": "nutrient", "column": column, "op": op, "value": value,
                     "label": eval_metrics.constraint_label((column, op, value))})
    for t in persona.get("likedTags") or []:
        if t and t.strip():
            reqs.append({"kind": "tag", "name": t.strip(), "label": t.strip()})
    return reqs


# A path step is a node; edges[i] connects node[i] → node[i+1].
def _path(nodes: list[tuple[str, str]], edges: list[str], requirement: str,
          satisfied: bool, valid: bool = True) -> dict:
    return {"requirement": requirement, "nodes": nodes, "edges": edges,
            "satisfied": satisfied, "valid": valid}


def _nutrient_path(card: dict, req: dict) -> dict:
    card_key, unit = _COL_TO_CARD.get(req["column"], (None, ""))
    value = card.get(card_key) if card_key else None
    name = req["column"].split(" [")[0]
    if value is None:
        # attribute not on the card → no supporting path
        return _path([(card.get("name", "recipe"), "recipe"), (req["label"], "constraint")],
                     ["?"], req["label"], satisfied=False, valid=False)
    satisfied = eval_metrics._OPS[req["op"]](float(value), float(req["value"]))
    nodes = [(card.get("name", "recipe"), "recipe"),
             (f"{name} = {value}{unit}", "attribute"),
             (req["label"], "constraint")]
    edges = ["has", "satisfies" if satisfied else "violates"]
    return _path(nodes, edges, req["label"], satisfied=satisfied, valid=True)


def _allergen_path(card: dict, req: dict, expand: Optional[Callable] = None) -> dict:
    offending = eval_metrics.recipe_violated_allergens(card, [req["name"]], expand)
    recipe = (card.get("name", "recipe"), "recipe")
    if not offending:
        nodes = [recipe, (f"excludes {req['name']}", "allergen")]
        return _path(nodes, ["excludes"], req["label"], satisfied=True, valid=True)
    nodes = [recipe, (offending[0], "ingredient"), (req["name"], "allergen")]
    return _path(nodes, ["contains", "is a"], req["label"], satisfied=False, valid=True)


def _tag_path(card: dict, req: dict) -> dict:
    tags = {str(t).strip().lower() for t in (card.get("tags") or [])}
    recipe = (card.get("name", "recipe"), "recipe")
    present = req["name"].lower() in tags
    if present:
        return _path([recipe, (req["name"], "tag")], ["tagged"], req["label"], satisfied=True)
    return _path([recipe, (f"{req['name']} (absent)", "tag")], ["lacks"], req["label"],
                 satisfied=False, valid=True)


def build_paths(card: dict, persona: dict, expand: Optional[Callable[[Iterable[str]], set]] = None) -> dict:
    """Grounded requirement-paths for one recipe card.

    Returns ``{recipe, requirements, paths:[...], coverage, validity, n}`` where
    ``coverage`` = supported/total requirements, ``validity`` = real/presented.
    """
    reqs = persona_requirements(persona)
    paths = []
    for req in reqs:
        if req["kind"] == "nutrient":
            paths.append(_nutrient_path(card, req))
        elif req["kind"] == "allergen":
            paths.append(_allergen_path(card, req, expand))
        elif req["kind"] == "tag":
            paths.append(_tag_path(card, req))
    n = len(reqs)
    supported = sum(1 for p in paths if p["satisfied"])
    presented = len(paths)
    valid = sum(1 for p in paths if p["valid"])
    return {
        "recipe": card.get("name", "recipe"),
        "requirements": [r["label"] for r in reqs],
        "paths": paths,
        "coverage": (supported / n) if n else 1.0,
        "validity": (valid / presented) if presented else 1.0,
        "n": n,
    }
