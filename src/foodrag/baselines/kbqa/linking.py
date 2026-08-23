"""Persona/question → KBQA inputs: topic entities + a constraint rendering.

The HUMMUS QA item is ``{question, persona, answer, ...}`` (qa_generator). The
KBQA systems need two things out of it:

* **Topic entities** — the seed nodes a graph walk starts from. For a food
  persona these are the *liked* ingredients and *liked* tags (the things the
  recipe must positively contain), resolved to real KG nodes via
  :class:`~foodrag.baselines.kbqa.kg.HummusKG`. Disliked ingredients are
  *exclusion* constraints, not walk seeds, so they are not linked as topics.
* **Constraint text** — a frozen, deterministic natural-language rendering of the
  persona's requirements, injected into every LLM prompt so each baseline gets
  the *same* health information FoodRAG gets.

Keeping both here means ToG, RoG and TFS-KBQA share one notion of "what the user
asked for", and it matches the persona structure the gold set was built from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from foodrag.baselines.kbqa.kg import Entity, HummusKG


@dataclass
class LinkedQuery:
    """A QA item resolved into KBQA inputs."""

    question: str
    persona: dict
    topic_entities: list[Entity] = field(default_factory=list)
    constraint_text: str = ""


def link_query(qa_item: dict, kg: HummusKG, *, query_field: str = "question",
               max_topics: int = 8) -> LinkedQuery:
    """Resolve a QA item to topic entities + constraint text.

    Liked ingredients are linked first (most specific seeds), then liked tags,
    capped at ``max_topics`` so a persona with many likes does not seed an
    unbounded beam.
    """
    persona = qa_item["persona"]
    ingredients = persona.get("ingredients", {})
    tags = persona.get("tags", {})

    topics: list[Entity] = []
    for name in ingredients.get("likes", []):
        topics.extend(kg.link_ingredient(name, limit=2))
    for name in tags.get("likes", []):
        topics.extend(kg.link_tag(name, limit=2))

    # De-dup on (label, key) while preserving order, then cap.
    seen, deduped = set(), []
    for e in topics:
        sig = (e.label, e.key)
        if sig not in seen:
            seen.add(sig)
            deduped.append(e)
    deduped = deduped[:max_topics]

    return LinkedQuery(
        question=qa_item.get(query_field) or qa_item.get("question", ""),
        persona=persona,
        topic_entities=deduped,
        constraint_text=constraint_text(persona),
    )


def constraint_text(persona: dict) -> str:
    """Render a persona's requirements as deterministic, prompt-ready bullet text.

    Mirrors the requirement families the gold set is built from (tags, included /
    excluded ingredients, relative- and total-nutrient bounds, nutrition-score
    minimums) so the LLM sees exactly the constraints the metrics score against.
    """
    lines: list[str] = []
    tags = persona.get("tags", {})
    if tags.get("likes"):
        lines.append(f"- Must have at least one of these tags: {', '.join(tags['likes'])}.")
    if tags.get("dislikes"):
        lines.append(f"- Must NOT have any of these tags: {', '.join(tags['dislikes'])}.")

    ingredients = persona.get("ingredients", {})
    if ingredients.get("likes"):
        lines.append(f"- Must include at least one of these ingredients: {', '.join(ingredients['likes'])}.")
    if ingredients.get("dislikes"):
        lines.append(f"- Must EXCLUDE all of these ingredients (allergens/blacklist): "
                     f"{', '.join(ingredients['dislikes'])}.")

    for nutrient, spec in persona.get("relative_nutrients", {}).items():
        limits = spec.get("limits", {})
        lines.append(f"- {nutrient} must be between {limits.get('lower')} and {limits.get('upper')} (per 100 g).")
    for nutrient, total_max in persona.get("total_nutrients", {}).items():
        lines.append(f"- Total {nutrient} per serving must be at most {total_max}.")
    for score_name, score_min in persona.get("nutrition_scores", {}).items():
        lines.append(f"- {score_name} must be at least {score_min} (0–1, higher is healthier).")

    return "\n".join(lines) if lines else "- No explicit constraints."
