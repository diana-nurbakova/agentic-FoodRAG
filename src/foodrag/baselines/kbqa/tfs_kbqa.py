"""TFS-KBQA — fine-tuned semantic parsing (2024), adapted to HUMMUS (§5.8).

TFS-KBQA answers by *compiling the question to a structured query* with a model
fine-tuned in three steps, then executing it — sidestepping the traversal fan-out
that ToG/RoG fight (§5.8). Here the structured query is **Cypher** over HUMMUS.

Three pieces:

* :func:`compile_cypher` — a deterministic persona → Cypher compiler. It mirrors
  ``constraints.filter_valid`` (the gold-answer definition) edge-for-edge, so the
  query it emits selects exactly the persona's valid recipes, ranked health-first.
  This serves three roles: the **gold target** for generating the fine-tuning data
  (the QA set supplies training data, §5.8), the **fallback** when no model/LLM is
  available, and the **oracle** the trainer's outputs are checked against.
* :class:`TFSParser` — produces Cypher from the question: a fine-tuned checkpoint
  (trained by ``kbqa/training/tfs_train.py``) if present, else the base LLM
  prompted with the schema (like the existing ``KGQueryEngine``), else the
  deterministic compiler.
* :class:`TFSKBQA` — parse → execute → ranked ``recipe_url`` list, with the
  compiler as a safety net if a generated query errors or returns nothing.

Cypher uses the flat, space-truncated property names from the LOAD CSV import
(:func:`_prop` = text up to the first space), e.g. ``r.sugars``, ``r.servingSize``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from foodrag.baselines.kbqa import llm_io
from foodrag.baselines.kbqa.base import KBQAOptions, KBQASystem
from foodrag.baselines.kbqa.kg import RECIPE, HummusKG
from foodrag.baselines.kbqa.linking import constraint_text

DEFAULT_RANK = "coalesce(r.nutri_score, 0) DESC, coalesce(r.average_rating, 0) DESC"


def _prop(column: str) -> str:
    """Recipe property name as stored by the KG import (text up to first space)."""
    return column.split(" ")[0]


@dataclass
class CypherQuery:
    text: str
    params: dict = field(default_factory=dict)


def compile_cypher(persona: dict, k: int = 10, *, rank: str = DEFAULT_RANK) -> CypherQuery:
    """Compile a persona to a parametrised Cypher query selecting its valid recipes.

    Faithful, clause-for-clause, to ``constraints.filter_valid``: relative- and
    total-nutrient bounds, nutrition-score minimums, liked/disliked tags and
    liked/disliked (allergen) ingredients. Ranking is health-first by default so
    the same query also serves health-aware retrieval.
    """
    where: list[str] = []
    params: dict = {"k": k}
    pi = 0  # parameter index, for unique names

    for nutrient, spec in persona.get("relative_nutrients", {}).items():
        limits = spec.get("limits", {})
        lo, hi = f"lo{pi}", f"hi{pi}"
        params[lo], params[hi] = limits.get("lower"), limits.get("upper")
        where.append(f"r.`{_prop(nutrient)}` >= ${lo} AND r.`{_prop(nutrient)}` <= ${hi}")
        pi += 1

    for score_name, score_min in persona.get("nutrition_scores", {}).items():
        p = f"sc{pi}"
        params[p] = score_min
        where.append(f"r.`{_prop(score_name)}` >= ${p}")
        pi += 1

    for nutrient, total_max in persona.get("total_nutrients", {}).items():
        p = f"tot{pi}"
        params[p] = total_max
        where.append(f"(r.`{_prop(nutrient)}` / 100.0 * r.servingSize) <= ${p}")
        pi += 1

    tags = persona.get("tags", {})
    if tags.get("likes"):
        params["liked_tags"] = list(tags["likes"])
        where.append("EXISTS { MATCH (r)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $liked_tags }")
    if tags.get("dislikes"):
        params["disliked_tags"] = list(tags["dislikes"])
        where.append("NOT EXISTS { MATCH (r)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $disliked_tags }")

    ingredients = persona.get("ingredients", {})
    if ingredients.get("likes"):
        params["liked_ingr"] = list(ingredients["likes"])
        where.append("EXISTS { MATCH (r)-[:HAS_INGREDIENT]->(i:Ingredient) WHERE i.name IN $liked_ingr }")
    if ingredients.get("dislikes"):
        params["disliked_ingr"] = list(ingredients["dislikes"])
        where.append("NOT EXISTS { MATCH (r)-[:HAS_INGREDIENT]->(i:Ingredient) WHERE i.name IN $disliked_ingr }")

    clause = ("WHERE " + " AND ".join(where) + "\n") if where else ""
    text = (f"MATCH (r:{RECIPE})\n{clause}"
            f"RETURN r.recipe_url AS recipe_url\n"
            f"ORDER BY {rank}\nLIMIT $k")
    return CypherQuery(text, params)


def render_inline(query: CypherQuery) -> str:
    """Inline a parametrised query's params as literals → a self-contained Cypher
    string. Used to build fine-tuning *targets* (the model must emit runnable
    Cypher, not ``$params``). JSON encoding yields valid Cypher literals for
    numbers, strings and string lists.
    """
    import json
    import re

    text = query.text
    for name in sorted(query.params, key=len, reverse=True):  # longest first: $lo1 vs $lo10
        literal = json.dumps(query.params[name])
        text = re.sub(rf"\${re.escape(name)}\b", literal, text)
    return text


SCHEMA_HINT = """\
Graph schema (Neo4j):
- (:Recipe) properties: recipe_url, title, nutri_score, fsa_score, who_score (0–1,
  higher = healthier), average_rating, plus nutrient properties such as sugars,
  sodium, protein, totalFat, saturatedFat, cholesterol, totalCarbohydrate,
  dietaryFiber, calories, servingSize (per-100g values; total per serving =
  value/100*servingSize).
- (:Recipe)-[:HAS_TAG]->(:Tag {name})
- (:Recipe)-[:HAS_INGREDIENT]->(:Ingredient {name})
"""

PARSE_PROMPT = """\
Convert the user's request into a single Cypher query over a food knowledge graph.

{schema}
Rules:
- Return recipes as `RETURN r.recipe_url AS recipe_url`.
- Apply ALL stated constraints (tags, included/excluded ingredients, nutrient
  limits, health-score minimums).
- Order healthiest first (nutri_score DESC) and end with `LIMIT {k}`.
- Output ONLY the Cypher query, no prose, no code fences.

User request: {question}
Constraints:
{constraints}

Cypher:
"""


@dataclass
class TFSParser:
    """Question → Cypher: fine-tuned checkpoint → base-LLM prompt → compiler."""

    llm: Optional[Any] = None
    checkpoint: Optional[Any] = None

    @staticmethod
    def load_checkpoint(path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        return {"tokenizer": tok, "model": model}

    def parse(self, question: str, persona: dict, k: int) -> Optional[str]:
        """Return an inline Cypher string, or ``None`` to fall back to the compiler."""
        if self.checkpoint is not None:
            return self._parse_checkpoint(question, persona, k)
        if self.llm is not None:
            text = llm_io.complete(self.llm, PARSE_PROMPT.format(
                schema=SCHEMA_HINT, k=k, question=question, constraints=constraint_text(persona)))
            return _clean_cypher(text) or None
        return None

    def _parse_checkpoint(self, question: str, persona: dict, k: int) -> Optional[str]:
        tok, model = self.checkpoint["tokenizer"], self.checkpoint["model"]
        prompt = parser_instruction(question, persona, k)
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _clean_cypher(text) or None


def parser_instruction(question: str, persona: dict, k: int) -> str:
    """Frozen training/inference prompt for the fine-tuned parser (shared with the
    trainer so prompts are byte-identical)."""
    return PARSE_PROMPT.format(schema=SCHEMA_HINT, k=k, question=question,
                               constraints=constraint_text(persona))


def _clean_cypher(text: str) -> str:
    """Strip code fences / stray prose, keep from the first MATCH onward."""
    if not text:
        return ""
    text = text.replace("```cypher", "").replace("```", "").strip()
    upper = text.upper()
    idx = upper.find("MATCH")
    return text[idx:].strip() if idx >= 0 else ""


class TFSKBQA(KBQASystem):
    name = "tfs_kbqa"

    def __init__(self, kg: HummusKG, llm: Optional[Any] = None,
                 parser: Optional[TFSParser] = None, options: Optional[KBQAOptions] = None):
        self.kg = kg
        self.llm = llm
        self.opt = options or KBQAOptions()
        self.parser = parser or TFSParser(llm=llm)

    def retrieve(self, qa_item: dict, k: int) -> list[str]:
        question = qa_item.get(self.opt.query_field) or qa_item.get("question", "")
        persona = qa_item.get("persona", {})

        generated = self.parser.parse(question, persona, k)
        if generated:
            urls = self._execute_inline(generated)
            if urls:
                return urls[:k]

        # Fallback / safety net: the deterministic compiler (parametrised).
        query = compile_cypher(persona, k)
        return self._execute(query)[:k]

    def _execute(self, query: CypherQuery) -> list[str]:
        try:
            rows = self.kg.client.run(query.text, **query.params)
        except Exception:
            return []
        return _urls(rows)

    def _execute_inline(self, cypher: str) -> list[str]:
        try:
            rows = self.kg.client.run(cypher)
        except Exception:
            return []
        return _urls(rows)


def _urls(rows: list[dict]) -> list[str]:
    out, seen = [], set()
    for row in rows:
        url = row.get("recipe_url")
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out
