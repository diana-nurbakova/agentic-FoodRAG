"""ToG — Think-on-Graph (Sun et al., ICLR 2024), adapted to HUMMUS (§5.8).

Training-free agentic KG traversal: the LLM does *beam search on the graph*,
alternating between exploring the graph and pruning with the model. Each depth is
the faithful two-step ToG cycle:

1. **Relation exploration** — for every entity at the frontier, list its candidate
   relations (schema-driven, §5.5 capped) and ask the LLM to *prune* to the most
   useful ones for the question + constraints (the ToG "relation search + prune").
2. **Entity exploration** — expand the kept relations one hop (bounded fan-out,
   hub-pruned, §5.5), then ask the LLM to *score* the candidate tail entities; keep
   the global top-``beam_width`` partial reasoning paths.

After each depth a **reasoning** call decides whether the paths gathered so far
are adequate to answer (recommend recipes); if so the walk stops early. Recipe
nodes met anywhere along surviving paths are the candidate answers, ranked by the
score of the best path that reached them and tie-broken health-aware (§5.7).

Adaptation notes (faithful-reproduction discipline, §5.8):
* Native by default — ToG is *not* handed the persona's valid set; constraints
  enter only through the prompts (fairness) and the final health blend. The
  optional ``constraint_first`` mode (§5.5) is a tractability switch, off here.
* The LLM is injected and may be ``None`` (deterministic fallback: uniform
  relation/entity scores, stop when ≥k recipes are found) so the traversal is
  testable without a generator or live graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from foodrag.baselines.kbqa import llm_io
from foodrag.baselines.kbqa.base import KBQAOptions, KBQASystem, blend_order
from foodrag.baselines.kbqa.kg import RECIPE, Entity, HummusKG, Relation, RecipeScope
from foodrag.baselines.kbqa.linking import LinkedQuery, link_query

RELATION_PRUNE_PROMPT = """\
You are exploring a food knowledge graph to recommend recipes.

Question: {question}
User constraints:
{constraints}

You are at the entity: "{entity}" (type: {label}).
Candidate relations to explore next:
{relations}

Score how useful following each relation is for reaching recipes that satisfy the
question and constraints, on a scale from 0 (useless) to 1 (essential).
Return ONLY a JSON list like [{{"id": 0, "score": 0.9}}, {{"id": 1, "score": 0.2}}].
"""

ENTITY_PRUNE_PROMPT = """\
You are exploring a food knowledge graph to recommend recipes.

Question: {question}
User constraints:
{constraints}

You followed the relation "{relation}" from "{entity}" and reached these candidates:
{entities}

Score how likely each candidate leads to (or is) a recipe satisfying the question
and constraints, from 0 to 1.
Return ONLY a JSON list like [{{"id": 0, "score": 0.8}}, {{"id": 1, "score": 0.1}}].
"""

REASONING_PROMPT = """\
You are recommending recipes from a food knowledge graph.

Question: {question}
User constraints:
{constraints}

Reasoning paths explored so far:
{paths}

Candidate recipes found so far: {n_recipes}.
Are these paths adequate to recommend recipes that satisfy the constraints?
Return ONLY {{"sufficient": true}} or {{"sufficient": false}}.
"""


@dataclass
class _Path:
    """A partial reasoning path: a chain of entities with a cumulative score."""

    entities: tuple[Entity, ...]
    score: float

    @property
    def tail(self) -> Entity:
        return self.entities[-1]


class ToG(KBQASystem):
    name = "tog"

    def __init__(self, kg: HummusKG, llm: Optional[Any] = None,
                 options: Optional[KBQAOptions] = None):
        self.kg = kg
        self.llm = llm
        self.opt = options or KBQAOptions()

    def retrieve(self, qa_item: dict, k: int) -> list[str]:
        linked = link_query(qa_item, self.kg, query_field=self.opt.query_field)
        if not linked.topic_entities:
            return []
        scope = self._scope(linked, qa_item)

        beam = [_Path((e,), 1.0) for e in linked.topic_entities]
        # recipe key -> best path score that reached it
        recipe_scores: dict[object, float] = {}
        self._collect_recipes(beam, recipe_scores)

        for _ in range(self.opt.max_depth):
            beam = self._expand(beam, linked, scope)
            if not beam:
                break
            self._collect_recipes(beam, recipe_scores)
            if len(recipe_scores) >= k and self._sufficient(linked, beam, len(recipe_scores)):
                break

        if not recipe_scores:
            return []
        key_to_url = self._recipe_urls(recipe_scores)
        ordered = blend_order(recipe_scores, self.kg, alpha=self.opt.health_alpha,
                              key_to_url=key_to_url)
        return ordered[:k]

    # ── one ToG depth: relation prune → entity expand → entity prune → re-beam ──
    def _expand(self, beam: list[_Path], linked: LinkedQuery,
                scope: Optional[RecipeScope]) -> list[_Path]:
        candidates: list[_Path] = []
        for path in beam:
            relations = self.kg.relations_of(path.tail)
            for rel in self._prune_relations(path.tail, relations, linked):
                tails = self.kg.tail_entities(path.tail, rel, limit=self.opt.fanout)
                if scope is not None:
                    tails = [t for t in tails if scope.allows(t)]
                if not tails:
                    continue
                scores = self._score_entities(path.tail, rel, tails, linked)
                for t, s in zip(tails, scores):
                    candidates.append(_Path(path.entities + (t,), path.score * max(s, 1e-3)))
        candidates.sort(key=lambda p: p.score, reverse=True)
        return candidates[: self.opt.beam_width]

    def _prune_relations(self, entity: Entity, relations: list[Relation],
                         linked: LinkedQuery) -> list[Relation]:
        if not relations:
            return []
        listing = "\n".join(f"{i}. {r.describe()}" for i, r in enumerate(relations))
        text = llm_io.complete(self.llm, RELATION_PRUNE_PROMPT.format(
            question=linked.question, constraints=linked.constraint_text,
            entity=str(entity), label=entity.label, relations=listing))
        scores = llm_io.parse_scores(text, len(relations))
        if not scores:  # no-LLM / unparsable → keep all (uniform)
            return relations
        kept = sorted(scores, key=scores.get, reverse=True)[: self.opt.beam_width]
        return [relations[i] for i in kept if scores[i] > 0]

    def _score_entities(self, entity: Entity, rel: Relation, tails: list[Entity],
                        linked: LinkedQuery) -> list[float]:
        listing = "\n".join(f"{i}. {str(t)}" for i, t in enumerate(tails))
        text = llm_io.complete(self.llm, ENTITY_PRUNE_PROMPT.format(
            question=linked.question, constraints=linked.constraint_text,
            relation=rel.describe(), entity=str(entity), entities=listing))
        scores = llm_io.parse_scores(text, len(tails))
        return [scores.get(i, 0.5) for i in range(len(tails))]

    def _sufficient(self, linked: LinkedQuery, beam: list[_Path], n_recipes: int) -> bool:
        if self.llm is None:
            return True  # deterministic: stop as soon as we have ≥k recipes
        paths = "\n".join(" -> ".join(str(e) for e in p.entities) for p in beam[:5])
        text = llm_io.complete(self.llm, REASONING_PROMPT.format(
            question=linked.question, constraints=linked.constraint_text,
            paths=paths, n_recipes=n_recipes))
        value = llm_io.parse_json(text, {})
        return bool(isinstance(value, dict) and value.get("sufficient"))

    # ── helpers ──
    def _collect_recipes(self, beam: list[_Path], recipe_scores: dict[object, float]) -> None:
        for path in beam:
            if path.tail.label == RECIPE:
                key = path.tail.key
                if path.score > recipe_scores.get(key, float("-inf")):
                    recipe_scores[key] = path.score

    def _recipe_urls(self, recipe_scores: dict[object, float]) -> dict[object, str]:
        """Map recipe keys → ``recipe_url`` (kept keyed so ``blend_order`` can
        align evidence scores, unlike the order-only :meth:`HummusKG.recipe_urls`)."""
        rows = self.kg.client.run(
            f"MATCH (r:{RECIPE}) WHERE r.recipe_id IN $keys "
            f"RETURN r.recipe_id AS key, r.recipe_url AS url",
            keys=list(recipe_scores),
        )
        return {r["key"]: r["url"] for r in rows if r.get("url")}

    def _scope(self, linked: LinkedQuery, qa_item: dict) -> Optional[RecipeScope]:
        """Constraint-first traversal scope (§5.5) — only when explicitly enabled."""
        if not self.opt.constraint_first:
            return None
        gold = qa_item.get("_valid_recipe_keys")
        return self.kg.restrict_to_recipes(gold) if gold else None
