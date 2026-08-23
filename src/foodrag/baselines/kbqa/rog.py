"""RoG — Reasoning on Graphs (Luo et al., ICLR 2024), adapted to HUMMUS.

RoG is *plan-then-retrieve*: a **planner** (a fine-tuned LLM) first generates
faithful **relation paths** grounded in the KG schema, then a **retrieval-
reasoning** module grounds those paths on the graph to fetch reasoning paths and
reasons over them to answer. RoG's explicit reasoning-plan generation is the part
the protocol pairs with FoodRAG's explanation story.

This module implements both stages over the HUMMUS schema:

* :class:`RoGPlanner` emits relation paths over a fixed directed relation
  vocabulary (:data:`REL_VOCAB`). It prefers a fine-tuned checkpoint (the.2
  "light fine-tuning"; trained by ``kbqa/training/rog_planner_train.py``), falls
  back to prompting the base LLM with RoG's planning instruction, and finally to a
  small set of canonical schema paths so grounding always has something to run.
* :class:`RoG` grounds each plan from the topic entities (bounded one-hop hops,
.5 fan-out cap + hub pruning), aggregates plan agreement per recipe as the
  evidence score, and ranks health-aware.

The planner checkpoint and the reasoning LLM are injected; with neither present
the deterministic canonical-plan path keeps RoG runnable and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from foodrag.baselines.kbqa import llm_io
from foodrag.baselines.kbqa.base import KBQAOptions, KBQASystem, blend_order
from foodrag.baselines.kbqa.kg import (FOOD_ON, INGREDIENT, RECIPE, REL_HAS_INGREDIENT,
                                       REL_HAS_TAG, REL_IS_EQUIVALENT_TO, REL_SUB_CLASS_OF,
                                       TAG, Entity, HummusKG, Relation)
from foodrag.baselines.kbqa.linking import LinkedQuery, link_query

# Directed relation vocabulary the planner emits and the grounder executes. Each
# token is one schema edge in one direction; this is RoG's "relation space".
REL_VOCAB: dict[str, Relation] = {
    "recipe_has_ingredient": Relation(REL_HAS_INGREDIENT, "out", INGREDIENT),
    "ingredient_of_recipe": Relation(REL_HAS_INGREDIENT, "in", RECIPE),
    "recipe_has_tag": Relation(REL_HAS_TAG, "out", TAG),
    "tag_of_recipe": Relation(REL_HAS_TAG, "in", RECIPE),
    "ingredient_maps_to_class": Relation(REL_IS_EQUIVALENT_TO, "out", FOOD_ON),
    "class_has_ingredient": Relation(REL_IS_EQUIVALENT_TO, "in", INGREDIENT),
    "class_subclass_of": Relation(REL_SUB_CLASS_OF, "out", FOOD_ON),
    "class_has_subclass": Relation(REL_SUB_CLASS_OF, "in", FOOD_ON),
}
PATH_SEP = " <SEP> "

# Canonical schema paths per topic label → recipes; the deterministic fallback and
# the grounding sanity floor. Keys are the topic entity label.
CANONICAL_PLANS: dict[str, list[list[str]]] = {
    INGREDIENT: [
        ["ingredient_of_recipe"],
        ["ingredient_maps_to_class", "class_has_ingredient", "ingredient_of_recipe"],
    ],
    TAG: [
        ["tag_of_recipe"],
    ],
}

PLANNER_PROMPT = """\
You generate relation paths over a food knowledge graph to find recipes.

Available relations (use these tokens exactly):
{vocab}

A relation path is a sequence of relation tokens joined by " <SEP> " that starts
at a seed entity and ends at a Recipe. Generate up to {n} diverse, valid paths
that help answer the question under the constraints.

Question: {question}
User constraints:
{constraints}
Seed entity types: {seed_labels}

Return ONLY a JSON list of strings, each a path, e.g.
["ingredient_of_recipe", "ingredient_maps_to_class <SEP> class_has_ingredient <SEP> ingredient_of_recipe"].
"""


@dataclass
class RoGPlanner:
    """Generates relation-path plans; checkpoint → base-LLM → canonical fallback."""

    llm: Optional[Any] = None
    checkpoint: Optional[Any] = None  # a loaded HF planner (see load_checkpoint)
    max_plans: int = 4

    @staticmethod
    def load_checkpoint(path: str):
        """Load a fine-tuned planner (HF causal LM + tokenizer) from ``path``.

        Imported lazily so the dependency is only needed when a checkpoint is
        actually used; the trainer in ``kbqa/training`` writes this artifact.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        return {"tokenizer": tok, "model": model}

    def plan(self, linked: LinkedQuery) -> list[list[str]]:
        seed_labels = sorted({e.label for e in linked.topic_entities}) or [INGREDIENT, TAG]
        raw = self._generate(linked, seed_labels)
        plans = [self._tokenize(p) for p in raw]
        plans = [p for p in plans if p]
        if not plans:  # nothing valid → canonical schema paths for the seed labels
            for label in seed_labels:
                plans.extend(CANONICAL_PLANS.get(label, []))
        return _dedupe_paths(plans)[: self.max_plans]

    def _generate(self, linked: LinkedQuery, seed_labels: list[str]) -> list[str]:
        if self.checkpoint is not None:
            return self._generate_checkpoint(linked)
        if self.llm is not None:
            vocab = "\n".join(f"- {t}: {r.describe()}" for t, r in REL_VOCAB.items())
            text = llm_io.complete(self.llm, PLANNER_PROMPT.format(
                vocab=vocab, n=self.max_plans, question=linked.question,
                constraints=linked.constraint_text, seed_labels=", ".join(seed_labels)))
            value = llm_io.parse_json(text, [])
            return [str(p) for p in value] if isinstance(value, list) else []
        return []

    def _generate_checkpoint(self, linked: LinkedQuery) -> list[str]:
        tok, model = self.checkpoint["tokenizer"], self.checkpoint["model"]
        prompt = planner_instruction(linked)
        inputs = tok(prompt, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=64, num_return_sequences=self.max_plans,
                             num_beams=max(self.max_plans, 4), do_sample=False)
        decoded = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    @staticmethod
    def _tokenize(path: str) -> list[str]:
        tokens = [t.strip() for t in str(path).split("<SEP>")]
        tokens = [t for t in tokens if t in REL_VOCAB]
        return tokens


def planner_instruction(linked: LinkedQuery) -> str:
    """The training/inference instruction for the fine-tuned planner (kept in one
    place so the trainer and the runtime use byte-identical prompts)."""
    return (f"Generate relation paths to find recipes for this request.\n"
            f"Question: {linked.question}\nConstraints:\n{linked.constraint_text}\nPaths:")


class RoG(KBQASystem):
    name = "rog"

    def __init__(self, kg: HummusKG, llm: Optional[Any] = None,
                 planner: Optional[RoGPlanner] = None, options: Optional[KBQAOptions] = None):
        self.kg = kg
        self.llm = llm
        self.opt = options or KBQAOptions()
        self.planner = planner or RoGPlanner(llm=llm, max_plans=self.opt.beam_width)

    def retrieve(self, qa_item: dict, k: int) -> list[str]:
        linked = link_query(qa_item, self.kg, query_field=self.opt.query_field)
        if not linked.topic_entities:
            return []

        plans = self.planner.plan(linked)
        if not plans:
            return []

        # Evidence = plan agreement: each plan that grounds to a recipe adds its
        # (uniform) weight; recipes reached by more/earlier plans rank higher.
        recipe_scores: dict[object, float] = {}
        weight = 1.0
        for plan in plans:
            for key in self._ground(plan, linked.topic_entities):
                recipe_scores[key] = recipe_scores.get(key, 0.0) + weight
            weight *= 0.9  # mild decay so earlier (higher-ranked) plans count more

        if not recipe_scores:
            return []
        key_to_url = self._recipe_urls(recipe_scores)
        ordered = blend_order(recipe_scores, self.kg, alpha=self.opt.health_alpha,
                              key_to_url=key_to_url)
        return ordered[:k]

    def _ground(self, plan: list[str], seeds: list[Entity]) -> list[object]:
        """Execute a relation path from the seeds; return reached recipe keys.

        Bounded one-hop expansion per relation (.5 fan-out cap + hub pruning via
        :meth:`HummusKG.tail_entities`); the frontier is capped each hop so a long
        path cannot blow up combinatorially.
        """
        frontier = [e for e in seeds if _path_starts_at(plan, e.label)]
        for token in plan:
            rel = REL_VOCAB[token]
            nxt: list[Entity] = []
            seen = set()
            for ent in frontier:
                if ent.label != _source_label(rel):
                    continue
                for tail in self.kg.tail_entities(ent, rel, limit=self.opt.fanout):
                    sig = (tail.label, tail.key)
                    if sig not in seen:
                        seen.add(sig)
                        nxt.append(tail)
                if len(nxt) >= self.opt.fanout * self.opt.beam_width:
                    break
            frontier = nxt
            if not frontier:
                break
        return [e.key for e in frontier if e.label == RECIPE]

    def _recipe_urls(self, recipe_scores: dict[object, float]) -> dict[object, str]:
        rows = self.kg.client.run(
            f"MATCH (r:{RECIPE}) WHERE r.recipe_id IN $keys "
            f"RETURN r.recipe_id AS key, r.recipe_url AS url",
            keys=list(recipe_scores),
        )
        return {r["key"]: r["url"] for r in rows if r.get("url")}


# Each schema edge as (out_source_label, out_target_label).
EDGE_ENDPOINTS: dict[str, tuple[str, str]] = {
    REL_HAS_INGREDIENT: (RECIPE, INGREDIENT),
    REL_HAS_TAG: (RECIPE, TAG),
    REL_IS_EQUIVALENT_TO: (INGREDIENT, FOOD_ON),
    REL_SUB_CLASS_OF: (FOOD_ON, FOOD_ON),
}


def _source_label(rel: Relation) -> str:
    """The label a directed token is traversed *from*: the edge's source side for
    an ``out`` token, its target side for an ``in`` token."""
    src, tgt = EDGE_ENDPOINTS[rel.rel_type]
    return src if rel.direction == "out" else tgt


def _path_starts_at(plan: list[str], label: str) -> bool:
    return bool(plan) and _source_label(REL_VOCAB[plan[0]]) == label


def _dedupe_paths(plans: list[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for p in plans:
        key = tuple(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out
