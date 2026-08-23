"""Unit tests for the external KBQA baselines.

ToG, RoG and TFS-KBQA are exercised against an in-memory fake graph + a scripted
(or absent) LLM, so the traversal, planning, parsing, ranking and dual-use
adapters are all covered without a live Neo4j or a generator. The deterministic
no-LLM fallbacks make every system runnable here, which is exactly what the tests
pin down.
"""
from __future__ import annotations

import pandas as pd
import pytest

from foodrag.baselines.kbqa.adapters import KBQABaseline, KBQARetriever
from foodrag.baselines.kbqa.base import KBQAOptions, blend_order
from foodrag.baselines.kbqa.kg import (INGREDIENT, RECIPE, TAG, Entity, HummusKG,
                                       Relation, REL_HAS_INGREDIENT, REL_HAS_TAG)
from foodrag.baselines.kbqa.linking import constraint_text, link_query
from foodrag.baselines.kbqa.rog import RoG
from foodrag.baselines.kbqa.tfs_kbqa import TFSKBQA, compile_cypher, render_inline
from foodrag.baselines.kbqa.tog import ToG
from foodrag.baselines.kbqa.training import data as kbqa_data

pytestmark = pytest.mark.unit


# ── in-memory fakes ──────────────────────────────────────────────────────────

class FakeKGClient:
    """Handles only the two direct ``client.run`` shapes the systems issue:
    the recipe-url lookup and the TFS query execution."""

    def __init__(self, recipes: dict, tfs_result: list[str] | None = None):
        self.recipes = recipes  # {recipe_id: url}
        self.tfs_result = tfs_result or []
        self.calls: list[tuple] = []

    def run(self, cypher: str, **params):
        self.calls.append((cypher, params))
        if "AS recipe_url" in cypher:                 # TFS execute (compiled/inline)
            return [{"recipe_url": u} for u in self.tfs_result]
        if "r.recipe_id AS key" in cypher:            # _recipe_urls
            keys = params.get("keys", [])
            return [{"key": k, "url": self.recipes[k]} for k in keys if k in self.recipes]
        return []


class FakeHummusKG(HummusKG):
    """Override the graph primitives with dict lookups (no Cypher interpreted)."""

    def __init__(self, *, ingredients, tags, recipes_url, attrs, neighbors,
                 tfs_result=None):
        super().__init__(FakeKGClient(recipes_url, tfs_result))
        self._ingredients = ingredients      # {name: Entity}
        self._tags = tags                    # {name: Entity}
        self._attrs = attrs                  # {recipe_id: {nutri_score, ...}}
        self._neighbors = neighbors          # {(label, key, signature): [Entity, ...]}

    def link_ingredient(self, name, limit=5):
        e = self._ingredients.get(name)
        return [e] if e else []

    def link_tag(self, name, limit=5):
        e = self._tags.get(name)
        return [e] if e else []

    def tail_entities(self, entity, relation, *, limit=None, prune_hubs=True):
        return list(self._neighbors.get((entity.label, entity.key, relation.signature), []))

    def recipe_attributes(self, recipe_keys, props=None):
        return {k: self._attrs[k] for k in recipe_keys if k in self._attrs}


@pytest.fixture
def kg() -> FakeHummusKG:
    spinach = Entity(INGREDIENT, "i_spin", "spinach")
    vegan = Entity(TAG, "t_vegan", "vegan")
    r1, r2 = Entity(RECIPE, 1, "Spinach Bowl"), Entity(RECIPE, 2, "Spinach Cake")
    return FakeHummusKG(
        ingredients={"spinach": spinach},
        tags={"vegan": vegan},
        recipes_url={1: "u1", 2: "u2"},
        attrs={1: {"nutri_score": 0.9, "fsa_score": 0.8, "average_rating": 4.0, "number_of_ratings": 10},
               2: {"nutri_score": 0.2, "fsa_score": 0.3, "average_rating": 5.0, "number_of_ratings": 99}},
        neighbors={
            # recipes that contain spinach (HAS_INGREDIENT incoming → Recipe)
            (INGREDIENT, "i_spin", Relation(REL_HAS_INGREDIENT, "in", RECIPE).signature): [r1, r2],
            # recipes with the vegan tag (HAS_TAG incoming → Recipe)
            (TAG, "t_vegan", Relation(REL_HAS_TAG, "in", RECIPE).signature): [r1],
        },
        tfs_result=["u1", "u2"],
    )


@pytest.fixture
def qa_item() -> dict:
    return {
        "question": "A healthy vegan dish with spinach and no peanuts.",
        "answer": ["u1"],
        "persona": {
            "tags": {"likes": ["vegan"], "dislikes": []},
            "ingredients": {"likes": ["spinach"], "dislikes": ["peanut"]},
            "relative_nutrients": {},
            "total_nutrients": {"sugars [g]": 10},
            "nutrition_scores": {"nutri_score": 0.5},
        },
    }


# ── linking + constraint rendering ───────────────────────────────────────────

def test_link_query_resolves_liked_topics(kg, qa_item):
    linked = link_query(qa_item, kg)
    labels = {(e.label, e.name) for e in linked.topic_entities}
    assert (INGREDIENT, "spinach") in labels and (TAG, "vegan") in labels
    assert "spinach" in linked.constraint_text and "peanut" in linked.constraint_text


def test_constraint_text_covers_families(qa_item):
    text = constraint_text(qa_item["persona"])
    assert "vegan" in text                      # liked tag
    assert "EXCLUDE" in text and "peanut" in text  # allergen
    assert "sugars" in text                     # total-nutrient bound
    assert "nutri_score" in text                # health-score minimum


# ── health-aware ranking ─────────────────────────────────────────────────────

def test_blend_order_prefers_health_on_ties(kg):
    # equal evidence → healthier recipe (key 1, nutri 0.9) ranks above key 2 (0.2)
    order = blend_order({1: 1.0, 2: 1.0}, kg, alpha=0.5, key_to_url={1: "u1", 2: "u2"})
    assert order == ["u1", "u2"]


# ── ToG ──────────────────────────────────────────────────────────────────────

def test_tog_reaches_recipes_without_llm(kg, qa_item):
    tog = ToG(kg, llm=None, options=KBQAOptions(max_depth=2, beam_width=5))
    urls = tog.retrieve(qa_item, k=10)
    assert set(urls) == {"u1", "u2"}
    assert urls[0] == "u1"  # health tie-break puts the healthier recipe first


# ── RoG ──────────────────────────────────────────────────────────────────────

def test_rog_canonical_plan_grounds_to_recipes(kg, qa_item):
    rog = RoG(kg, llm=None, options=KBQAOptions(beam_width=4))
    urls = rog.retrieve(qa_item, k=10)
    # spinach→recipes (both) and vegan→recipes (u1); u1 gets more plan agreement.
    assert set(urls) == {"u1", "u2"}
    assert urls[0] == "u1"


# ── TFS-KBQA ─────────────────────────────────────────────────────────────────

def test_compile_cypher_mirrors_constraints(qa_item):
    query = compile_cypher(qa_item["persona"], k=10)
    text = query.text
    assert "MATCH (r:Recipe)" in text
    assert "r.`nutri_score` >= $sc0" in text
    assert "(r.`sugars` / 100.0 * r.servingSize) <= $tot1" in text
    assert "EXISTS { MATCH (r)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $liked_tags }" in text
    assert "NOT EXISTS { MATCH (r)-[:HAS_INGREDIENT]->(i:Ingredient) WHERE i.name IN $disliked_ingr }" in text
    assert query.params["liked_ingr"] == ["spinach"]


def test_render_inline_substitutes_params(qa_item):
    inline = render_inline(compile_cypher(qa_item["persona"], k=10))
    assert "$" not in inline                      # all params inlined
    assert '["spinach"]' in inline and "LIMIT 10" in inline


def test_tfs_executes_and_falls_back(kg, qa_item):
    tfs = TFSKBQA(kg, llm=None)                    # no LLM → compiler fallback
    urls = tfs.retrieve(qa_item, k=10)
    assert urls == ["u1", "u2"]                    # from the fake execution


# ── dual-use adapters ────────────────────────────────────────────────────────

def test_kbqa_baseline_is_a_ladder_rung(kg, qa_item):
    base = KBQABaseline(TFSKBQA(kg, llm=None), rung="4", config_id="tfs_kbqa")
    assert base.config_id == "tfs_kbqa" and base.rung == "4"
    assert base.recommend(qa_item, 10) == ["u1", "u2"]


def test_kbqa_retriever_returns_recipe_frame(kg, qa_item):
    recipes_df = pd.DataFrame([{"recipe_url": "u1", "title": "A"}, {"recipe_url": "u2", "title": "B"}])
    retr = KBQARetriever(TFSKBQA(kg, llm=None), recipes_df, k=10).bind(qa_item)
    out = retr.retrieve(qa_item["question"])
    assert out["recipe_url"].tolist() == ["u1", "u2"]


# ── training data generation ─────────────────────────────────────────────────

def test_tfs_examples_have_runnable_targets(qa_item):
    examples = kbqa_data.tfs_examples([qa_item], k=10)
    assert len(examples) == 1
    ex = examples[0]
    assert "Cypher:" in ex["instruction"] and ex["output"].startswith("MATCH (r:Recipe)")
    assert "$" not in ex["output"]


def test_rog_examples_offline_use_canonical_paths(qa_item):
    examples = kbqa_data.rog_examples([qa_item], kg=None)
    assert len(examples) == 1
    assert "ingredient_of_recipe" in examples[0]["output"]


def test_row_to_tokens_infers_direction():
    # node0 (Recipe) -[HAS_INGREDIENT]-> node1 (Ingredient): forward → out token
    fwd = kbqa_data._row_to_tokens(["n0", "n1"], [REL_HAS_INGREDIENT], ["n0"])
    assert fwd == ["recipe_has_ingredient"]
    # reversed orientation → in token
    bwd = kbqa_data._row_to_tokens(["n0", "n1"], [REL_HAS_INGREDIENT], ["n1"])
    assert bwd == ["ingredient_of_recipe"]
