"""HUMMUS knowledge-graph access layer for the external KBQA baselines.

The three KBQA systems (ToG, RoG, TFS-KBQA) all traverse the same HUMMUS graph.
This module is the single place that knows the graph's *shape* — its node labels,
relationship types and the flat, space-truncated property names produced by the
``LOAD CSV`` import (``neo4J_helper.clean_column_name`` keeps the text up to the
first space, so ``servingSize [g]`` is stored as ``servingSize`` and queried as
``r.servingSize``). Everything above this file reasons in terms of
:class:`Entity` / :class:`Relation`, never raw Cypher.

Three faithfulness-relevant decisions live here, all from:

* **Capped depth** — the recipe→ingredient→FoodOn-class→ingredient→recipe pattern
  (length 5) is the unit of traversal; callers pass a small ``max_depth`` and the
  one-hop primitives never expand neighbours-of-neighbours implicitly.
* **Hub pruning** — high-degree FoodOn hub classes (``vegetable``, ``salt``, …)
  drive the combinatorial fan-out the thesis hit. :meth:`HummusKG.tail_entities`
  drops tail nodes whose degree exceeds ``hub_degree_max`` so a single hub cannot
  explode the beam. This prunes the *traversal*, not the *data*, so it
  does not compromise cross-retriever comparability.
* **Constraint-first is opt-in** — prefiltering to the persona's valid set (which
  equals the gold set on satisfiable personas,.7 caveat) would trivially
  saturate retrieval metrics, so the KBQA systems traverse *natively* by default;
  constraint-first is exposed as a tractability mode the adapters can switch on.

The graph itself is reached through a tiny :class:`KGClient` protocol so tests can
inject an in-memory fake — no Neo4j required to exercise the traversal logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, runtime_checkable

# ── HUMMUS schema (transform_kg.py) ──────────────────────────────────────────
RECIPE = "Recipe"
INGREDIENT = "Ingredient"
FOOD_ON = "Food_on"
TAG = "Tag"

REL_HAS_INGREDIENT = "HAS_INGREDIENT"      # (Recipe)-->(Ingredient)
REL_IS_EQUIVALENT_TO = "IS_EQUIVALENT_TO"  # (Ingredient)-->(Food_on)
REL_SUB_CLASS_OF = "SUB_CLASS_OF"          # (Food_on)-->(Food_on)
REL_HAS_TAG = "HAS_TAG"                    # (Recipe)-->(Tag)

# id / name properties (flat, space-truncated by clean_column_name)
KEY_PROP = {RECIPE: "recipe_id", INGREDIENT: "ingredient_id", TAG: "tag_id", FOOD_ON: "id"}
NAME_PROP = {RECIPE: "title", INGREDIENT: "name", TAG: "name", FOOD_ON: "label"}
RECIPE_URL = "recipe_url"

# Health/nutrition recipe properties the KBQA prompts and rankers reason over.
RECIPE_SCORE_PROPS = ("nutri_score", "fsa_score", "who_score")
RECIPE_RANK_PROPS = ("average_rating", "number_of_ratings")

DEFAULT_HUB_DEGREE_MAX = 2_000   # a Food_on class above this is treated as a hub
DEFAULT_FANOUT = 50              # tail nodes kept per (entity, relation) expansion


@runtime_checkable
class KGClient(Protocol):
    """Minimal parametrised Cypher runner (Neo4jManager.query takes no params)."""

    def run(self, cypher: str, **params) -> list[dict]:
        ...


class Neo4jKGClient:
    """:class:`KGClient` over a ``neo4j`` driver, with parameter binding.

    Construct from the project's store config (:func:`from_env`) or wrap an
    existing :class:`~foodrag.food_rag.utils.neo4J_helper.Neo4jManager` driver
    (:func:`from_manager`) to share its connection pool.
    """

    def __init__(self, driver, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    @classmethod
    def from_env(cls, database: Optional[str] = None) -> "Neo4jKGClient":
        from neo4j import GraphDatabase

        from foodrag.config.stores import neo4j_config

        cfg = neo4j_config(database)
        driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]),
                                      database=cfg["database"])
        return cls(driver, cfg["database"])

    @classmethod
    def from_manager(cls, manager, database: str = "neo4j") -> "Neo4jKGClient":
        return cls(manager.driver, database)

    def run(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, **params).data()

    def close(self) -> None:
        self._driver.close()


# ── graph value objects ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """A KG node: its ``label``, its stable ``key`` (the id-property value) and a
    human-readable ``name`` used in LLM prompts."""

    label: str
    key: object
    name: str = ""

    def __str__(self) -> str:
        return self.name or f"{self.label}#{self.key}"


@dataclass(frozen=True)
class Relation:
    """A directed edge type incident to an entity.

    ``direction`` is ``"out"`` when the entity is the source of ``rel_type`` and
    ``"in"`` when it is the target. ``tail_label`` is the label reached by
    following it. :attr:`describe` is the surface form the LLM prunes over.
    """

    rel_type: str
    direction: str
    tail_label: str

    @property
    def signature(self) -> str:
        arrow = "->" if self.direction == "out" else "<-"
        return f"{self.rel_type}{arrow}{self.tail_label}"

    def describe(self) -> str:
        verb = {
            (REL_HAS_INGREDIENT, "out"): "has ingredient",
            (REL_HAS_INGREDIENT, "in"): "is an ingredient of recipe",
            (REL_HAS_TAG, "out"): "has tag",
            (REL_HAS_TAG, "in"): "is a tag of recipe",
            (REL_IS_EQUIVALENT_TO, "out"): "maps to FoodOn class",
            (REL_IS_EQUIVALENT_TO, "in"): "is an ingredient in FoodOn class",
            (REL_SUB_CLASS_OF, "out"): "is a subclass of",
            (REL_SUB_CLASS_OF, "in"): "has subclass",
        }.get((self.rel_type, self.direction), self.signature)
        return f"{verb} ({self.tail_label})"


# The structural relations reachable from each label (the HUMMUS schema is fixed,
# so the relation-exploration step is a schema lookup intersected with what the
# node actually has — no expensive distinct-relationship scan per node).
_SCHEMA_RELATIONS: dict[str, tuple[Relation, ...]] = {
    RECIPE: (
        Relation(REL_HAS_INGREDIENT, "out", INGREDIENT),
        Relation(REL_HAS_TAG, "out", TAG),
    ),
    INGREDIENT: (
        Relation(REL_HAS_INGREDIENT, "in", RECIPE),
        Relation(REL_IS_EQUIVALENT_TO, "out", FOOD_ON),
    ),
    TAG: (
        Relation(REL_HAS_TAG, "in", RECIPE),
    ),
    FOOD_ON: (
        Relation(REL_IS_EQUIVALENT_TO, "in", INGREDIENT),
        Relation(REL_SUB_CLASS_OF, "out", FOOD_ON),
        Relation(REL_SUB_CLASS_OF, "in", FOOD_ON),
    ),
}


def _back(name: str) -> str:
    return f"`{name}`"


class HummusKG:
    """High-level, schema-aware view of the HUMMUS graph for the KBQA baselines.

    Wraps a :class:`KGClient` and exposes the handful of primitives the KBQA
    algorithms need — entity linking, relation exploration, bounded one-hop
    expansion, recipe-url resolution and capped-depth path grounding — with the
.5 traversal guards (fan-out cap, hub pruning) applied consistently.
    """

    def __init__(self, client: KGClient, *, hub_degree_max: int = DEFAULT_HUB_DEGREE_MAX,
                 fanout: int = DEFAULT_FANOUT):
        self.client = client
        self.hub_degree_max = hub_degree_max
        self.fanout = fanout

    # — entity linking —
    def link_ingredient(self, name: str, limit: int = 5) -> list[Entity]:
        return self._link(INGREDIENT, name, limit)

    def link_tag(self, name: str, limit: int = 5) -> list[Entity]:
        return self._link(TAG, name, limit)

    def link_foodon(self, label: str, limit: int = 5) -> list[Entity]:
        return self._link(FOOD_ON, label, limit)

    def _link(self, label: str, surface: str, limit: int) -> list[Entity]:
        """Resolve a surface form to nodes of ``label`` by case-insensitive name
        match (exact first, then containment), capped at ``limit``."""
        key, name = KEY_PROP[label], NAME_PROP[label]
        rows = self.client.run(
            f"MATCH (n:{_back(label)}) "
            f"WHERE toLower(n.{_back(name)}) = toLower($s) "
            f"RETURN n.{_back(key)} AS key, n.{_back(name)} AS name LIMIT $lim",
            s=surface, lim=limit,
        )
        if not rows:
            rows = self.client.run(
                f"MATCH (n:{_back(label)}) "
                f"WHERE toLower(n.{_back(name)}) CONTAINS toLower($s) "
                f"RETURN n.{_back(key)} AS key, n.{_back(name)} AS name LIMIT $lim",
                s=surface, lim=limit,
            )
        return [Entity(label, r["key"], r.get("name") or surface) for r in rows]

    # — relation exploration (ToG step 1) —
    def relations_of(self, entity: Entity) -> list[Relation]:
        """Candidate relations to explore from ``entity`` (schema-driven,.5
        capped: only the one-hop relation types of the node's label)."""
        return list(_SCHEMA_RELATIONS.get(entity.label, ()))

    # — bounded one-hop expansion (ToG step 2 / RoG grounding) —
    def tail_entities(self, entity: Entity, relation: Relation, *,
                      limit: Optional[int] = None, prune_hubs: bool = True) -> list[Entity]:
        """Entities reached from ``entity`` via ``relation``, capped at ``limit``.

        Hub tails (Food_on classes whose degree exceeds ``hub_degree_max``) are
        dropped when ``prune_hubs`` so a high-degree class cannot blow up the beam
. The cap is applied in Cypher (``LIMIT``) so the fan-out never
        materialises server-side.
        """
        limit = self.fanout if limit is None else limit
        key, name = KEY_PROP[entity.label], NAME_PROP[entity.label]
        tkey, tname = KEY_PROP[relation.tail_label], NAME_PROP[relation.tail_label]
        pattern = (f"(n)-[:{relation.rel_type}]->(t:{_back(relation.tail_label)})"
                   if relation.direction == "out" else
                   f"(n)<-[:{relation.rel_type}]-(t:{_back(relation.tail_label)})")
        hub_clause = ""
        if prune_hubs and relation.tail_label == FOOD_ON:
            hub_clause = "AND COUNT { (t)--() } <= $hub "
        rows = self.client.run(
            f"MATCH (n:{_back(entity.label)} {{{_back(key)}: $k}}) "
            f"MATCH {pattern} "
            f"WHERE t.{_back(tname)} IS NOT NULL {hub_clause}"
            f"RETURN DISTINCT t.{_back(tkey)} AS key, t.{_back(tname)} AS name LIMIT $lim",
            k=entity.key, hub=self.hub_degree_max, lim=limit,
        )
        return [Entity(relation.tail_label, r["key"], r.get("name") or "") for r in rows]

    # — recipe resolution —
    def recipe_urls(self, recipe_entities: Iterable[Entity]) -> list[str]:
        """Map Recipe entities to their ``recipe_url`` (order-preserving, deduped)."""
        keys = [e.key for e in recipe_entities if e.label == RECIPE]
        if not keys:
            return []
        rows = self.client.run(
            f"MATCH (r:{RECIPE}) WHERE r.recipe_id IN $keys "
            f"RETURN r.recipe_id AS key, r.{RECIPE_URL} AS url",
            keys=list(keys),
        )
        by_key = {r["key"]: r["url"] for r in rows}
        out, seen = [], set()
        for k in keys:
            url = by_key.get(k)
            if url and url not in seen:
                seen.add(url)
                out.append(url)
        return out

    def recipe_attributes(self, recipe_keys: Iterable[object],
                          props: Iterable[str] = RECIPE_SCORE_PROPS + RECIPE_RANK_PROPS) -> dict[object, dict]:
        """Fetch ``props`` for a set of recipe keys — used by the health/rating
        rankers and to inject health evidence into the KBQA reasoning prompts."""
        keys = list(recipe_keys)
        if not keys:
            return {}
        proj = ", ".join(f"r.{_back(p)} AS {_back(p)}" for p in props)
        rows = self.client.run(
            f"MATCH (r:{RECIPE}) WHERE r.recipe_id IN $keys "
            f"RETURN r.recipe_id AS key, {proj}",
            keys=keys,
        )
        return {r.pop("key"): r for r in rows}

    # — constraint-first tractability mode —
    def restrict_to_recipes(self, recipe_keys: Iterable[object]) -> "RecipeScope":
        """Return a scope object the algorithms can intersect recipe tails with,
        so traversal stays inside an already-filtered valid set."""
        return RecipeScope(frozenset(recipe_keys))


@dataclass(frozen=True)
class RecipeScope:
    """An optional whitelist of recipe keys for constraint-first traversal."""

    keys: frozenset = field(default_factory=frozenset)

    def allows(self, entity: Entity) -> bool:
        return entity.label != RECIPE or entity.key in self.keys
