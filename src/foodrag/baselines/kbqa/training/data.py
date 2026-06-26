"""Generate fine-tuning data for the trained KBQA baselines from a QA set (§5.8).

Each emitted record is ``{"instruction": <prompt>, "output": <target>}`` (JSONL),
the shape the LoRA trainer consumes. The prompts are the byte-identical
``planner_instruction`` / ``parser_instruction`` used at inference, so training and
runtime never drift.

* **TFS parser** — target is the deterministic gold Cypher
  (:func:`~foodrag.baselines.kbqa.tfs_kbqa.compile_cypher` →
  :func:`~foodrag.baselines.kbqa.tfs_kbqa.render_inline`). It selects exactly the
  persona's valid recipes, so these are clean, verifiable supervision pairs.
* **RoG planner** — target is the set of relation paths that connect the topic
  entities to the gold answer recipes. By default these are the canonical schema
  paths for the seed labels; with a KG, :func:`mine_relation_paths` recovers the
  *actual* shortest relation paths to a sample of gold recipes (RoG's own target
  construction), which captures longer substitution paths.

CLI::

    python -m foodrag.baselines.kbqa.training.data --task tfs --qa <qa.pkl> --out data/kbqa/tfs.jsonl
    python -m foodrag.baselines.kbqa.training.data --task rog --qa <qa.pkl> --out data/kbqa/rog.jsonl [--mine-paths]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

from foodrag.baselines.kbqa.kg import RECIPE, Entity, HummusKG
from foodrag.baselines.kbqa.linking import LinkedQuery, link_query
from foodrag.baselines.kbqa.rog import (CANONICAL_PLANS, PATH_SEP, REL_VOCAB,
                                        planner_instruction)
from foodrag.baselines.kbqa.tfs_kbqa import compile_cypher, parser_instruction, render_inline

# (rel_type, direction) → planner token, the inverse of REL_VOCAB.
TOKEN_BY_EDGE = {(r.rel_type, r.direction): tok for tok, r in REL_VOCAB.items()}


# ── TFS parser data ──────────────────────────────────────────────────────────

def tfs_examples(qa_set: Iterable[dict], k: int = 10, *, query_field: str = "question") -> list[dict]:
    """One instruction→Cypher pair per QA item (deterministic gold target)."""
    examples = []
    for item in qa_set:
        persona = item.get("persona", {})
        question = item.get(query_field) or item.get("question", "")
        target = render_inline(compile_cypher(persona, k))
        examples.append({"instruction": parser_instruction(question, persona, k), "output": target})
    return examples


# ── RoG planner data ─────────────────────────────────────────────────────────

def rog_examples(qa_set: Iterable[dict], kg: Optional[HummusKG] = None, *,
                 query_field: str = "question", max_len: int = 4,
                 sample_gold: int = 20) -> list[dict]:
    """One instruction→paths pair per QA item.

    With a ``kg`` the targets are the actual shortest relation paths from the
    topic entities to a sample of gold recipes (RoG construction); without one,
    the canonical schema paths for the present seed labels.
    """
    examples = []
    for item in qa_set:
        linked = link_query(item, kg, query_field=query_field) if kg is not None \
            else _link_offline(item, query_field)
        paths = _target_paths(linked, item, kg, max_len, sample_gold)
        if not paths:
            continue
        output = "\n".join(PATH_SEP.join(p) for p in paths)
        examples.append({"instruction": planner_instruction(linked), "output": output})
    return examples


def _target_paths(linked: LinkedQuery, item: dict, kg: Optional[HummusKG],
                  max_len: int, sample_gold: int) -> list[list[str]]:
    if kg is not None and linked.topic_entities:
        gold = [g for g in (item.get("answer") or [])][:sample_gold]
        mined = []
        for topic in linked.topic_entities:
            mined.extend(mine_relation_paths(kg, topic, gold, max_len=max_len))
        if mined:
            return _dedupe(mined)
    seed_labels = {e.label for e in linked.topic_entities} or set(CANONICAL_PLANS)
    canonical = [p for lbl in seed_labels for p in CANONICAL_PLANS.get(lbl, [])]
    return _dedupe(canonical)


def mine_relation_paths(kg: HummusKG, topic: Entity, gold_urls: list[str], *,
                        max_len: int = 4, limit: int = 8) -> list[list[str]]:
    """Shortest relation paths from ``topic`` to gold recipes, as token sequences.

    Direction is recovered from the physical edge orientation (``startNode``), so
    it is label-independent and handles the symmetric ``SUB_CLASS_OF`` edge. Paths
    longer than ``max_len`` are excluded (§5.5 capped traversal).
    """
    if not gold_urls:
        return []
    rels = "|".join({r.rel_type for r in REL_VOCAB.values()})
    cypher = (
        f"MATCH (s:`{topic.label}` {{`{_key_prop(topic.label)}`: $key}}) "
        f"MATCH (r:{RECIPE}) WHERE r.recipe_url IN $gold "
        f"MATCH p = shortestPath((s)-[:{rels}*1..{max_len}]-(r)) "
        f"RETURN [n IN nodes(p) | elementId(n)] AS node_ids, "
        f"[rel IN relationships(p) | type(rel)] AS types, "
        f"[rel IN relationships(p) | elementId(startNode(rel))] AS starts LIMIT $limit"
    )
    try:
        rows = kg.client.run(cypher, key=topic.key, gold=list(gold_urls), limit=limit)
    except Exception:
        return []

    paths = []
    for row in rows:
        tokens = _row_to_tokens(row["node_ids"], row["types"], row["starts"])
        if tokens:
            paths.append(tokens)
    return _dedupe(paths)


def _row_to_tokens(node_ids: list, types: list, starts: list) -> list[str]:
    tokens = []
    for i, rel_type in enumerate(types):
        forward = starts[i] == node_ids[i]  # edge oriented node_i -> node_{i+1}
        direction = "out" if forward else "in"
        token = TOKEN_BY_EDGE.get((rel_type, direction))
        if token is None:
            return []
        tokens.append(token)
    return tokens


def _key_prop(label: str) -> str:
    from foodrag.baselines.kbqa.kg import KEY_PROP

    return KEY_PROP[label]


def _link_offline(item: dict, query_field: str) -> LinkedQuery:
    """Build a LinkedQuery without a KG (topic *labels* only, for canonical paths)."""
    from foodrag.baselines.kbqa.kg import INGREDIENT, TAG
    from foodrag.baselines.kbqa.linking import constraint_text

    persona = item.get("persona", {})
    topics = []
    for nm in persona.get("ingredients", {}).get("likes", []):
        topics.append(Entity(INGREDIENT, nm, nm))
    for nm in persona.get("tags", {}).get("likes", []):
        topics.append(Entity(TAG, nm, nm))
    return LinkedQuery(
        question=item.get(query_field) or item.get("question", ""),
        persona=persona, topic_entities=topics, constraint_text=constraint_text(persona))


def _dedupe(paths: list[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for p in paths:
        t = tuple(p)
        if t and t not in seen:
            seen.add(t)
            out.append(p)
    return out


# ── IO + CLI ─────────────────────────────────────────────────────────────────

def write_jsonl(examples: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out_path


def _load_qa(path: str | Path) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate KBQA fine-tuning data from a QA set.")
    p.add_argument("--task", choices=["tfs", "rog"], required=True)
    p.add_argument("--qa", required=True, help="path to a QA-set pickle")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mine-paths", action="store_true",
                   help="RoG: recover real shortest paths from the KG (needs Neo4j)")
    args = p.parse_args(argv)

    qa_set = _load_qa(args.qa)
    if args.task == "tfs":
        examples = tfs_examples(qa_set, k=args.k)
    else:
        kg = None
        if args.mine_paths:
            from foodrag.baselines.kbqa.kg import Neo4jKGClient

            kg = HummusKG(Neo4jKGClient.from_env())
        examples = rog_examples(qa_set, kg=kg)
    path = write_jsonl(examples, args.out)
    print(f"[{args.task}] wrote {len(examples)} examples -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
