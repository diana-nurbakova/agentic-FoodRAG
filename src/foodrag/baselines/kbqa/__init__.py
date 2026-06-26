"""External KBQA baselines (Evaluation Protocol §5.8): ToG, RoG, TFS-KBQA.

Three modern KBQA paradigms adapted to the HUMMUS food graph, used in both roles
of the §5.7 dual-use design:

* **Use 1 — retrieval (EXP-1):** each system as a native retriever, scored on the
  retrieval/health/safety metrics. Wrap with :class:`KBQABaseline` (tag ``EXP-1``).
* **Use 2 — full-pipeline (EXP-7 rung 4):** each system's retrieval feeding
  FoodRAG's meal-plan assembly layer. Wrap with :class:`KBQARetriever`.

The systems share the HUMMUS KG layer (:mod:`kg`), entity linking (:mod:`linking`)
and health-aware ranking (:mod:`base`). ToG is training-free; RoG and TFS-KBQA are
trained (planner / parser fine-tuned by :mod:`training`), with base-LLM and
deterministic fallbacks so they are runnable and testable without a checkpoint.

pFoodReq/BAMnet (the trained prior-work anchor, §5.8) is intentionally out of this
package — it is handled separately.
"""
from foodrag.baselines.kbqa.adapters import KBQABaseline, KBQARetriever
from foodrag.baselines.kbqa.agentic import build_heuristic_workflow_builder
from foodrag.baselines.kbqa.base import KBQAOptions, KBQASystem
from foodrag.baselines.kbqa.factory import (KBQA_NAMES, build_agentic_pipeline, build_baseline,
                                            build_full_pipeline, build_kg, build_retriever,
                                            build_system)
from foodrag.baselines.kbqa.kg import HummusKG, KGClient, Neo4jKGClient
from foodrag.baselines.kbqa.rog import RoG, RoGPlanner
from foodrag.baselines.kbqa.tfs_kbqa import TFSKBQA, TFSParser, compile_cypher
from foodrag.baselines.kbqa.tog import ToG

__all__ = [
    "KBQASystem", "KBQAOptions",
    "ToG", "RoG", "RoGPlanner", "TFSKBQA", "TFSParser", "compile_cypher",
    "HummusKG", "KGClient", "Neo4jKGClient",
    "KBQABaseline", "KBQARetriever",
    "build_kg", "build_system", "build_baseline", "build_retriever",
    "build_full_pipeline", "build_agentic_pipeline", "build_heuristic_workflow_builder",
    "KBQA_NAMES",
]
