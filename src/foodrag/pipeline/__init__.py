"""Reproducible pipeline stages as importable, testable modules.

Each stage exposes a ``run(...)`` function (dependency-injectable for testing)
and a ``main(argv)`` CLI entry guarded by ``if __name__ == '__main__'`` so the
module can be imported without executing the pipeline. The ``foodrag`` CLI and
the thin wrappers in ``scripts/`` both delegate here.

Stages (in order):
    init_hummus     -> load + clean the HUMMUS dataset
    embed           -> compute recipe/user/... embeddings
    vector_store    -> load embeddings into pgvector
    graph_stores    -> build the Neo4j (and optional Nebula) KG
    gen_qa          -> generate the QA evaluation set
    run_agents      -> run the agentic meal-plan pipeline
    evaluate        -> evaluate retrievers against the QA set
"""
