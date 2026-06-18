"""Orchestration tests for the pipeline stage modules.

These exercise each stage's run() wiring (which dependency it calls, with which
args) by injecting fakes / monkeypatching the heavy calls -- no data, models,
or stores required.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _FakeLoader:
    def __init__(self):
        self.validated = False

    def require(self):
        self.validated = True

    def hummus_kwargs(self):
        return {"data_location": "D", "graph_location": "G", "additional_location": "A"}


def test_init_hummus_orchestration(monkeypatch):
    from foodrag.pipeline import init_hummus
    from foodrag.utils import load_util

    seen = {}

    def fake_get_hummus(dl, gl, al, **kw):
        seen["locations"] = (dl, gl, al)
        seen["kw"] = kw
        return {"recipes_df": [1, 2, 3]}

    monkeypatch.setattr(load_util, "get_hummus", fake_get_hummus)
    loader = _FakeLoader()
    out = init_hummus.run(loader=loader, k_user=5, include_usda=False, annotate_meal_type=False)

    assert loader.validated
    assert seen["locations"] == ("D", "G", "A")
    assert seen["kw"]["k_user"] == 5
    assert seen["kw"]["include_usda"] is False
    assert seen["kw"]["do_annotate_meal_type"] is False
    assert out["recipes_df"] == [1, 2, 3]


def test_embed_orchestration(monkeypatch):
    from foodrag.food_rag.utils import embedding_helper, hummus_docify
    from foodrag.pipeline import embed

    monkeypatch.setattr(hummus_docify, "recipes_to_docs", lambda df: ["d1", "d2", "d3"])
    calls = []
    monkeypatch.setattr(
        embedding_helper, "compute_embeddings",
        lambda docs, model, **kw: calls.append((kw.get("path"), len(docs))) or {},
    )
    out = embed.run(doc_types=["recipe"], sample=2, embed_model=object(), hummus_data={"recipes_df": "DF"})

    assert out["counts"] == {"recipe": 2}
    assert calls[0][1] == 2  # sampled down to 2
    assert "recipe_documents" in calls[0][0]


def test_vector_store_orchestration(monkeypatch):
    import llama_index.core as lic
    from foodrag.food_rag.utils import embedding_helper, hummus_docify
    from foodrag.pipeline import vector_store

    monkeypatch.setattr(hummus_docify, "recipes_to_docs", lambda df: ["d"])
    monkeypatch.setattr(embedding_helper, "load_existing_batches", lambda path: ({"b0": ["n1", "n2"]}, ["b0"]))
    monkeypatch.setattr(embedding_helper, "check_batches", lambda a, b: None)

    class FakeIndex:
        def build_index_from_nodes(self, nodes):
            self.nodes = nodes

    monkeypatch.setattr(
        lic.VectorStoreIndex, "from_vector_store", classmethod(lambda cls, vs, **kw: FakeIndex())
    )

    seen = {}

    class FakeProvider:
        def connect(self, table_name, embed_dim):
            seen["table"] = table_name
            seen["dim"] = embed_dim
            return "VS"

    out = vector_store.run(
        hummus_data={"recipes_df": "DF"}, provider=FakeProvider(), embed_dim=1024, embed_model=object()
    )
    assert out == {"table": "hummus_recipes", "embed_dim": 1024, "n_nodes": 2}
    assert seen == {"table": "hummus_recipes", "dim": 1024}


def test_graph_stores_orchestration(monkeypatch):
    from foodrag.hummus_transformation import transform_kg
    from foodrag.pipeline import graph_stores

    events = []

    class FakeTransformer:
        def __init__(self, mode):
            events.append(("init", mode))

        def transform(self, data, limit_size=None):
            events.append(("transform", limit_size))

        def save(self, data_path):
            events.append(("save", data_path))

    monkeypatch.setattr(transform_kg, "Neo4JTransformer", FakeTransformer)

    class FakeMgr:
        def delete_all(self):
            events.append(("delete",))

        def insert_data(self, data_path, mode, batch_size, verbose, use_absolute_paths=False):
            events.append(("insert", mode))

    out = graph_stores.run(hummus_data={"x": 1}, manager=FakeMgr(), mode="all", limit_size=10)
    assert ("transform", 10) in events
    assert ("delete",) in events
    assert ("insert", "all") in events
    assert out["inserted"] is True


def test_graph_stores_no_insert(monkeypatch):
    from foodrag.hummus_transformation import transform_kg
    from foodrag.pipeline import graph_stores

    class FakeTransformer:
        def __init__(self, mode):
            pass

        def transform(self, data, limit_size=None):
            pass

        def save(self, data_path):
            pass

    monkeypatch.setattr(transform_kg, "Neo4JTransformer", FakeTransformer)
    out = graph_stores.run(hummus_data={}, insert=False)
    assert out["inserted"] is False


def test_gen_qa_orchestration(monkeypatch):
    from foodrag.persona import qa_generator
    from foodrag.pipeline import gen_qa
    from foodrag.utils import load_util

    monkeypatch.setattr(load_util, "get_config", lambda p: {})
    captured = []
    monkeypatch.setattr(
        qa_generator, "generate_question_answer_set",
        lambda data, qtype, *a, **k: captured.append(qtype) or [1, 2, 3],
    )
    out = gen_qa.run(qa_name="t", question_types=["recipe"], llm=object(), hummus_data={}, food_on=object())
    assert out["counts"] == {"recipe": 3}
    assert out["qa_name"] == "t"


def test_run_agents_delegates(monkeypatch):
    from foodrag.food_rag.agents import workflow_runner
    from foodrag.pipeline import run_agents

    monkeypatch.setattr(workflow_runner, "run_all_questions", lambda q, w, dp, sd, **k: {"q1": "answer"})
    out = run_agents.run(["q1"], agent_workflow=object(), data_path="d", sub_dir_name="s")
    assert out == {"q1": "answer"}


def test_run_agents_requires_workflow():
    from foodrag.pipeline import run_agents

    with pytest.raises(ValueError):
        run_agents.run(["q"], agent_workflow=None)


def test_evaluate_delegates():
    from foodrag.pipeline import evaluate

    class FakeEval:
        def evaluate_dataset(self, qa, dp, rn, batch_size, show_progress):
            return {"done": True, "run": rn}

    out = evaluate.run({"q": 1}, evaluator=FakeEval(), data_path="d", run_name="r")
    assert out == {"done": True, "run": "r"}


def test_evaluate_requires_retriever_or_evaluator():
    from foodrag.pipeline import evaluate

    with pytest.raises(ValueError):
        evaluate.run({"q": 1})


def test_cli_routes_to_stage(monkeypatch):
    from foodrag import cli
    import foodrag.pipeline.init_hummus as ih

    monkeypatch.setattr(ih, "main", lambda argv: 42)
    assert cli.main(["init-hummus", "--data-path", "x"]) == 42
