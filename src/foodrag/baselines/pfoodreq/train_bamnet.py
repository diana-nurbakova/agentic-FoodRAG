"""Train the (ported) BAMnet on HUMMUS-derived KBQA — pFoodReq sub-task (a).

A thin driver mirroring the vendored ``BAMnet/src/train.py``, with three
differences that keep the port faithful while making it runnable here:

* imports **only** ``BAMnetAgent`` (the pfoodreq config's model) so the
  matchnn/bow/pbow agents — which we did not port — are never imported;
* puts the vendored ``BAMnet/src`` on ``sys.path`` so its ``core.*`` imports
  resolve from anywhere;
* derives ``vocab_size`` / ``num_ent_types`` / ``num_relations`` from the built
  ``*2id.json`` files and accepts config overrides (epochs, CPU, embedding file).

The BAMnet method itself is unchanged (see ``third_party/pfoodreq/NOTICE.md`` for
the handful of torch-2.x/ PyYAML compat patches). With ``pre_word2vec`` unset the
agent initialises word embeddings randomly, so a smoke train needs no GloVe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional


def _bamnet_src() -> Path:
    from foodrag.config.settings import repo_root
    return repo_root() / "third_party" / "pfoodreq" / "BAMnet" / "src"


def _ensure_on_path() -> None:
    src = str(_bamnet_src())
    if src not in sys.path:
        sys.path.insert(0, src)


def _base_config() -> dict:
    """Load the pfoodreq.yml defaults from the vendored config."""
    _ensure_on_path()
    from core.utils.utils import get_config
    return get_config(str(_bamnet_src() / "config" / "pfoodreq.yml"))


def train(vec_dir: str | Path, *, model_file: str | Path, pre_word2vec: str = "",
          num_epochs: int = 5, no_cuda: bool = True, sample: bool = False,
          overrides: Optional[dict] = None) -> dict:
    """Train BAMnet on the vectorised KBQA in ``vec_dir``; return summary stats."""
    _ensure_on_path()
    from core.bamnet.bamnet import BAMnetAgent
    from core.build_data.utils import vectorize_data
    from core.config import STOPWORDS
    from core.utils.utils import load_json

    vec_dir = Path(vec_dir)
    opt = _base_config()
    opt["data_dir"] = str(vec_dir)
    opt["no_cuda"] = no_cuda
    opt["pre_word2vec"] = pre_word2vec
    opt["num_epochs"] = num_epochs
    opt["model_file"] = str(model_file)
    # data-derived sizes (override the yml's FoodKG-era constants)
    opt["vocab_size"] = len(load_json(str(vec_dir / "vocab2id.json")))
    opt["num_ent_types"] = len(load_json(str(vec_dir / "entityType2id.json")))
    opt["num_relations"] = len(load_json(str(vec_dir / "relation2id.json")))
    if overrides:
        opt.update(overrides)

    vocab2id = load_json(str(vec_dir / "vocab2id.json"))
    train_vec = load_json(str(vec_dir / opt["train_data"]))
    valid_vec = load_json(str(vec_dir / opt["valid_data"]))
    if sample:
        train_vec = [x[:5] for x in train_vec]
        valid_vec = [x[:5] for x in valid_vec]

    (train_queries, train_raw_queries, train_query_mentions, train_query_marks,
     train_memories, _, train_gold_ans_inds, _, _, _) = train_vec
    train_queries, train_query_words, train_query_marks, train_query_lengths, train_memories, _ = vectorize_data(
        train_queries, train_query_mentions, train_query_marks, train_memories,
        max_query_size=opt["query_size"], max_ans_path_bow_size=opt["ans_path_bow_size"], vocab2id=vocab2id)

    (valid_queries, valid_raw_queries, valid_query_mentions, valid_query_marks,
     valid_memories, valid_cand_labels, valid_gold_ans_inds, valid_gold_ans_labels, _, _) = valid_vec
    valid_queries, valid_query_words, valid_query_marks, valid_query_lengths, valid_memories, _ = vectorize_data(
        valid_queries, valid_query_mentions, valid_query_marks, valid_memories,
        max_query_size=opt["query_size"], max_ans_path_bow_size=opt["ans_path_bow_size"], vocab2id=vocab2id)

    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    agent = BAMnetAgent(opt, STOPWORDS, vocab2id)
    agent.train(
        [train_memories, train_queries, train_query_words, train_raw_queries,
         train_query_mentions, train_query_marks, train_query_lengths], train_gold_ans_inds,
        [valid_memories, valid_queries, valid_query_words, valid_raw_queries,
         valid_query_mentions, valid_query_marks, valid_query_lengths],
        valid_gold_ans_inds, valid_cand_labels, valid_gold_ans_labels,
    )
    return {"model_file": str(model_file), "n_train": len(train_gold_ans_inds),
            "n_valid": len(valid_gold_ans_inds), "vocab_size": opt["vocab_size"],
            "num_ent_types": opt["num_ent_types"], "num_relations": opt["num_relations"]}


def evaluate(vec_dir: str | Path, *, model_file: str | Path, split: str = "valid",
             margin: Optional[float] = None, no_cuda: bool = True,
             overrides: Optional[dict] = None) -> dict:
    """Load a trained BAMnet and report pFoodReq's native metrics on ``split``.

    Returns avg recall/precision/F1 over the split (the anchor numbers reported
    for pFoodReq on its own HUMMUS-derived test set — Path A, §5.8). Uses the
    same predict→label→``calc_avg_f1`` path as the training loop's validation.
    """
    _ensure_on_path()
    from core.bamnet.bamnet import BAMnetAgent
    from core.build_data.utils import vectorize_data
    from core.config import STOPWORDS
    from core.utils.generic_utils import unique
    from core.utils.metrics import calc_avg_f1
    from core.utils.utils import load_json

    vec_dir = Path(vec_dir)
    opt = _base_config()
    opt["data_dir"] = str(vec_dir)
    opt["no_cuda"] = no_cuda
    opt["pre_word2vec"] = ""
    opt["model_file"] = str(model_file)
    opt["vocab_size"] = len(load_json(str(vec_dir / "vocab2id.json")))
    opt["num_ent_types"] = len(load_json(str(vec_dir / "entityType2id.json")))
    opt["num_relations"] = len(load_json(str(vec_dir / "relation2id.json")))
    if overrides:
        opt.update(overrides)
    margin = margin if margin is not None else opt["test_margin"][0]

    vocab2id = load_json(str(vec_dir / "vocab2id.json"))
    vec = load_json(str(vec_dir / f"{split}_vec.json"))
    (queries, raw_queries, query_mentions, query_marks, memories,
     cand_labels, gold_ans_inds, gold_ans_labels, _, _) = vec
    queries, query_words, query_marks, query_lengths, memories, _ = vectorize_data(
        queries, query_mentions, query_marks, memories,
        max_query_size=opt["query_size"], max_ans_path_bow_size=opt["ans_path_bow_size"], vocab2id=vocab2id)

    agent = BAMnetAgent(opt, STOPWORDS, vocab2id)  # __init__ loads model_file if it exists
    X = [memories, queries, query_words, raw_queries, query_mentions, query_marks, query_lengths]
    pred, _ = agent.predict(X, cand_labels, batch_size=1, margin=margin, silence=True)
    predictions = [unique([cand_labels[qid][x[0]] for x in each]) for qid, each in enumerate(pred)]
    count, recall, precision, f1 = calc_avg_f1(gold_ans_labels, predictions, verbose=False)
    return {"split": split, "n": count, "recall": recall, "precision": precision, "f1": f1,
            "predictions": predictions, "gold": gold_ans_labels}
