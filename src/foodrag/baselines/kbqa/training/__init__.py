"""Fine-tuning support for the trained KBQA baselines (RoG planner, TFS parser).

Two of the three systems are *trained*, not prompted: RoG's planner (light
fine-tuning) and TFS-KBQA's semantic parser (three-step fine-tuning). The QA set
supplies the training data — exactly the protocol's plan for these systems.

* :mod:`~foodrag.baselines.kbqa.training.data` turns a QA set into instruction →
  output JSONL: RoG planning targets (relation paths, canonical or KG-mined) and
  TFS parsing targets (deterministic gold Cypher from :func:`compile_cypher`).
* :mod:`~foodrag.baselines.kbqa.training.train` is a LoRA SFT trainer over that
  JSONL, writing a checkpoint the runtime loads via ``*.load_checkpoint``.

Both keep heavy deps (transformers/peft/datasets) behind lazy imports so importing
the baselines package never requires a training stack; the trainer is meant to run
on the project's Colab GPU (same pattern as the embedding notebook).
"""
