"""LoRA SFT trainer for the trained KBQA baselines (RoG planner, TFS parser).

Reads the ``{"instruction", "output"}`` JSONL produced by
:mod:`~foodrag.baselines.kbqa.training.data` and fine-tunes a causal LM with LoRA,
writing a checkpoint that the runtime loads via ``RoGPlanner.load_checkpoint`` /
``TFSParser.load_checkpoint``. Designed to run on the project's Colab GPU (bf16 +
gradient checkpointing + small LoRA rank), the same execution pattern as the
embedding notebook. All heavy deps are imported lazily.

Faithfulness to the source methods:
* **RoG** uses *light* fine-tuning of the planner — a single LoRA pass
  (:func:`train`, ``--task rog``).
* **TFS-KBQA** uses a *three-step* fine-tuning strategy — :func:`train_three_step`
  runs three sequential LoRA passes (parse → execution-filtered → full), each
  resuming the previous adapter, which is the curriculum the paper prescribes;
  document any deviation from the original step definitions in the run manifest.

CLI::

    python -m foodrag.baselines.kbqa.training.train --task rog \
        --data data/kbqa/rog.jsonl --base-model meta-llama/Llama-3.2-3B-Instruct \
        --out checkpoints/rog_planner

    python -m foodrag.baselines.kbqa.training.train --task tfs --three-step \
        --data data/kbqa/tfs.jsonl --base-model meta-llama/Llama-3.2-3B-Instruct \
        --out checkpoints/tfs_parser
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# LoRA defaults — small rank keeps the planner/parser fine-tune Colab-feasible.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_LEN = 1024


@dataclass
class TrainConfig:
    data_path: str
    base_model: str
    out_dir: str
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = MAX_LEN
    resume_adapter: Optional[str] = None  # continue from a prior adapter (three-step)


def train(cfg: TrainConfig) -> str:
    """Run one LoRA SFT pass; return the output checkpoint dir."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    model.gradient_checkpointing_enable()
    if cfg.resume_adapter:  # three-step: keep accumulating on the same adapter
        model = PeftModel.from_pretrained(model, cfg.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))

    def tokenize(example):
        return _tokenize_sft(example, tokenizer, cfg.max_len)

    dataset = load_dataset("json", data_files=cfg.data_path, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=cfg.out_dir, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        bf16=torch.cuda.is_available(), logging_steps=20, save_strategy="epoch",
        report_to=[], remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)).train()

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    return cfg.out_dir


def train_three_step(data_path: str, base_model: str, out_dir: str, **kw) -> str:
    """TFS-KBQA three-step fine-tuning: three sequential LoRA passes, each resuming
    the previous adapter (curriculum: parse → execution-filtered → full).

    The data files default to ``<data>`` for all three steps; pass
    ``step_data=[s1, s2, s3]`` to supply step-specific slices (e.g. only examples
    whose gold Cypher executes non-empty for step 2).
    """
    steps = kw.pop("step_data", None) or [data_path, data_path, data_path]
    adapter = None
    for i, step_path in enumerate(steps, start=1):
        step_out = out_dir if i == len(steps) else f"{out_dir}_step{i}"
        cfg = TrainConfig(data_path=step_path, base_model=base_model, out_dir=step_out,
                          resume_adapter=adapter, **kw)
        adapter = train(cfg)
    return out_dir


def _tokenize_sft(example: dict, tokenizer, max_len: int) -> dict:
    """Standard SFT tokenization: concatenate instruction+output, mask the
    instruction tokens in the labels so loss is only on the target."""
    instruction, output = example["instruction"], example["output"]
    prompt_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(output + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LoRA fine-tune a KBQA planner/parser.")
    p.add_argument("--task", choices=["rog", "tfs"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--three-step", action="store_true", help="TFS three-step curriculum")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args(argv)

    hp = dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    if args.task == "tfs" and args.three_step:
        out = train_three_step(args.data, args.base_model, args.out, **hp)
    else:
        out = train(TrainConfig(data_path=args.data, base_model=args.base_model,
                                out_dir=args.out, **hp))
    print(f"[{args.task}] checkpoint -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
