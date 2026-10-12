"""Re-run the judge scoring of the QA sets' ``llm_questions`` with the fixed rating prompt.

The original ``rate_reformulated_questions`` prompt dropped the original question and persona
for recipe questions (see ``qa_generator._rating_prompt``), so every stored ``score`` is
invalid. This script clears them and rescores with the configured main LLM
(``config/models.yml:llm``, the same Llama-3.3-70B that produced the originals).

Writes, per QA set:

* ``data/qa/_backup_prescore/<qa>/recipe_question_answer_set.pkl`` -- the untouched original
  (kept outside the QA folder: ``load_data`` loads every file there by stem);
* ``data/qa/_rescore_ckpt/<qa>.json`` -- ``{qa_index: [scores]}``, saved after every chunk so
  an interrupted run resumes without re-paying for finished entries;
* the QA pickle itself, rewritten once at the end with the new scores.

Usage
-----
    python scripts/rescore_llm_questions.py --qa main_10k_tight --limit 5 --dry-run
    python scripts/rescore_llm_questions.py --qa main_10k_tight main_10k
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodrag.config.settings import build_llm  # noqa: E402
from foodrag.persona import qa_generator  # noqa: E402
from foodrag.persona.qa_generator import QuestionType  # noqa: E402

FILE_NAME = "recipe_question_answer_set"
QA_DIR = ROOT / "data" / "qa"


def rescore(qa_name: str, llm, *, batch_size: int, chunk: int, passes: int,
            limit: int | None, dry_run: bool) -> None:
    pkl = QA_DIR / qa_name / f"{FILE_NAME}.pkl"
    ckpt_path = QA_DIR / "_rescore_ckpt" / f"{qa_name}.json"
    print(f"\n=== {qa_name}: loading {pkl} ...", flush=True)
    with open(pkl, "rb") as f:
        qa_set = pickle.load(f)

    # Clear every old score, then re-apply any already-rescored entries from the checkpoint.
    ckpt = {} if dry_run or not ckpt_path.exists() else json.loads(ckpt_path.read_text())
    for qa in qa_set:
        for q in qa.get("llm_questions") or []:
            q.pop("score", None)
    for idx, scores in ckpt.items():
        for q, s in zip(qa_set[int(idx)]["llm_questions"], scores):
            q["score"] = s

    todo = [i for i, qa in enumerate(qa_set) if qa.get("llm_questions")]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} entries with rewrites; {len(ckpt)} already rescored (checkpoint)", flush=True)

    for p in range(passes):
        pending = [i for i in todo if "score" not in qa_set[i]["llm_questions"][0]]
        if not pending:
            break
        print(f"pass {p + 1}: scoring {len(pending)} entries", flush=True)
        for start in range(0, len(pending), chunk):
            idxs = pending[start:start + chunk]
            # path=None + huge storage_frequency: the rater never saves; we checkpoint below.
            asyncio.run(qa_generator.rate_reformulated_questions(
                [qa_set[i] for i in idxs], llm, None, FILE_NAME, QuestionType.RECIPE,
                batch_size=batch_size, storage_frequency=10 ** 12))
            for i in idxs:
                if "score" in qa_set[i]["llm_questions"][0]:
                    ckpt[str(i)] = [q["score"] for q in qa_set[i]["llm_questions"]]
            if not dry_run:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_text(json.dumps(ckpt))
            print(f"  {min(start + chunk, len(pending))}/{len(pending)} done; "
                  f"{len(ckpt)} rescored in total", flush=True)

    scored = [i for i in todo if "score" in qa_set[i]["llm_questions"][0]]
    dist = Counter(s for i in scored for s in ckpt[str(i)])
    best5 = sum(1 for i in scored if 5 in ckpt[str(i)])
    print(f"{qa_name}: scored {len(scored)}/{len(todo)}; score distribution {dict(sorted(dist.items()))}; "
          f"entries with a 5: {best5}", flush=True)

    if dry_run:
        for i in todo[:3]:
            print(f"\n[{i}]", [(q['question'][:70], q.get('score')) for q in qa_set[i]["llm_questions"]])
        print("\n(dry run: nothing written)")
        return

    backup = QA_DIR / "_backup_prescore" / qa_name / f"{FILE_NAME}.pkl"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        print(f"backing up original -> {backup}", flush=True)
        shutil.copy2(pkl, backup)
    tmp = pkl.with_name(pkl.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(qa_set, f)
    tmp.replace(pkl)
    print(f"wrote {pkl}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qa", nargs="+", required=True, help="QA set names under data/qa")
    ap.add_argument("--batch-size", type=int, default=32, help="concurrent LLM requests")
    ap.add_argument("--chunk", type=int, default=500, help="entries per checkpoint")
    ap.add_argument("--passes", type=int, default=3, help="retry passes for unparsable replies")
    ap.add_argument("--limit", type=int, help="only the first N entries (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="score but write nothing")
    args = ap.parse_args()
    if args.limit and not args.dry_run:
        ap.error("--limit is for smoke tests; combine it with --dry-run")

    llm = build_llm(logged=False)
    print(f"judge: {llm.model} @ {llm.additional_kwargs.get('api_base')}", flush=True)
    for name in args.qa:
        rescore(name, llm, batch_size=args.batch_size, chunk=args.chunk, passes=args.passes,
                limit=args.limit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
