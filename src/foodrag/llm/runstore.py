"""Resumable, checkpointed processing keyed by item id.

A :class:`RunStore` persists one record per item to a JSONL file with status
``ok`` / ``error`` and the result payload. Re-running skips items already
``ok`` and (optionally) retries ``error`` items — so a crashed QA-generation,
embedding, or eval run continues only on the remaining/failed items.

    store = RunStore(Path("data/runs/eval_1/items.jsonl"))
    store.process(questions, key=lambda q: q["id"], work=run_one)
    results = store.results()
"""
from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

OK = "ok"
ERROR = "error"


@dataclass
class ItemRecord:
    id: str
    status: str
    payload: Any = None
    error: Optional[str] = None
    ts: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == OK


class RunStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._records: dict[str, ItemRecord] = {}
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> "RunStore":
        self._records.clear()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self._records[d["id"]] = ItemRecord(
                        id=d["id"], status=d["status"], payload=d.get("payload"),
                        error=d.get("error"), ts=d.get("ts", ""),
                    )
        return self

    def _append(self, rec: ItemRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._records[rec.id] = rec
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.__dict__, ensure_ascii=False, default=str) + "\n")

    # -- status ------------------------------------------------------------
    def status(self, item_id: str) -> Optional[str]:
        rec = self._records.get(item_id)
        return rec.status if rec else None

    def is_done(self, item_id: str) -> bool:
        return self.status(item_id) == OK

    def mark_ok(self, item_id: str, payload: Any = None) -> None:
        self._append(ItemRecord(id=item_id, status=OK, payload=payload, ts=_now()))

    def mark_error(self, item_id: str, error: str) -> None:
        self._append(ItemRecord(id=item_id, status=ERROR, error=error, ts=_now()))

    def results(self) -> dict[str, Any]:
        return {rid: r.payload for rid, r in self._records.items() if r.is_ok}

    def errors(self) -> dict[str, str]:
        return {rid: r.error for rid, r in self._records.items() if r.status == ERROR}

    def pending_ids(self, all_ids: Iterable[str], retry_errors: bool = True) -> list[str]:
        out = []
        for i in all_ids:
            st = self.status(i)
            if st == OK:
                continue
            if st == ERROR and not retry_errors:
                continue
            out.append(i)
        return out

    def summary(self) -> dict[str, int]:
        n_ok = sum(1 for r in self._records.values() if r.is_ok)
        n_err = sum(1 for r in self._records.values() if r.status == ERROR)
        return {"ok": n_ok, "error": n_err, "total_recorded": len(self._records)}

    # -- driver ------------------------------------------------------------
    def process(
        self,
        items: Iterable[Any],
        key: Callable[[Any], str],
        work: Callable[[Any], Any],
        retry_errors: bool = True,
        on_done: Optional[Callable[[str, Any], None]] = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run ``work(item)`` for each not-yet-ok item, persisting each result.

        Exceptions in ``work`` are recorded as errors and do not abort the run.
        Returns the map of ok payloads (including those from prior runs).
        """
        items = list(items)
        total = len(items)
        skipped = done = failed = 0
        for idx, item in enumerate(items):
            item_id = str(key(item))
            st = self.status(item_id)
            if st == OK or (st == ERROR and not retry_errors):
                skipped += 1
                continue
            try:
                result = work(item)
                self.mark_ok(item_id, result)
                done += 1
                if on_done:
                    on_done(item_id, result)
            except Exception as exc:  # noqa: BLE001 - persist and continue
                self.mark_error(item_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                failed += 1
            if verbose and (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{total}] done={done} failed={failed} skipped={skipped}")
        if verbose:
            print(f"RunStore done: {done} new, {failed} failed, {skipped} skipped of {total}.")
        return self.results()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
