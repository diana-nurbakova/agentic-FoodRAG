"""Unit tests for the resumable run-store and the retry decorator."""
from __future__ import annotations

from pathlib import Path

import pytest

from foodrag.llm.retry import is_retryable, llm_retry
from foodrag.llm.runstore import RunStore

pytestmark = pytest.mark.unit


# -- RunStore --------------------------------------------------------------
def test_process_skips_done_and_resumes(tmp_path: Path):
    path = tmp_path / "items.jsonl"
    items = [{"id": str(i)} for i in range(5)]
    calls = []

    store = RunStore(path)
    store.process(items, key=lambda x: x["id"], work=lambda x: calls.append(x["id"]) or f"r{x['id']}",
                  verbose=False)
    assert sorted(calls) == ["0", "1", "2", "3", "4"]
    assert store.summary()["ok"] == 5

    # New store from the same file: everything is done, nothing re-runs.
    calls.clear()
    store2 = RunStore(path)
    store2.process(items, key=lambda x: x["id"], work=lambda x: calls.append(x["id"]), verbose=False)
    assert calls == []
    assert len(store2.results()) == 5


def test_errors_recorded_and_retried(tmp_path: Path):
    path = tmp_path / "items.jsonl"
    items = [{"id": "a"}, {"id": "b"}]
    attempts = {"a": 0, "b": 0}

    def flaky(x):
        attempts[x["id"]] += 1
        if x["id"] == "b" and attempts["b"] == 1:
            raise RuntimeError("boom")
        return "ok"

    store = RunStore(path)
    store.process(items, key=lambda x: x["id"], work=flaky, verbose=False)
    assert store.is_done("a")
    assert not store.is_done("b")
    assert "b" in store.errors()

    # Resume: only the errored item re-runs (and now succeeds).
    store2 = RunStore(path)
    assert store2.pending_ids(["a", "b"]) == ["b"]
    store2.process(items, key=lambda x: x["id"], work=flaky, verbose=False)
    assert store2.is_done("b")
    assert attempts["a"] == 1  # 'a' never re-ran


def test_pending_excludes_errors_when_not_retrying(tmp_path: Path):
    store = RunStore(tmp_path / "i.jsonl")
    store.mark_ok("a")
    store.mark_error("b", "boom")
    assert store.pending_ids(["a", "b", "c"], retry_errors=False) == ["c"]
    assert store.pending_ids(["a", "b", "c"], retry_errors=True) == ["b", "c"]


# -- retry -----------------------------------------------------------------
class RateLimitError(Exception):
    pass


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    @llm_retry(max_attempts=4, initial=0.0, max_wait=0.0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("slow down")
        return "done"

    assert flaky() == "done"
    assert calls["n"] == 3


def test_retry_reraises_non_retryable():
    @llm_retry(max_attempts=4, initial=0.0, max_wait=0.0)
    def boom():
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        boom()


def test_is_retryable_by_name():
    assert is_retryable(RateLimitError("x"))
    assert not is_retryable(ValueError("x"))
    assert is_retryable(ValueError("x"), extra=(ValueError,))
