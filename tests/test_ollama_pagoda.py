"""Unit tests for the Pagoda Ollama LLM resilience logic (no network)."""
from __future__ import annotations

import pytest

from foodrag.llm import ollama_pagoda
from foodrag.llm.ollama_pagoda import PagodaOllamaLLM, _is_transient

pytestmark = pytest.mark.unit


class _FakeClient:
    """Fake ollama client. `script` is a list of per-attempt behaviors:
    each item is either a list of response chunks (success) or an Exception
    (raised after yielding `partial_chunks`)."""

    def __init__(self, script, partial_chunks=()):
        self.script = list(script)
        self.partial_chunks = list(partial_chunks)
        self.calls = 0

    def generate(self, model, prompt, stream, keep_alive, options):
        behavior = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1

        def _gen():
            if isinstance(behavior, Exception):
                for c in self.partial_chunks:
                    yield {"response": c}
                raise behavior
            for c in behavior:
                yield {"response": c}

        return _gen()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ollama_pagoda.time, "sleep", lambda *_: None)


def _llm(client, **kw):
    llm = PagodaOllamaLLM(base_url="http://x", api_key="k", log_calls=False, **kw)
    llm._client = client
    return llm


def test_complete_accumulates_stream():
    llm = _llm(_FakeClient([["Break", "fast", "."]]))
    resp = llm.complete("classify")
    assert str(resp).strip() == "Breakfast."
    assert resp.additional_kwargs.get("partial", False) is False


def test_retry_on_transient_then_success():
    client = _FakeClient([RuntimeError("connection reset by peer"),
                          RuntimeError("read error"),
                          ["ok"]])
    llm = _llm(client, max_attempts=5)
    resp = llm.complete("x")
    assert str(resp).strip() == "ok"
    assert client.calls == 3  # failed twice, succeeded on third


def test_partial_recovery_when_exhausted():
    # always fails after streaming some content; returns the partial
    client = _FakeClient([RuntimeError("server disconnected")] * 3,
                         partial_chunks=["par", "tial"])
    llm = _llm(client, max_attempts=2)
    resp = llm.complete("x")
    assert str(resp).strip() == "partial"
    assert resp.additional_kwargs.get("partial") is True


def test_non_transient_with_no_partial_raises():
    client = _FakeClient([ValueError("bad request")])
    llm = _llm(client, max_attempts=5)
    with pytest.raises(ValueError):
        llm.complete("x")
    assert client.calls == 1  # not retried


def test_num_predict_from_max_tokens():
    llm = _llm(_FakeClient([["hi"]]))
    opts = llm._options(max_tokens=42)
    assert opts["num_predict"] == 42


def test_is_transient_classification():
    assert _is_transient(RuntimeError("Connection reset"))
    assert _is_transient(RuntimeError("read timed out"))
    assert _is_transient(RuntimeError("server disconnected (EOF)"))
    assert not _is_transient(ValueError("malformed prompt"))


def test_metadata_model_name():
    llm = _llm(_FakeClient([["x"]]), model="llama3.3:70b")
    assert llm.metadata.model_name == "llama3.3:70b"
