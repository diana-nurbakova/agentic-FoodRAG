"""Logging-service unit tests: HF push path (mocked) + status."""
import json
import sys
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gui]


@pytest.fixture
def mock_hf(monkeypatch):
    """Inject a fake huggingface_hub.HfApi capturing uploads."""
    captured = {}

    class FakeApi:
        def __init__(self, token=None):
            captured["token"] = token

        def create_repo(self, **kw):
            captured["create"] = kw

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            with open(path_or_fileobj, encoding="utf-8") as fh:
                captured["content"] = fh.read()
            captured.update(path=path_in_repo, repo=repo_id, type=repo_type)

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return captured


def test_status_remote_off(monkeypatch):
    monkeypatch.delenv("FOODRAG_DATASET_REPO", raising=False)
    from ui.services import logging_service as L

    assert L.remote_enabled() is False
    st = L.status()
    assert st["remote_enabled"] is False and st["interval_s"] == 300


def test_flush_session_pushes_jsonl(monkeypatch, mock_hf):
    monkeypatch.setenv("FOODRAG_DATASET_REPO", "tester/foodrag-eval")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    from ui.services import logging_service as L

    events = [{"session_id": "abc", "event_type": "consent_given", "payload": {}},
              {"session_id": "abc", "event_type": "hexad_completed", "payload": {"x": 1}}]
    ok, msg = L.flush_session("abc", events)
    assert ok and msg == "ok"
    assert mock_hf["path"] == "logs/abc.jsonl" and mock_hf["repo"] == "tester/foodrag-eval"
    lines = [ln for ln in mock_hf["content"].splitlines() if ln.strip()]
    assert len(lines) == 2 and json.loads(lines[0])["event_type"] == "consent_given"
    last = L.last_push("abc")
    assert last["ok"] is True and last["count"] == 2


def test_flush_session_noop_when_remote_disabled(monkeypatch):
    monkeypatch.delenv("FOODRAG_DATASET_REPO", raising=False)
    from ui.services import logging_service as L

    ok, msg = L.flush_session("abc", [{"event_type": "x"}])
    assert ok is False and "disabled" in msg
