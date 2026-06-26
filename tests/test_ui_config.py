"""Study-config persistence unit tests (spec §6.3)."""
import json
import sys
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gui]


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config storage at a tmp file and disable the dataset repo."""
    monkeypatch.setenv("FOODRAG_STUDY_CONFIG", str(tmp_path / "study_config.json"))
    monkeypatch.delenv("FOODRAG_DATASET_REPO", raising=False)
    # config_service caches a one-shot remote-pull flag at module scope; reset it
    from ui.services import config_service as CS
    CS._remote_pull_done = False
    yield


def test_local_round_trip():
    from ui.services import config_service as CS

    assert CS.load_persisted() is None  # nothing saved yet
    cfg = CS.effective_config()
    cfg["active_flow"] = "evaluation"
    cfg["pipeline_defaults"]["arena_available"] = False
    ok, msg = CS.save_config(cfg)
    assert ok and CS.local_path().exists()
    loaded = CS.load_persisted()
    assert loaded["active_flow"] == "evaluation"
    assert loaded["pipeline_defaults"]["arena_available"] is False


def test_deep_merge_backfills_new_keys():
    from ui.services import config_service as CS

    cfg = CS.effective_config()
    cfg["active_flow"] = "evaluation"
    CS.save_config(cfg)
    # Simulate an OLD persisted config missing a newer default key
    raw = json.loads(CS.local_path().read_text(encoding="utf-8"))
    del raw["pipeline_defaults"]["use_real_backend"]
    CS.local_path().write_text(json.dumps(raw), encoding="utf-8")

    merged = CS.effective_config()
    assert "use_real_backend" in merged["pipeline_defaults"]   # back-filled from defaults
    assert merged["active_flow"] == "evaluation"               # persisted value preserved


def test_remote_save_mocked(monkeypatch):
    monkeypatch.setenv("FOODRAG_DATASET_REPO", "tester/foodrag-eval")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    captured = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kw):
            captured["repo"] = kw.get("repo_id")

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            captured["path"] = path_in_repo
            captured["content"] = open(path_or_fileobj, encoding="utf-8").read()

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    from ui.services import config_service as CS

    ok, msg = CS.save_config({"active_flow": "exploration"})
    assert ok and captured["path"] == "config/study_config.json"
    assert json.loads(captured["content"])["active_flow"] == "exploration"
