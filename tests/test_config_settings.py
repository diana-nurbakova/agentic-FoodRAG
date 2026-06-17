"""Unit tests for config/provider wiring (no network)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from foodrag.config import settings
from foodrag.config.settings import load_models_config

pytestmark = pytest.mark.unit

MODELS_YML = """
providers:
  deepinfra:
    api_base: "https://api.deepinfra.com/v1/openai"
    api_key_env: DEEPINFRA_API_KEY
  openrouter:
    api_base: "https://openrouter.ai/api/v1"
    api_key_env: OPENROUTER_API_KEY
  huggingface:
    local: true
    api_key_env: HF_TOKEN
llm:
  provider: deepinfra
  model: "meta-llama/Llama-3.3-70B-Instruct"
embedding:
  provider: deepinfra
  model: "BAAI/bge-m3"
  dim: 1024
  chunk_size: 512
embedding_candidates:
  - {name: bge-m3, provider: deepinfra, model: "BAAI/bge-m3", dim: 1024}
  - {name: mpnet-base, provider: huggingface, model: "sentence-transformers/all-mpnet-base-v2", dim: 768}
judges:
  default: deepseek-v3
  models:
    - {name: deepseek-v3, provider: deepinfra, model: "deepseek-ai/DeepSeek-V3"}
    - {name: gemma, provider: openrouter, model: "google/gemma-3-27b-it"}
"""


@pytest.fixture
def cfg(tmp_path: Path):
    load_models_config.cache_clear()
    p = tmp_path / "models.yml"
    p.write_text(textwrap.dedent(MODELS_YML), encoding="utf-8")
    config = load_models_config(str(p))
    yield config
    load_models_config.cache_clear()


def test_parses_providers_and_models(cfg):
    assert set(cfg.providers) == {"deepinfra", "openrouter", "huggingface"}
    assert cfg.llm["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert cfg.embedding["dim"] == 1024
    assert cfg.providers["huggingface"].local is True


def test_judge_lookup_and_default(cfg):
    assert cfg.judge_spec()["name"] == "deepseek-v3"  # default
    assert cfg.judge_spec("gemma")["model"] == "google/gemma-3-27b-it"
    with pytest.raises(KeyError):
        cfg.judge_spec("does-not-exist")


def test_candidate_lookup(cfg):
    assert cfg.candidate("mpnet-base")["dim"] == 768
    with pytest.raises(KeyError):
        cfg.candidate("nope")


def test_litellm_model_name_prefix():
    p = settings.ProviderSpec(name="deepinfra", api_base="x", api_key_env="K")
    assert settings._litellm_model_name(p, "BAAI/bge-m3") == "openai/BAAI/bge-m3"
    assert settings._litellm_model_name(p, "openai/already") == "openai/already"


def test_provider_api_key_missing_raises(cfg, monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        cfg.providers["deepinfra"].api_key()


def test_local_provider_key_optional(cfg, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # local provider: missing key returns None instead of raising
    assert cfg.providers["huggingface"].api_key() is None


def test_judge_must_differ_from_generator(cfg, monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "dummy")
    # make the generator equal to the deepseek judge -> guard must fire
    cfg.llm["model"] = "deepseek-ai/DeepSeek-V3"
    with pytest.raises(ValueError):
        settings.build_judge("deepseek-v3", config=cfg)
