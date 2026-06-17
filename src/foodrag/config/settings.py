"""Central configuration + provider wiring.

Single place that turns ``config/models.yml`` + ``.env`` into ready-to-use
LlamaIndex LLM / embedding objects, so notebooks and scripts never hardcode an
endpoint, model id, or API key again.

Typical use::

    from foodrag.config import settings
    settings.apply_global_settings()          # sets llama_index Settings.llm/embed_model
    llm = settings.build_llm()
    judge = settings.build_judge("gemma")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_models_path() -> Path:
    return repo_root() / "config" / "models.yml"


def load_env(dotenv_path: Optional[Path] = None, override: bool = False) -> None:
    """Load .env into os.environ (idempotent). No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    path = Path(dotenv_path) if dotenv_path else repo_root() / ".env"
    if path.exists():
        load_dotenv(path, override=override)


@dataclass
class ProviderSpec:
    name: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    local: bool = False

    def api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env)
        if not key and not self.local:
            raise RuntimeError(
                f"Provider {self.name!r} needs env var {self.api_key_env!r}; "
                f"it is not set (add it to .env)."
            )
        return key


@dataclass
class ModelsConfig:
    providers: dict[str, ProviderSpec]
    llm: dict[str, Any]
    embedding: dict[str, Any]
    embedding_candidates: list[dict[str, Any]] = field(default_factory=list)
    judges: dict[str, Any] = field(default_factory=dict)

    def provider(self, name: str) -> ProviderSpec:
        if name not in self.providers:
            raise KeyError(f"Unknown provider {name!r}. Known: {list(self.providers)}")
        return self.providers[name]

    @property
    def judge_models(self) -> list[dict[str, Any]]:
        return self.judges.get("models", [])

    def judge_spec(self, name: Optional[str] = None) -> dict[str, Any]:
        name = name or self.judges.get("default")
        for m in self.judge_models:
            if m["name"] == name:
                return m
        raise KeyError(f"Unknown judge {name!r}. Known: {[m['name'] for m in self.judge_models]}")

    def candidate(self, name: str) -> dict[str, Any]:
        for c in self.embedding_candidates:
            if c["name"] == name:
                return c
        raise KeyError(f"Unknown embedding candidate {name!r}.")


@lru_cache(maxsize=4)
def load_models_config(path: Optional[str] = None) -> ModelsConfig:
    p = Path(path) if path else default_models_path()
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    providers = {
        name: ProviderSpec(
            name=name,
            api_base=spec.get("api_base"),
            api_key_env=spec.get("api_key_env"),
            local=spec.get("local", False),
        )
        for name, spec in (raw.get("providers") or {}).items()
    }
    return ModelsConfig(
        providers=providers,
        llm=raw.get("llm") or {},
        embedding=raw.get("embedding") or {},
        embedding_candidates=raw.get("embedding_candidates") or [],
        judges=raw.get("judges") or {},
    )


# -- builders --------------------------------------------------------------
def _litellm_model_name(provider: ProviderSpec, model: str) -> str:
    """OpenAI-compatible routing: LiteLLM calls api_base when prefixed openai/."""
    return model if model.startswith("openai/") else f"openai/{model}"


def build_llm(spec: Optional[dict[str, Any]] = None, config: Optional[ModelsConfig] = None, **overrides):
    """Build a LlamaIndex LLM from a model spec (defaults to config.llm)."""
    from llama_index.llms.litellm import LiteLLM

    config = config or load_models_config()
    load_env()
    spec = {**(spec or config.llm), **overrides}
    provider = config.provider(spec["provider"])
    return LiteLLM(
        api_base=provider.api_base,
        api_key=provider.api_key(),
        model=_litellm_model_name(provider, spec["model"]),
        max_retries=spec.get("max_retries", 5),
        temperature=spec.get("temperature", 0.1),
        request_timeout=spec.get("request_timeout", 120),
    )


def build_judge(name: Optional[str] = None, config: Optional[ModelsConfig] = None, **overrides):
    """Build a judge LLM; enforces that it differs from the main generator."""
    config = config or load_models_config()
    spec = config.judge_spec(name)
    if spec["model"] == config.llm.get("model"):
        raise ValueError(
            f"Judge model {spec['model']!r} must differ from the main agent model."
        )
    return build_llm(spec=spec, config=config, **overrides)


def build_embedding(spec: Optional[dict[str, Any]] = None, config: Optional[ModelsConfig] = None):
    """Build a LlamaIndex embedding (API via LiteLLM, or local via HuggingFace)."""
    config = config or load_models_config()
    load_env()
    spec = spec or config.embedding
    provider = config.provider(spec["provider"])

    if provider.local:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        token = provider.api_key()
        if token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            os.environ.setdefault("HF_TOKEN", token)
        return HuggingFaceEmbedding(
            model_name=spec["model"],
            cache_folder=str(repo_root() / "data" / "cache"),
        )

    from llama_index.embeddings.litellm import LiteLLMEmbedding

    return LiteLLMEmbedding(
        api_base=provider.api_base,
        api_key=provider.api_key(),
        model_name=_litellm_model_name(provider, spec["model"]),
    )


def apply_global_settings(
    llm: Any = None,
    embed_model: Any = None,
    chunk_size: Optional[int] = None,
    config: Optional[ModelsConfig] = None,
) -> None:
    """Set llama_index global Settings.{llm,embed_model,chunk_size}."""
    from llama_index.core import Settings

    config = config or load_models_config()
    Settings.llm = llm if llm is not None else build_llm(config=config)
    Settings.embed_model = embed_model if embed_model is not None else build_embedding(config=config)
    Settings.chunk_size = chunk_size or config.embedding.get("chunk_size", 512)


def embedding_dim(spec: Optional[dict[str, Any]] = None, config: Optional[ModelsConfig] = None) -> int:
    config = config or load_models_config()
    return (spec or config.embedding).get("dim", 1024)
