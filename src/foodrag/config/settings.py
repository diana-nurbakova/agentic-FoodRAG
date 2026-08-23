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
    api_base_env: Optional[str] = None  # resolve base URL from env (e.g. OLLAMA_BASE_URL)
    api_key_env: Optional[str] = None
    kind: str = "openai"  # "openai" (LiteLLM) | "ollama" (PagodaOllamaLLM)
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

    def resolved_api_base(self) -> Optional[str]:
        if self.api_base_env:
            base = os.environ.get(self.api_base_env)
            if not base:
                raise RuntimeError(
                    f"Provider {self.name!r} needs env var {self.api_base_env!r}; "
                    f"it is not set (add it to .env)."
                )
            return base
        return self.api_base


@dataclass
class ModelsConfig:
    providers: dict[str, ProviderSpec]
    llm: dict[str, Any]
    embedding: dict[str, Any]
    embedding_candidates: list[dict[str, Any]] = field(default_factory=list)
    judges: dict[str, Any] = field(default_factory=dict)
    annotation: dict[str, Any] = field(default_factory=dict)
    retriever_backbones: list[dict[str, Any]] = field(default_factory=list)

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

    def backbone(self, name: str) -> dict[str, Any]:
        """The Text-to-Pandas code-generation LLM spec named ``name``."""
        for b in self.retriever_backbones:
            if b["name"] == name:
                return b
        raise KeyError(
            f"Unknown retriever backbone {name!r}. "
            f"Known: {[b['name'] for b in self.retriever_backbones]}"
        )


@lru_cache(maxsize=4)
def load_models_config(path: Optional[str] = None) -> ModelsConfig:
    p = Path(path) if path else default_models_path()
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    providers = {
        name: ProviderSpec(
            name=name,
            api_base=spec.get("api_base"),
            api_base_env=spec.get("api_base_env"),
            api_key_env=spec.get("api_key_env"),
            kind=spec.get("kind", "openai"),
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
        annotation=raw.get("annotation") or {},
        retriever_backbones=raw.get("retriever_backbones") or [],
    )


# -- builders --------------------------------------------------------------
def _litellm_model_name(provider: ProviderSpec, model: str) -> str:
    """OpenAI-compatible routing: LiteLLM calls api_base when prefixed openai/."""
    return model if model.startswith("openai/") else f"openai/{model}"


def build_llm(
    spec: Optional[dict[str, Any]] = None,
    config: Optional[ModelsConfig] = None,
    logged: bool = True,
    **overrides,
):
    """Build a LlamaIndex LLM from a model spec (defaults to config.llm).

    By default returns a :class:`~foodrag.llm.logged_llm.LoggedLiteLLM` with
    structured call logging + standardized retry. Set ``logged=False`` for a
    plain LiteLLM.
    """
    config = config or load_models_config()
    load_env()
    spec = {**(spec or config.llm), **overrides}
    provider = config.provider(spec["provider"])

    # Ollama-on-Pagoda uses the official ollama client (not LiteLLM) for its
    # proxy-aware streaming/retry behavior.
    if provider.kind == "ollama":
        from foodrag.llm.ollama_pagoda import PagodaOllamaLLM

        return PagodaOllamaLLM(
            model=spec["model"],
            base_url=provider.resolved_api_base(),
            api_key=provider.api_key() or "",
            temperature=spec.get("temperature", 0.1),
            num_predict=spec.get("num_predict", -1),
            keep_alive=spec.get("keep_alive", "30m"),
            max_attempts=spec.get("max_attempts", 12),
            read_timeout=spec.get("read_timeout", 600.0),
            context_window=spec.get("context_window", 8192),
            log_calls=logged,
        )

    kwargs = dict(
        api_base=provider.resolved_api_base(),
        api_key=provider.api_key(),
        model=_litellm_model_name(provider, spec["model"]),
        max_retries=spec.get("max_retries", 5),
        temperature=spec.get("temperature", 0.1),
        request_timeout=spec.get("request_timeout", 120),
    )
    if logged:
        from foodrag.llm.logged_llm import LoggedLiteLLM

        return LoggedLiteLLM(**kwargs)
    from llama_index.llms.litellm import LiteLLM

    return LiteLLM(**kwargs)


def build_annotation_llm(config: Optional[ModelsConfig] = None, logged: bool = True, **overrides):
    """Build the meal-type annotation LLM (defaults to config.annotation, e.g. Pagoda Ollama)."""
    config = config or load_models_config()
    spec = config.annotation or config.llm
    return build_llm(spec=spec, config=config, logged=logged, **overrides)


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
