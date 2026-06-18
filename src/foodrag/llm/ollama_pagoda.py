"""LlamaIndex LLM backed by Ollama on Pagoda (lab server, behind an auth proxy).

Pagoda hosts open models (e.g. llama3.3:70b) for free but is slow and the
reverse proxy drops connections under load. This wrapper follows the lab's
guidance:

- official ``ollama`` client with a Bearer token (host/key from env);
- generous httpx timeouts (read=600s) for a proxied 70B;
- ``keep_alive`` so the model stays GPU-resident between calls;
- a process-wide ``Semaphore(2)`` cap on in-flight requests;
- streamed responses with extended retries on transient/disconnect errors and
  partial-content recovery (return what streamed rather than discard it).

Requires the lab VPN to resolve the host. Used mainly for the (huge, but free)
meal-type annotation pass; see ``foodrag.config.settings.build_annotation_llm``.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

# The proxy drops connections under load: cap total in-flight requests to 2
# across the whole process (shared by every PagodaOllamaLLM instance).
_SEMAPHORE = threading.Semaphore(2)

_TRANSIENT_SUBSTRINGS = (
    "disconnect", "eof", "broken pipe", "connection reset",
    "timed out", "timeout", "read error", "incomplete",
)


def _is_transient(exc: BaseException) -> bool:
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                            httpx.RemoteProtocolError, httpx.WriteError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_SUBSTRINGS)


class PagodaOllamaLLM(CustomLLM):
    """Ollama-on-Pagoda LLM with proxy-aware resilience."""

    model: str = "llama3.3:70b"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.1
    num_predict: int = -1  # ollama: -1 = no limit
    keep_alive: str = "30m"
    max_attempts: int = 12
    connect_timeout: float = 30.0
    read_timeout: float = 600.0
    context_window: int = 8192
    log_calls: bool = True

    _client: Any = PrivateAttr(default=None)

    # -- client ------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import httpx
            import ollama

            self._client = ollama.Client(
                host=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=httpx.Timeout(
                    connect=self.connect_timeout, read=self.read_timeout,
                    write=30.0, pool=self.read_timeout,
                ),
            )
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self.context_window,
            num_output=self.num_predict if self.num_predict > 0 else 4096,
            is_chat_model=False,
        )

    def _options(self, **overrides) -> dict:
        opts = {"temperature": self.temperature, "num_predict": self.num_predict}
        # ollama uses num_predict, not max_tokens
        if "max_tokens" in overrides and overrides["max_tokens"] is not None:
            opts["num_predict"] = overrides.pop("max_tokens")
        opts.update({k: v for k, v in overrides.items() if k in ("temperature", "num_predict", "top_p", "seed")})
        return opts

    # -- core retrying streamed generate ----------------------------------
    def _generate(self, prompt: str, options: dict) -> tuple[str, bool]:
        """Stream a generation with retries. Returns (text, is_partial)."""
        client = self._get_client()
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            partial = ""
            try:
                with _SEMAPHORE:
                    for chunk in client.generate(
                        model=self.model, prompt=prompt, stream=True,
                        keep_alive=self.keep_alive, options=options,
                    ):
                        partial += chunk.get("response", "")
                return partial, False
            except Exception as exc:  # noqa: BLE001
                exc._partial = partial  # type: ignore[attr-defined]
                last_exc = exc
                if not _is_transient(exc) or attempt == self.max_attempts:
                    break
                time.sleep(min(2.0 * (2 ** (attempt - 1)), 60.0))  # exp backoff, capped 60s
        # exhausted (or non-transient): salvage any partial content
        partial = getattr(last_exc, "_partial", "") if last_exc else ""
        if partial:
            return partial, True
        assert last_exc is not None
        raise last_exc

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        options = self._options(**kwargs)
        if not self.log_calls:
            text, is_partial = self._generate(prompt, options)
            return CompletionResponse(text=text, additional_kwargs={"partial": True} if is_partial else {})

        from foodrag.llm import call_log

        with call_log.record_call("complete", model=self.model, prompt=prompt) as box:
            text, is_partial = self._generate(prompt, options)
            resp = CompletionResponse(text=text, additional_kwargs={"partial": True} if is_partial else {})
            box["response"] = resp
            return resp

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        client = self._get_client()
        options = self._options(**kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            with _SEMAPHORE:
                for chunk in client.generate(
                    model=self.model, prompt=prompt, stream=True,
                    keep_alive=self.keep_alive, options=options,
                ):
                    delta = chunk.get("response", "")
                    text += delta
                    yield CompletionResponse(text=text, delta=delta)

        return gen()
