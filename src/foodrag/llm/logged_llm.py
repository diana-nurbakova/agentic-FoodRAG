"""LiteLLM subclass that logs every call and applies the standard retry policy.

Drop-in replacement for ``llama_index.llms.litellm.LiteLLM`` used by the
config factory. Wraps ``complete`` / ``chat`` with structured JSONL logging
(:mod:`foodrag.llm.call_log`) and exponential-backoff retry
(:mod:`foodrag.llm.retry`).
"""
from __future__ import annotations

from typing import Any

from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.llms.litellm import LiteLLM

from foodrag.llm import call_log
from foodrag.llm.retry import llm_retry


class LoggedLiteLLM(LiteLLM):
    """LiteLLM with call logging + retry. Extra fields are pydantic fields."""

    log_calls: bool = True
    retry_attempts: int = 6
    retry_initial: float = 2.0
    retry_max_wait: float = 60.0

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        @llm_retry(max_attempts=self.retry_attempts, initial=self.retry_initial, max_wait=self.retry_max_wait)
        def _do():
            return super(LoggedLiteLLM, self).complete(prompt, formatted=formatted, **kwargs)

        if not self.log_calls:
            return _do()
        with call_log.record_call("complete", model=self.model, prompt=prompt) as box:
            resp = _do()
            box["response"] = resp
            return resp

    @llm_chat_callback()
    def chat(self, messages, **kwargs: Any):
        @llm_retry(max_attempts=self.retry_attempts, initial=self.retry_initial, max_wait=self.retry_max_wait)
        def _do():
            return super(LoggedLiteLLM, self).chat(messages, **kwargs)

        if not self.log_calls:
            return _do()
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        with call_log.record_call("chat", model=self.model, prompt=prompt) as box:
            resp = _do()
            box["response"] = resp
            return resp
