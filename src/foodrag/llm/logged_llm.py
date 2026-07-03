"""LiteLLM subclass that logs every call and applies the standard retry policy.

Drop-in replacement for ``llama_index.llms.litellm.LiteLLM`` used by the
config factory. Wraps ``complete`` / ``chat`` with structured JSONL logging
(:mod:`foodrag.llm.call_log`) and exponential-backoff retry
(:mod:`foodrag.llm.retry`).

It also makes native **function calling** usable with the agentic workflow
(:class:`~llama_index.core.agent.workflow.FunctionAgent`) on OpenAI-compatible
providers whose models LiteLLM's static model DB doesn't recognise as
tool-callers (e.g. DeepInfra-hosted ``meta-llama/Llama-3.3-70B-Instruct``):

* ``metadata.is_function_calling_model`` is forced ``True`` so llama-index sends
  the ``tools`` payload (verified: DeepInfra returns proper ``tool_calls``);
* ``astream_chat_with_tools`` is routed through the **non-streaming** chat path,
  because the upstream ``llama-index-llms-litellm`` streaming implementation only
  aggregates the legacy ``function_call`` delta and silently drops modern
  ``tool_calls`` deltas — which made FunctionAgent see zero tool calls and the
  whole agent chain collapse to empty plans.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ChatResponseAsyncGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.tools.types import BaseTool
from llama_index.llms.litellm import LiteLLM

from foodrag.llm import call_log
from foodrag.llm.retry import llm_retry


class LoggedLiteLLM(LiteLLM):
    """LiteLLM with call logging + retry. Extra fields are pydantic fields."""

    log_calls: bool = True
    retry_attempts: int = 6
    retry_initial: float = 2.0
    retry_max_wait: float = 60.0
    # Force native tool-calling on for OpenAI-compatible providers whose models
    # LiteLLM's static DB doesn't list as function-callers (e.g. DeepInfra Llama-3.3).
    force_function_calling: bool = True

    @property
    def metadata(self) -> LLMMetadata:
        md = super().metadata
        if self.force_function_calling:
            md.is_function_calling_model = True
        return md

    async def astream_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Stream chat-with-tools via the NON-streaming path.

        Upstream's streaming ``_astream_chat`` only handles the deprecated
        ``function_call`` delta and drops ``tool_calls`` deltas, so tool calls never
        reach ``additional_kwargs`` and FunctionAgent sees none. The non-streaming
        ``achat`` uses ``from_litellm_message``, which preserves ``tool_calls``. We
        run it once and yield a single terminal chunk (agents consume only the last).
        """
        chat_kwargs = self._prepare_chat_with_tools_compat(
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_required=tool_required,
            **kwargs,
        )
        response = await self.achat(**chat_kwargs)

        async def _gen() -> ChatResponseAsyncGen:
            yield response

        return _gen()

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
