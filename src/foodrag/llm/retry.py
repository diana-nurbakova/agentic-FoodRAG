"""One standardized retry policy for LLM/embedding calls.

Replaces the bespoke retry loops scattered across the codebase (the cooldown
loop in embedding_helper, the per-method tenacity decorators in eval_parser).
Exponential backoff + jitter, with extra patience for rate-limit errors.
"""
from __future__ import annotations

import functools
from typing import Callable, Iterable, Optional, Type

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Error class names that signal "retry, possibly after a longer wait".
_RATE_LIMIT_NAMES = {"RateLimitError", "Timeout", "APIConnectionError", "ServiceUnavailableError",
                     "InternalServerError", "APIError", "TryAgain"}


def is_retryable(exc: BaseException, extra: Iterable[Type[BaseException]] = ()) -> bool:
    name = type(exc).__name__
    if name in _RATE_LIMIT_NAMES:
        return True
    return isinstance(exc, tuple(extra)) if extra else False


def llm_retry(
    max_attempts: int = 6,
    initial: float = 2.0,
    max_wait: float = 60.0,
    extra_retryable: Iterable[Type[BaseException]] = (),
    on_retry: Optional[Callable] = None,
):
    """Decorator applying the standard LLM retry policy.

    Retries on known-transient provider errors (rate limits, timeouts, 5xx)
    plus any ``extra_retryable`` exception types, with exponential backoff and
    jitter. Re-raises the last exception after ``max_attempts``.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tenacity_retry = retry(
                reraise=True,
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential_jitter(initial=initial, max=max_wait),
                retry=retry_if_exception(lambda e: is_retryable(e, extra_retryable)),
                before_sleep=on_retry,
            )
            return tenacity_retry(func)(*args, **kwargs)

        return wrapper

    return decorator
