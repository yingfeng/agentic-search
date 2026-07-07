"""Error handling and retry utilities."""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"All {attempts} retries exhausted: {last_error}")


async def retry_async(
    fn: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Any:
    """Retry an async function with exponential backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable_exceptions as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (backoff ** attempt), max_delay)
                logger.warning(f"Retry {attempt + 1}/{max_attempts} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    raise RetryExhaustedError(max_attempts, last_error)


def async_retry_decorator(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """Decorator: retry an async function on failure."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                lambda: fn(*args, **kwargs),
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
            )
        return wrapper
    return decorator
