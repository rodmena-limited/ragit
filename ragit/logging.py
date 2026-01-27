#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Structured logging for ragit.

Provides consistent logging across all ragit components with:
- Operation timing
- Context tracking
- Configurable log levels
"""

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

# Create ragit logger
logger = logging.getLogger("ragit")

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def setup_logging(level: str = "INFO", format_string: str | None = None) -> None:
    """Configure ragit logging.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    format_string : str, optional
        Custom format string. If None, uses default format.

    Examples
    --------
    >>> from ragit.logging import setup_logging
    >>> setup_logging("DEBUG")
    """
    logger.setLevel(level.upper())

    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level.upper())

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


@contextmanager
def log_operation(operation: str, **context: Any) -> Generator[dict[str, Any], None, None]:
    """Context manager for logging operations with timing.

    Parameters
    ----------
    operation : str
        Name of the operation being performed.
    **context
        Additional context to include in log messages.

    Yields
    ------
    dict
        Mutable dict to add additional context during the operation.

    Examples
    --------
    >>> with log_operation("embed", model="nomic-embed-text") as ctx:
    ...     result = provider.embed(text, model)
    ...     ctx["dimensions"] = len(result.embedding)
    """
    start = time.perf_counter()
    extra_context: dict[str, Any] = {}

    # Build context string
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""

    logger.debug(f"{operation}.start" + (f" [{ctx_str}]" if ctx_str else ""))

    try:
        yield extra_context
        duration_ms = (time.perf_counter() - start) * 1000

        # Combine original context with extra context
        all_context = {**context, **extra_context, "duration_ms": f"{duration_ms:.2f}"}
        ctx_str = ", ".join(f"{k}={v}" for k, v in all_context.items())

        logger.info(f"{operation}.success [{ctx_str}]")
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        all_context = {**context, **extra_context, "duration_ms": f"{duration_ms:.2f}", "error": str(e)}
        ctx_str = ", ".join(f"{k}={v}" for k, v in all_context.items())

        logger.error(f"{operation}.failed [{ctx_str}]", exc_info=True)
        raise


def log_method(operation: str) -> Callable[[F], F]:
    """Decorator for logging method calls with timing.

    Parameters
    ----------
    operation : str
        Name of the operation for logging.

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> class MyProvider:
    ...     @log_method("embed")
    ...     def embed(self, text: str, model: str):
    ...         ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_operation(operation, method=func.__name__):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class LogContext:
    """Context tracker for correlating related log messages.

    Useful for tracing operations across multiple components.

    Examples
    --------
    >>> ctx = LogContext("query-123")
    >>> ctx.log("Starting retrieval", top_k=5)
    >>> ctx.log("Retrieved chunks", count=3)
    """

    def __init__(self, request_id: str | None = None):
        """Initialize log context.

        Parameters
        ----------
        request_id : str, optional
            Unique identifier for this context. Auto-generated if not provided.
        """
        self.request_id = request_id or f"req-{int(time.time() * 1000) % 100000}"
        self._start_time = time.perf_counter()

    def log(self, message: str, level: str = "INFO", **context: Any) -> None:
        """Log a message with this context.

        Parameters
        ----------
        message : str
            Log message.
        level : str
            Log level (DEBUG, INFO, WARNING, ERROR).
        **context
            Additional context key-value pairs.
        """
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
        full_msg = f"[{self.request_id}] {message}" + (f" [{ctx_str}]" if ctx_str else "") + f" (+{elapsed_ms:.0f}ms)"

        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, full_msg)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self.log(message, "DEBUG", **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self.log(message, "INFO", **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self.log(message, "WARNING", **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self.log(message, "ERROR", **context)
