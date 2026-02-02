#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Custom exception hierarchy for ragit.

Provides structured exceptions for different failure types,
enabling better error handling and debugging.

Pattern inspired by ai4rag exception_handler.py.
"""

from typing import Any


class RagitError(Exception):
    """Base exception for all ragit errors.

    All ragit-specific exceptions inherit from this class,
    making it easy to catch all ragit errors with a single handler.

    Parameters
    ----------
    message : str
        Human-readable error message.
    original_exception : BaseException, optional
        The underlying exception that caused this error.

    Examples
    --------
    >>> try:
    ...     provider.embed("text", "model")
    ... except RagitError as e:
    ...     print(f"Ragit error: {e}")
    ...     if e.original_exception:
    ...         print(f"Caused by: {e.original_exception}")
    """

    def __init__(self, message: str, original_exception: BaseException | None = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message, including original exception if present."""
        if self.original_exception:
            return f"{self.message}: {self.original_exception}"
        return self.message


class ConfigurationError(RagitError):
    """Configuration validation or loading failed.

    Raised when:
    - Environment variables have invalid values
    - Required configuration is missing
    - URL formats are invalid
    """

    pass


class ProviderError(RagitError):
    """Provider communication or operation failed.

    Raised when:
    - Network connection to provider fails
    - Provider returns an error response
    - Provider timeout occurs
    """

    pass


class IndexingError(RagitError):
    """Document indexing or embedding failed.

    Raised when:
    - Embedding generation fails
    - Document chunking fails
    - Index building fails
    """

    pass


class RetrievalError(RagitError):
    """Retrieval operation failed.

    Raised when:
    - Query embedding fails
    - Search operation fails
    - No results can be retrieved
    """

    pass


class GenerationError(RagitError):
    """LLM generation failed.

    Raised when:
    - LLM call fails
    - Response parsing fails
    - Context exceeds model limits
    """

    pass


class EvaluationError(RagitError):
    """Evaluation or scoring failed.

    Raised when:
    - Metric calculation fails
    - Benchmark validation fails
    - Score extraction fails
    """

    pass


class ExceptionAggregator:
    """Collect and report exceptions during batch operations.

    Useful for operations that should continue even when some
    items fail, then report all failures at the end.

    Pattern from ai4rag exception_handler.py.

    Examples
    --------
    >>> aggregator = ExceptionAggregator()
    >>> for doc in documents:
    ...     try:
    ...         process(doc)
    ...     except Exception as e:
    ...         aggregator.record(f"doc:{doc.id}", e)
    >>> if aggregator.has_errors:
    ...     print(aggregator.get_summary())
    """

    def __init__(self) -> None:
        self._exceptions: list[tuple[str, Exception]] = []

    def record(self, context: str, exception: Exception) -> None:
        """Record an exception with context.

        Parameters
        ----------
        context : str
            Description of where/why the exception occurred.
        exception : Exception
            The exception that was raised.
        """
        self._exceptions.append((context, exception))

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self._exceptions) > 0

    @property
    def error_count(self) -> int:
        """Get the number of recorded errors."""
        return len(self._exceptions)

    @property
    def exceptions(self) -> list[tuple[str, Exception]]:
        """Get all recorded exceptions with their contexts."""
        return list(self._exceptions)

    def get_by_type(self, exc_type: type[Exception]) -> list[tuple[str, Exception]]:
        """Get exceptions of a specific type.

        Parameters
        ----------
        exc_type : type
            The exception type to filter by.

        Returns
        -------
        list[tuple[str, Exception]]
            Exceptions matching the type with their contexts.
        """
        return [(ctx, exc) for ctx, exc in self._exceptions if isinstance(exc, exc_type)]

    def get_summary(self) -> str:
        """Get a summary of all recorded errors.

        Returns
        -------
        str
            Human-readable summary of errors.
        """
        if not self._exceptions:
            return "No errors recorded"

        # Group by exception type
        by_type: dict[str, int] = {}
        for _, exc in self._exceptions:
            exc_type = type(exc).__name__
            by_type[exc_type] = by_type.get(exc_type, 0) + 1

        most_common = max(by_type.items(), key=lambda x: x[1])
        type_summary = ", ".join(f"{t}:{c}" for t, c in sorted(by_type.items(), key=lambda x: -x[1]))

        return f"{self.error_count} errors ({type_summary}). Most common: {most_common[0]} ({most_common[1]}x)"

    def get_details(self) -> str:
        """Get detailed information about all errors.

        Returns
        -------
        str
            Detailed error information with contexts.
        """
        if not self._exceptions:
            return "No errors recorded"

        lines = [f"Total errors: {self.error_count}", ""]
        for i, (context, exc) in enumerate(self._exceptions, 1):
            lines.append(f"{i}. [{context}] {type(exc).__name__}: {exc}")

        return "\n".join(lines)

    def raise_if_errors(self, message: str = "Operation failed") -> None:
        """Raise RagitError if any errors were recorded.

        Parameters
        ----------
        message : str
            Base message for the raised error.

        Raises
        ------
        RagitError
            If any errors were recorded.
        """
        if self.has_errors:
            raise RagitError(f"{message}: {self.get_summary()}")

    def clear(self) -> None:
        """Clear all recorded exceptions."""
        self._exceptions.clear()

    def merge_from(self, other: "ExceptionAggregator") -> None:
        """Merge exceptions from another aggregator.

        Parameters
        ----------
        other : ExceptionAggregator
            Another aggregator to merge from.
        """
        self._exceptions.extend(other._exceptions)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for JSON serialization.

        Returns
        -------
        dict
            Dictionary representation of aggregated errors.
        """
        return {
            "error_count": self.error_count,
            "errors": [
                {"context": ctx, "type": type(exc).__name__, "message": str(exc)} for ctx, exc in self._exceptions
            ],
        }
