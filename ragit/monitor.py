#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Execution monitoring with timing and JSON export.

Pattern inspired by ai4rag experiment_monitor.py.

Provides structured tracking of:
- Pattern execution times (e.g., experiment configurations)
- Step execution times within patterns
- Summary statistics and JSON export
"""

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepTiming:
    """Timing information for a single step."""

    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            **self.metadata,
        }


@dataclass
class PatternTiming:
    """Timing information for a pattern (e.g., experiment configuration)."""

    name: str
    start_time: float
    end_time: float | None = None
    steps: list[StepTiming] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            **self.metadata,
        }


class ExecutionMonitor:
    """
    Monitor experiment execution with timing and export.

    Tracks pattern execution times, step timings within patterns,
    and provides summary statistics and JSON export.

    Pattern from ai4rag experiment_monitor.py.

    Examples
    --------
    >>> monitor = ExecutionMonitor()
    >>> with monitor.pattern("config-1"):
    ...     with monitor.step("indexing", chunk_size=512):
    ...         # Index documents
    ...         pass
    ...     with monitor.step("retrieval", top_k=3):
    ...         # Retrieve results
    ...         pass
    >>> monitor.print_summary()
    >>> monitor.export_json("timing.json")
    """

    def __init__(self) -> None:
        self._patterns: list[PatternTiming] = []
        self._current_pattern: PatternTiming | None = None
        self._current_step: StepTiming | None = None
        self._start_time = time.perf_counter()

    @contextmanager
    def pattern(self, name: str, **metadata: Any) -> Generator[PatternTiming, None, None]:
        """
        Context manager for timing a pattern execution.

        Parameters
        ----------
        name : str
            Pattern name (e.g., configuration identifier).
        **metadata
            Additional metadata to attach to the pattern.

        Yields
        ------
        PatternTiming
            The pattern timing object (can be modified).
        """
        pattern = PatternTiming(name=name, start_time=time.perf_counter(), metadata=metadata)
        self._current_pattern = pattern

        try:
            yield pattern
        finally:
            pattern.end_time = time.perf_counter()
            self._patterns.append(pattern)
            self._current_pattern = None

    @contextmanager
    def step(self, name: str, **metadata: Any) -> Generator[StepTiming, None, None]:
        """
        Context manager for timing a step within a pattern.

        Parameters
        ----------
        name : str
            Step name (e.g., "indexing", "retrieval", "evaluation").
        **metadata
            Additional metadata to attach to the step.

        Yields
        ------
        StepTiming
            The step timing object (can be modified).
        """
        step = StepTiming(name=name, start_time=time.perf_counter(), metadata=metadata)
        self._current_step = step

        try:
            yield step
        finally:
            step.end_time = time.perf_counter()
            if self._current_pattern is not None:
                self._current_pattern.steps.append(step)
            self._current_step = None

    def on_pattern_start(self, pattern_name: str, **metadata: Any) -> None:
        """Manual pattern start (alternative to context manager)."""
        self._current_pattern = PatternTiming(name=pattern_name, start_time=time.perf_counter(), metadata=metadata)

    def on_pattern_finish(self, **metadata: Any) -> None:
        """Manual pattern finish (alternative to context manager)."""
        if self._current_pattern:
            self._current_pattern.end_time = time.perf_counter()
            self._current_pattern.metadata.update(metadata)
            self._patterns.append(self._current_pattern)
            self._current_pattern = None

    def on_step_start(self, step_name: str, **metadata: Any) -> None:
        """Manual step start (alternative to context manager)."""
        self._current_step = StepTiming(name=step_name, start_time=time.perf_counter(), metadata=metadata)

    def on_step_finish(self, **metadata: Any) -> None:
        """Manual step finish (alternative to context manager)."""
        if self._current_step:
            self._current_step.end_time = time.perf_counter()
            self._current_step.metadata.update(metadata)
            if self._current_pattern is not None:
                self._current_pattern.steps.append(self._current_step)
            self._current_step = None

    @property
    def total_duration_ms(self) -> float:
        """Total duration since monitor creation in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000

    @property
    def pattern_count(self) -> int:
        """Number of completed patterns."""
        return len(self._patterns)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics as dictionary.

        Returns
        -------
        dict
            Summary with total duration, pattern count, and pattern details.
        """
        return {
            "total_duration_ms": self.total_duration_ms,
            "pattern_count": self.pattern_count,
            "patterns": [p.to_dict() for p in self._patterns],
        }

    def get_step_aggregates(self) -> dict[str, dict[str, float]]:
        """
        Get aggregated step statistics across all patterns.

        Returns
        -------
        dict
            Step name -> {count, total_ms, avg_ms, min_ms, max_ms}
        """
        step_stats: dict[str, list[float]] = {}

        for pattern in self._patterns:
            for step in pattern.steps:
                if step.duration_ms is not None:
                    if step.name not in step_stats:
                        step_stats[step.name] = []
                    step_stats[step.name].append(step.duration_ms)

        aggregates = {}
        for name, durations in step_stats.items():
            aggregates[name] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
            }

        return aggregates

    def export_json(self, path: Path | str, indent: int = 2) -> None:
        """
        Export monitoring data to JSON file.

        Parameters
        ----------
        path : Path or str
            Output file path.
        indent : int
            JSON indentation (default: 2).
        """
        path = Path(path)
        data = {
            **self.get_summary(),
            "step_aggregates": self.get_step_aggregates(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    def print_summary(self, show_steps: bool = True) -> None:
        """
        Print human-readable summary to console.

        Parameters
        ----------
        show_steps : bool
            Include step-level details (default: True).
        """
        summary = self.get_summary()

        print(f"\n{'=' * 60}")
        print(f"Execution Summary (Total: {summary['total_duration_ms']:.0f}ms)")
        print(f"Patterns: {summary['pattern_count']}")
        print(f"{'=' * 60}")

        for pattern in summary["patterns"]:
            duration = pattern.get("duration_ms")
            duration_str = f"{duration:.0f}ms" if duration else "in progress"
            print(f"\n{pattern['name']}: {duration_str}")

            if show_steps:
                for step in pattern.get("steps", []):
                    step_duration = step.get("duration_ms")
                    step_duration_str = f"{step_duration:.0f}ms" if step_duration else "in progress"
                    # Show first few metadata items
                    meta_items = [(k, v) for k, v in step.items() if k not in ("name", "duration_ms")][:3]
                    meta_str = ", ".join(f"{k}={v}" for k, v in meta_items) if meta_items else ""
                    print(f"  - {step['name']}: {step_duration_str}" + (f" ({meta_str})" if meta_str else ""))

        # Print step aggregates
        aggregates = self.get_step_aggregates()
        if aggregates:
            print(f"\n{'-' * 60}")
            print("Step Aggregates:")
            for name, stats in sorted(aggregates.items(), key=lambda x: -x[1]["total_ms"]):
                print(
                    f"  {name}: {stats['count']}x, total={stats['total_ms']:.0f}ms, "
                    f"avg={stats['avg_ms']:.0f}ms, range=[{stats['min_ms']:.0f}-{stats['max_ms']:.0f}]ms"
                )

    def reset(self) -> None:
        """Reset the monitor, clearing all recorded patterns."""
        self._patterns.clear()
        self._current_pattern = None
        self._current_step = None
        self._start_time = time.perf_counter()
