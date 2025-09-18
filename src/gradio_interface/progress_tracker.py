"""
Progress tracking utilities for Gradio integration.

This module provides decorators and context managers for tracking
operation progress in the Felix Framework when deployed via Gradio.

Key Features:
- Thread-safe progress tracking
- Nested operation support
- Time tracking and estimation
- Memory-efficient implementation
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import deque
import functools


@dataclass
class ProgressState:
    """State for a single progress operation."""
    operation_id: str
    start_time: float
    current_progress: float = 0.0
    message: str = ""
    sub_operations: List[str] = field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def estimated_remaining(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        if self.current_progress <= 0 or self.current_progress >= 100:
            return None

        elapsed = self.elapsed_time
        rate = self.current_progress / elapsed if elapsed > 0 else 0

        if rate > 0:
            remaining = (100 - self.current_progress) / rate
            return remaining

        return None


class ProgressTracker:
    """
    Thread-safe progress tracker for Gradio operations.

    Provides context managers and decorators for tracking
    operation progress with support for nested operations.
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize progress tracker.

        Args:
            max_history: Maximum number of completed operations to track
        """
        self.max_history = max_history
        self._lock = threading.Lock()
        self._active_operations: Dict[str, ProgressState] = {}
        self._history: deque = deque(maxlen=max_history)
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable[[str, float, str], None]):
        """
        Register a progress callback.

        Args:
            callback: Function that receives (operation_id, progress, message)
        """
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """Remove a progress callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    @contextmanager
    def track_operation(self, operation_id: str):
        """
        Context manager for tracking an operation.

        Usage:
            with tracker.track_operation("my_operation") as op:
                op.update(50, "Half way done")
                # ... do work ...
                op.update(100, "Complete")
        """
        # Start operation
        state = self._start_operation(operation_id)

        try:
            yield state
            # Mark as completed if not already
            if not state.completed:
                state.completed = True
                state.current_progress = 100.0
                state.message = "Completed"
                self._notify_callbacks(operation_id, 100.0, "Completed")

        except Exception as e:
            # Mark as error
            state.error = str(e)
            state.completed = True
            self._notify_callbacks(operation_id, state.current_progress, f"Error: {e}")
            raise

        finally:
            # Move to history
            self._complete_operation(operation_id)

    def _start_operation(self, operation_id: str) -> ProgressState:
        """Start tracking an operation."""
        with self._lock:
            state = ProgressState(
                operation_id=operation_id,
                start_time=time.time()
            )
            self._active_operations[operation_id] = state
            return state

    def _complete_operation(self, operation_id: str):
        """Move operation to history."""
        with self._lock:
            if operation_id in self._active_operations:
                state = self._active_operations.pop(operation_id)
                self._history.append(state)

    def _notify_callbacks(self, operation_id: str, progress: float, message: str):
        """Notify registered callbacks of progress update."""
        callbacks = list(self._callbacks)  # Copy to avoid lock during callback
        for callback in callbacks:
            try:
                callback(operation_id, progress, message)
            except Exception:
                pass  # Ignore callback errors

    def update(self, operation_id: str, progress: float, message: str = ""):
        """
        Update progress for an operation.

        Args:
            operation_id: Operation identifier
            progress: Progress percentage (0-100)
            message: Status message
        """
        with self._lock:
            if operation_id in self._active_operations:
                state = self._active_operations[operation_id]
                state.current_progress = min(100.0, max(0.0, progress))
                state.message = message
                self._notify_callbacks(operation_id, state.current_progress, message)

    def get_progress(self, operation_id: str) -> Optional[ProgressState]:
        """Get current progress for an operation."""
        with self._lock:
            return self._active_operations.get(operation_id)

    def get_all_active(self) -> Dict[str, ProgressState]:
        """Get all active operations."""
        with self._lock:
            return dict(self._active_operations)

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            completed_times = [
                op.elapsed_time for op in self._history
                if op.completed and not op.error
            ]

            error_count = sum(1 for op in self._history if op.error)

            stats = {
                "active_operations": len(self._active_operations),
                "completed_operations": len(self._history),
                "error_count": error_count,
                "average_time": sum(completed_times) / len(completed_times) if completed_times else 0,
                "total_operations": len(self._history) + len(self._active_operations)
            }

            return stats


def track_progress(operation_name: str = None):
    """
    Decorator for tracking function progress.

    Usage:
        @track_progress("data_processing")
        def process_data(data, progress_tracker=None):
            # progress_tracker is automatically injected
            progress_tracker.update(50, "Processing...")
            # ... do work ...
            return result

    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func):
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a progress tracker if not provided
            tracker = kwargs.pop('progress_tracker', None)
            if tracker is None:
                tracker = ProgressTracker()

            # Track the operation
            with tracker.track_operation(op_name) as state:
                # Inject the state's update method
                kwargs['progress_update'] = lambda p, m: tracker.update(op_name, p, m)
                return func(*args, **kwargs)

        return wrapper
    return decorator


class GradioProgressAdapter:
    """
    Adapter to bridge Felix progress tracking with Gradio's progress API.

    This adapter converts Felix progress updates to Gradio progress calls.
    """

    def __init__(self, gradio_progress: Optional[Callable] = None):
        """
        Initialize adapter.

        Args:
            gradio_progress: Gradio progress callback (gr.Progress)
        """
        self.gradio_progress = gradio_progress
        self.tracker = ProgressTracker()

        # Register callback to forward to Gradio
        if gradio_progress:
            self.tracker.register_callback(self._forward_to_gradio)

    def _forward_to_gradio(self, operation_id: str, progress: float, message: str):
        """Forward progress updates to Gradio."""
        if self.gradio_progress:
            # Convert percentage to 0-1 range for Gradio
            self.gradio_progress(progress / 100.0, desc=f"{operation_id}: {message}")

    @contextmanager
    def track(self, operation_id: str):
        """
        Track an operation with Gradio progress forwarding.

        Usage:
            with adapter.track("processing") as op:
                op.update(50, "Half done")
        """
        with self.tracker.track_operation(operation_id) as state:
            # Create an update wrapper that's easier to use
            class OperationHandle:
                def update(self, progress: float, message: str = ""):
                    self.tracker.update(operation_id, progress, message)

                @property
                def elapsed_time(self):
                    return state.elapsed_time

                @property
                def estimated_remaining(self):
                    return state.estimated_remaining

            handle = OperationHandle()
            handle.tracker = self.tracker
            yield handle


def create_progress_bar(tracker: ProgressTracker, operation_id: str) -> str:
    """
    Create a text-based progress bar for display.

    Args:
        tracker: Progress tracker instance
        operation_id: Operation to display

    Returns:
        Text representation of progress bar
    """
    state = tracker.get_progress(operation_id)
    if not state:
        return "No operation found"

    progress = state.current_progress
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = '█' * filled + '░' * (bar_length - filled)

    text = f"[{bar}] {progress:.1f}%"

    if state.message:
        text += f" - {state.message}"

    if state.elapsed_time > 0:
        text += f" ({state.elapsed_time:.1f}s)"

    if state.estimated_remaining:
        text += f" - ETA: {state.estimated_remaining:.1f}s"

    return text