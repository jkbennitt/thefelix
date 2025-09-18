"""
GPU resource management for ZeroGPU integration.

This module provides GPU resource management specifically designed
for deployment on Hugging Face Spaces with ZeroGPU support.

Key Features:
- Automatic GPU detection and allocation
- Resource limits for fair sharing
- Memory management and cleanup
- Fallback to CPU when GPU unavailable
"""

import os
import time
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUContext:
    """Context for GPU resource allocation."""
    device_id: Optional[int]
    memory_allocated: int
    priority: str
    start_time: float
    is_gpu: bool

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since allocation."""
        return time.time() - self.start_time


class GPUResourceManager:
    """
    Manages GPU resources for ZeroGPU deployment.

    Provides automatic resource management with fallback to CPU
    when GPU is unavailable or quota exceeded.
    """

    # ZeroGPU typical limits
    DEFAULT_TIME_LIMIT = 120  # 2 minutes per request
    DEFAULT_MEMORY_LIMIT = 4096  # 4GB VRAM

    # Priority levels
    PRIORITY_LEVELS = {
        "high": {"time_limit": 180, "memory_limit": 6144},
        "normal": {"time_limit": 120, "memory_limit": 4096},
        "low": {"time_limit": 60, "memory_limit": 2048}
    }

    def __init__(self,
                 enable_gpu: bool = True,
                 auto_cleanup: bool = True,
                 max_concurrent: int = 3):
        """
        Initialize GPU resource manager.

        Args:
            enable_gpu: Whether to attempt GPU usage
            auto_cleanup: Enable automatic resource cleanup
            max_concurrent: Maximum concurrent GPU operations
        """
        self.enable_gpu = enable_gpu and self._check_gpu_available()
        self.auto_cleanup = auto_cleanup
        self.max_concurrent = max_concurrent

        # Resource tracking
        self._lock = threading.Lock()
        self._active_contexts: List[GPUContext] = []
        self._total_memory_used = 0
        self._request_count = 0

        # Check for ZeroGPU environment
        self.is_zerogpu = self._detect_zerogpu_environment()

        if self.is_zerogpu:
            logger.info("ZeroGPU environment detected")
            self._setup_zerogpu()
        elif self.enable_gpu:
            logger.info("GPU available but not ZeroGPU environment")
        else:
            logger.info("Running in CPU-only mode")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _detect_zerogpu_environment(self) -> bool:
        """Detect if running in ZeroGPU environment."""
        # Check for ZeroGPU-specific environment variables
        zerogpu_indicators = [
            "SPACES_ZERO_GPU",
            "ZEROGPU_AVAILABLE",
            "HF_SPACES"
        ]

        for indicator in zerogpu_indicators:
            if os.environ.get(indicator):
                return True

        return False

    def _setup_zerogpu(self):
        """Setup ZeroGPU-specific configurations."""
        try:
            # Import spaces library if available
            import spaces

            # Register with ZeroGPU
            @spaces.GPU
            def dummy_gpu_function():
                pass

            # Test GPU allocation
            dummy_gpu_function()
            logger.info("ZeroGPU setup successful")

        except ImportError:
            logger.warning("spaces library not available, ZeroGPU features limited")
        except Exception as e:
            logger.warning(f"ZeroGPU setup failed: {e}")

    @contextmanager
    def acquire_resources(self, priority: str = "normal",
                         memory_required: int = 2048):
        """
        Context manager for acquiring GPU resources.

        Args:
            priority: Priority level (high/normal/low)
            memory_required: Required memory in MB

        Yields:
            GPUContext object with resource allocation details
        """
        context = self._allocate_resources(priority, memory_required)

        try:
            yield context
        finally:
            self._release_resources(context)

    def _allocate_resources(self, priority: str, memory_required: int) -> GPUContext:
        """Allocate GPU resources."""
        with self._lock:
            # Check if GPU is available and within limits
            if not self.enable_gpu or len(self._active_contexts) >= self.max_concurrent:
                # Fallback to CPU
                context = GPUContext(
                    device_id=None,
                    memory_allocated=0,
                    priority=priority,
                    start_time=time.time(),
                    is_gpu=False
                )
                logger.info(f"Allocated CPU resources (GPU unavailable or limit reached)")
                return context

            # Check memory limits
            limits = self.PRIORITY_LEVELS.get(priority, self.PRIORITY_LEVELS["normal"])

            if memory_required > limits["memory_limit"]:
                memory_required = limits["memory_limit"]
                logger.warning(f"Memory request reduced to {memory_required}MB (priority limit)")

            if self._total_memory_used + memory_required > self.DEFAULT_MEMORY_LIMIT * 2:
                # Total system limit exceeded, use CPU
                context = GPUContext(
                    device_id=None,
                    memory_allocated=0,
                    priority=priority,
                    start_time=time.time(),
                    is_gpu=False
                )
                logger.warning("System memory limit exceeded, falling back to CPU")
                return context

            # Allocate GPU
            device_id = self._get_available_device()
            context = GPUContext(
                device_id=device_id,
                memory_allocated=memory_required,
                priority=priority,
                start_time=time.time(),
                is_gpu=True
            )

            self._active_contexts.append(context)
            self._total_memory_used += memory_required
            self._request_count += 1

            logger.info(f"Allocated GPU {device_id} with {memory_required}MB for {priority} priority")

            # Start cleanup timer if needed
            if self.auto_cleanup:
                self._schedule_cleanup(context, limits["time_limit"])

            return context

    def _release_resources(self, context: GPUContext):
        """Release GPU resources."""
        with self._lock:
            if context in self._active_contexts:
                self._active_contexts.remove(context)
                self._total_memory_used -= context.memory_allocated

                if context.is_gpu:
                    logger.info(f"Released GPU {context.device_id} ({context.memory_allocated}MB)")

                    # Clear GPU cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass

    def _get_available_device(self) -> int:
        """Get available GPU device ID."""
        # For ZeroGPU, typically use device 0
        # In multi-GPU setups, could implement round-robin
        return 0

    def _schedule_cleanup(self, context: GPUContext, timeout: int):
        """Schedule automatic cleanup after timeout."""
        def cleanup():
            time.sleep(timeout)
            with self._lock:
                if context in self._active_contexts:
                    logger.warning(f"Auto-cleaning GPU context after {timeout}s timeout")
                    self._release_resources(context)

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    def get_device_string(self, context: Optional[GPUContext] = None) -> str:
        """
        Get device string for PyTorch/TensorFlow.

        Args:
            context: GPU context (uses CPU if None or not GPU)

        Returns:
            Device string like "cuda:0" or "cpu"
        """
        if context and context.is_gpu and context.device_id is not None:
            return f"cuda:{context.device_id}"
        return "cpu"

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        with self._lock:
            stats = {
                "gpu_enabled": self.enable_gpu,
                "is_zerogpu": self.is_zerogpu,
                "active_contexts": len(self._active_contexts),
                "total_memory_used_mb": self._total_memory_used,
                "total_requests": self._request_count,
                "max_concurrent": self.max_concurrent
            }

            if self._active_contexts:
                stats["active_priorities"] = [c.priority for c in self._active_contexts]
                stats["oldest_context_age_s"] = max(c.elapsed_time for c in self._active_contexts)

            return stats

    def cleanup_all(self):
        """Force cleanup of all GPU resources."""
        with self._lock:
            contexts = list(self._active_contexts)

        for context in contexts:
            self._release_resources(context)

        logger.info("Cleaned up all GPU resources")


def zerogpu_decorator(memory_mb: int = 2048, priority: str = "normal"):
    """
    Decorator for functions that need GPU resources.

    Usage:
        @zerogpu_decorator(memory_mb=4096, priority="high")
        def my_gpu_function(data):
            # Function automatically gets GPU resources
            ...

    Args:
        memory_mb: Required memory in MB
        priority: Priority level
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = GPUResourceManager()

            with manager.acquire_resources(priority=priority,
                                          memory_required=memory_mb) as gpu_ctx:
                # Inject GPU context into kwargs
                kwargs['gpu_context'] = gpu_ctx
                return func(*args, **kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Global manager instance for shared use
_global_gpu_manager: Optional[GPUResourceManager] = None


def get_gpu_manager() -> GPUResourceManager:
    """Get or create global GPU manager instance."""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        _global_gpu_manager = GPUResourceManager()
    return _global_gpu_manager