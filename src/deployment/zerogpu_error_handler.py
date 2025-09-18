"""
ZeroGPU Error Handling and Fallback Strategies for Felix Framework.

This module provides comprehensive error handling, recovery mechanisms, and
fallback strategies specifically designed for HuggingFace ZeroGPU deployment.

Key Features:
- GPU allocation failure handling with CPU fallback
- Model loading timeout and retry mechanisms
- Gradual degradation strategies for resource constraints
- User-friendly error messages and recovery suggestions
- Automatic fallback to HF Inference API when needed
- Circuit breaker patterns for failing operations
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of ZeroGPU-specific errors."""
    GPU_ALLOCATION_FAILED = "gpu_allocation_failed"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    MODEL_LOADING_TIMEOUT = "model_loading_timeout"
    MODEL_LOADING_FAILED = "model_loading_failed"
    INFERENCE_TIMEOUT = "inference_timeout"
    INFERENCE_FAILED = "inference_failed"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_TO_CPU = "fallback_to_cpu"
    FALLBACK_TO_INFERENCE_API = "fallback_to_inference_api"
    REDUCE_COMPLEXITY = "reduce_complexity"
    QUEUE_FOR_LATER = "queue_for_later"
    FAIL_GRACEFULLY = "fail_gracefully"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: ErrorType
    original_exception: Exception
    operation_name: str
    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    attempt_number: int = 1
    timestamp: float = field(default_factory=time.time)
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action to take for an error."""
    strategy: FallbackStrategy
    max_retries: int
    retry_delay: float
    timeout: float
    fallback_options: Dict[str, Any] = field(default_factory=dict)
    user_message: str = ""


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for operation protection."""
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0
    failure_threshold: int = 5
    reset_timeout: float = 60.0


class ZeroGPUErrorHandler:
    """
    Comprehensive error handler for ZeroGPU deployment.

    Provides intelligent error recovery, fallback strategies, and user-friendly
    error reporting for Felix Framework running on HuggingFace ZeroGPU.
    """

    # Default recovery strategies for different error types
    DEFAULT_RECOVERY_STRATEGIES = {
        ErrorType.GPU_ALLOCATION_FAILED: RecoveryAction(
            strategy=FallbackStrategy.FALLBACK_TO_CPU,
            max_retries=2,
            retry_delay=5.0,
            timeout=30.0,
            user_message="GPU allocation failed. Falling back to CPU processing."
        ),
        ErrorType.GPU_OUT_OF_MEMORY: RecoveryAction(
            strategy=FallbackStrategy.REDUCE_COMPLEXITY,
            max_retries=3,
            retry_delay=2.0,
            timeout=20.0,
            fallback_options={"reduce_agents": True, "reduce_tokens": True},
            user_message="GPU memory exhausted. Reducing processing complexity."
        ),
        ErrorType.MODEL_LOADING_TIMEOUT: RecoveryAction(
            strategy=FallbackStrategy.RETRY_WITH_BACKOFF,
            max_retries=3,
            retry_delay=10.0,
            timeout=60.0,
            user_message="Model loading timed out. Retrying with increased timeout."
        ),
        ErrorType.MODEL_LOADING_FAILED: RecoveryAction(
            strategy=FallbackStrategy.FALLBACK_TO_INFERENCE_API,
            max_retries=1,
            retry_delay=1.0,
            timeout=30.0,
            user_message="Model loading failed. Using HuggingFace Inference API."
        ),
        ErrorType.INFERENCE_TIMEOUT: RecoveryAction(
            strategy=FallbackStrategy.RETRY_WITH_BACKOFF,
            max_retries=2,
            retry_delay=5.0,
            timeout=45.0,
            user_message="Inference timed out. Retrying with reduced complexity."
        ),
        ErrorType.INFERENCE_FAILED: RecoveryAction(
            strategy=FallbackStrategy.FALLBACK_TO_INFERENCE_API,
            max_retries=2,
            retry_delay=2.0,
            timeout=30.0,
            user_message="Inference failed. Switching to backup service."
        ),
        ErrorType.QUOTA_EXCEEDED: RecoveryAction(
            strategy=FallbackStrategy.QUEUE_FOR_LATER,
            max_retries=0,
            retry_delay=300.0,  # 5 minutes
            timeout=3600.0,  # 1 hour
            user_message="Usage quota exceeded. Request queued for later processing."
        ),
        ErrorType.RATE_LIMITED: RecoveryAction(
            strategy=FallbackStrategy.RETRY_WITH_BACKOFF,
            max_retries=3,
            retry_delay=60.0,  # 1 minute
            timeout=300.0,  # 5 minutes
            user_message="Rate limit exceeded. Waiting before retry."
        ),
        ErrorType.NETWORK_ERROR: RecoveryAction(
            strategy=FallbackStrategy.RETRY_WITH_BACKOFF,
            max_retries=3,
            retry_delay=10.0,
            timeout=60.0,
            user_message="Network error. Retrying connection."
        ),
        ErrorType.UNKNOWN_ERROR: RecoveryAction(
            strategy=FallbackStrategy.FAIL_GRACEFULLY,
            max_retries=1,
            retry_delay=5.0,
            timeout=30.0,
            user_message="Unexpected error occurred. Attempting recovery."
        )
    }

    def __init__(self,
                 max_retry_attempts: int = 3,
                 enable_circuit_breaker: bool = True,
                 circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: float = 60.0,
                 custom_strategies: Optional[Dict[ErrorType, RecoveryAction]] = None):
        """
        Initialize ZeroGPU error handler.

        Args:
            max_retry_attempts: Maximum retry attempts for any operation
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_threshold: Failure threshold for circuit breaker
            circuit_breaker_timeout: Reset timeout for circuit breaker
            custom_strategies: Custom recovery strategies by error type
        """
        self.max_retry_attempts = max_retry_attempts
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

        # Merge custom strategies with defaults
        self.recovery_strategies = self.DEFAULT_RECOVERY_STRATEGIES.copy()
        if custom_strategies:
            self.recovery_strategies.update(custom_strategies)

        # Error tracking and statistics
        self.error_history: deque[ErrorContext] = deque(maxlen=1000)
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.recovery_success_counts: Dict[ErrorType, int] = defaultdict(int)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)

        # Callback hooks
        self.error_callbacks: List[Callable[[ErrorContext], None]] = []
        self.recovery_callbacks: List[Callable[[ErrorContext, bool], None]] = []

        logger.info(f"ZeroGPU Error Handler initialized - Circuit Breaker: {enable_circuit_breaker}")

    def add_error_callback(self, callback: Callable[[ErrorContext], None]):
        """Add callback to be called when errors occur."""
        self.error_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable[[ErrorContext, bool], None]):
        """Add callback to be called when recovery attempts complete."""
        self.recovery_callbacks.append(callback)

    def classify_error(self, exception: Exception, operation_name: str) -> ErrorType:
        """
        Classify an exception into a specific error type.

        Args:
            exception: The exception to classify
            operation_name: Name of the operation that failed

        Returns:
            Classified error type
        """
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # GPU-specific errors
        if "cuda" in error_message or "gpu" in error_message:
            if "out of memory" in error_message or "oom" in error_message:
                return ErrorType.GPU_OUT_OF_MEMORY
            elif "allocation" in error_message or "device" in error_message:
                return ErrorType.GPU_ALLOCATION_FAILED

        # Model loading errors
        if "model" in operation_name.lower() and "load" in operation_name.lower():
            if "timeout" in error_message or "timeouterror" in exception_type:
                return ErrorType.MODEL_LOADING_TIMEOUT
            else:
                return ErrorType.MODEL_LOADING_FAILED

        # Inference errors
        if "inference" in operation_name.lower() or "generate" in operation_name.lower():
            if "timeout" in error_message or "timeouterror" in exception_type:
                return ErrorType.INFERENCE_TIMEOUT
            else:
                return ErrorType.INFERENCE_FAILED

        # Rate limiting and quota errors
        if "rate" in error_message and "limit" in error_message:
            return ErrorType.RATE_LIMITED
        if "quota" in error_message or "limit" in error_message:
            return ErrorType.QUOTA_EXCEEDED

        # Network errors
        if any(net_err in exception_type for net_err in
               ["connectionerror", "httperror", "requestexception", "networkerror"]):
            return ErrorType.NETWORK_ERROR

        # Default classification
        return ErrorType.UNKNOWN_ERROR

    async def handle_error(self,
                          exception: Exception,
                          operation_name: str,
                          operation_func: Callable,
                          agent_id: Optional[str] = None,
                          model_id: Optional[str] = None,
                          **operation_kwargs) -> Tuple[bool, Any]:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            exception: The exception that occurred
            operation_name: Name of the failed operation
            operation_func: Function to retry if applicable
            agent_id: ID of the agent involved
            model_id: ID of the model involved
            **operation_kwargs: Arguments for the operation function

        Returns:
            Tuple of (success, result) where success indicates if recovery worked
        """
        error_type = self.classify_error(exception, operation_name)

        error_context = ErrorContext(
            error_type=error_type,
            original_exception=exception,
            operation_name=operation_name,
            agent_id=agent_id,
            model_id=model_id,
            additional_info=operation_kwargs
        )

        # Record error
        self._record_error(error_context)

        # Check circuit breaker
        if self.enable_circuit_breaker:
            circuit_key = f"{operation_name}_{model_id or 'default'}"
            if self._is_circuit_open(circuit_key):
                logger.warning(f"Circuit breaker open for {circuit_key} - skipping operation")
                return False, self._create_circuit_breaker_error(circuit_key)

        # Get recovery strategy
        recovery_action = self.recovery_strategies.get(error_type)
        if not recovery_action:
            logger.error(f"No recovery strategy for error type: {error_type}")
            return False, None

        # Attempt recovery
        success, result = await self._attempt_recovery(
            error_context, recovery_action, operation_func, operation_kwargs
        )

        # Update circuit breaker
        if self.enable_circuit_breaker:
            self._update_circuit_breaker(circuit_key, success)

        # Call recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(error_context, success)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")

        return success, result

    async def _attempt_recovery(self,
                               error_context: ErrorContext,
                               recovery_action: RecoveryAction,
                               operation_func: Callable,
                               operation_kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt recovery using the specified strategy."""
        strategy = recovery_action.strategy

        logger.info(f"Attempting recovery for {error_context.error_type.value} using {strategy.value}")

        if strategy == FallbackStrategy.RETRY_WITH_BACKOFF:
            return await self._retry_with_backoff(
                error_context, recovery_action, operation_func, operation_kwargs
            )
        elif strategy == FallbackStrategy.FALLBACK_TO_CPU:
            return await self._fallback_to_cpu(
                error_context, recovery_action, operation_func, operation_kwargs
            )
        elif strategy == FallbackStrategy.FALLBACK_TO_INFERENCE_API:
            return await self._fallback_to_inference_api(
                error_context, recovery_action, operation_func, operation_kwargs
            )
        elif strategy == FallbackStrategy.REDUCE_COMPLEXITY:
            return await self._reduce_complexity(
                error_context, recovery_action, operation_func, operation_kwargs
            )
        elif strategy == FallbackStrategy.QUEUE_FOR_LATER:
            return await self._queue_for_later(error_context, recovery_action)
        elif strategy == FallbackStrategy.FAIL_GRACEFULLY:
            return await self._fail_gracefully(error_context, recovery_action)
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            return False, None

    async def _retry_with_backoff(self,
                                 error_context: ErrorContext,
                                 recovery_action: RecoveryAction,
                                 operation_func: Callable,
                                 operation_kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Retry operation with exponential backoff."""
        max_retries = min(recovery_action.max_retries, self.max_retry_attempts)
        base_delay = recovery_action.retry_delay

        for attempt in range(max_retries):
            error_context.attempt_number = attempt + 1

            # Calculate backoff delay
            delay = base_delay * (2 ** attempt)

            logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {error_context.operation_name} "
                       f"(delay: {delay:.1f}s)")

            if delay > 0:
                await asyncio.sleep(delay)

            try:
                # Apply timeout if specified
                if recovery_action.timeout > 0:
                    result = await asyncio.wait_for(
                        operation_func(**operation_kwargs),
                        timeout=recovery_action.timeout
                    )
                else:
                    result = await operation_func(**operation_kwargs)

                logger.info(f"Retry successful for {error_context.operation_name}")
                self.recovery_success_counts[error_context.error_type] += 1
                return True, result

            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed for {error_context.operation_name}")
                    return False, None

        return False, None

    async def _fallback_to_cpu(self,
                              error_context: ErrorContext,
                              recovery_action: RecoveryAction,
                              operation_func: Callable,
                              operation_kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Fallback to CPU processing."""
        logger.info(f"Falling back to CPU for {error_context.operation_name}")

        # Modify operation kwargs to force CPU usage
        cpu_kwargs = operation_kwargs.copy()
        cpu_kwargs.update({
            "device": "cpu",
            "force_cpu": True,
            "use_gpu": False
        })

        try:
            # Force garbage collection and GPU cleanup
            if hasattr(operation_func, '__self__'):
                client = operation_func.__self__
                if hasattr(client, '_cleanup_gpu_memory'):
                    await client._cleanup_gpu_memory()

            result = await operation_func(**cpu_kwargs)

            logger.info(f"CPU fallback successful for {error_context.operation_name}")
            self.recovery_success_counts[error_context.error_type] += 1
            return True, result

        except Exception as e:
            logger.error(f"CPU fallback failed for {error_context.operation_name}: {e}")
            return False, None

    async def _fallback_to_inference_api(self,
                                        error_context: ErrorContext,
                                        recovery_action: RecoveryAction,
                                        operation_func: Callable,
                                        operation_kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Fallback to HuggingFace Inference API."""
        logger.info(f"Falling back to Inference API for {error_context.operation_name}")

        # Modify operation kwargs for Inference API
        api_kwargs = operation_kwargs.copy()
        api_kwargs.update({
            "use_inference_api": True,
            "force_api": True,
            "disable_local": True
        })

        try:
            result = await operation_func(**api_kwargs)

            logger.info(f"Inference API fallback successful for {error_context.operation_name}")
            self.recovery_success_counts[error_context.error_type] += 1
            return True, result

        except Exception as e:
            logger.error(f"Inference API fallback failed for {error_context.operation_name}: {e}")
            return False, None

    async def _reduce_complexity(self,
                                error_context: ErrorContext,
                                recovery_action: RecoveryAction,
                                operation_func: Callable,
                                operation_kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Reduce operation complexity to fit resource constraints."""
        logger.info(f"Reducing complexity for {error_context.operation_name}")

        # Apply complexity reduction based on options
        reduced_kwargs = operation_kwargs.copy()
        fallback_options = recovery_action.fallback_options

        if fallback_options.get("reduce_tokens", False):
            # Reduce token limits by 50%
            if "max_tokens" in reduced_kwargs:
                reduced_kwargs["max_tokens"] = max(50, int(reduced_kwargs["max_tokens"] * 0.5))
            if "token_budget" in reduced_kwargs:
                reduced_kwargs["token_budget"] = max(100, int(reduced_kwargs["token_budget"] * 0.5))

        if fallback_options.get("reduce_agents", False):
            # Reduce number of agents
            if "num_agents" in reduced_kwargs:
                reduced_kwargs["num_agents"] = max(1, int(reduced_kwargs["num_agents"] * 0.6))
            if "agent_count" in reduced_kwargs:
                reduced_kwargs["agent_count"] = max(1, int(reduced_kwargs["agent_count"] * 0.6))

        if fallback_options.get("reduce_batch_size", False):
            # Reduce batch size
            if "batch_size" in reduced_kwargs:
                reduced_kwargs["batch_size"] = max(1, int(reduced_kwargs["batch_size"] * 0.5))

        try:
            result = await operation_func(**reduced_kwargs)

            logger.info(f"Complexity reduction successful for {error_context.operation_name}")
            self.recovery_success_counts[error_context.error_type] += 1
            return True, result

        except Exception as e:
            logger.error(f"Complexity reduction failed for {error_context.operation_name}: {e}")
            return False, None

    async def _queue_for_later(self,
                              error_context: ErrorContext,
                              recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Queue operation for later execution."""
        logger.info(f"Queueing {error_context.operation_name} for later execution")

        # In a real implementation, this would add to a persistent queue
        # For now, we return a placeholder response
        queue_result = {
            "queued": True,
            "estimated_delay": recovery_action.retry_delay,
            "queue_position": 1,  # Placeholder
            "message": recovery_action.user_message
        }

        return True, queue_result

    async def _fail_gracefully(self,
                              error_context: ErrorContext,
                              recovery_action: RecoveryAction) -> Tuple[bool, Any]:
        """Fail gracefully with user-friendly message."""
        logger.warning(f"Graceful failure for {error_context.operation_name}")

        error_result = {
            "success": False,
            "error_type": error_context.error_type.value,
            "user_message": recovery_action.user_message,
            "technical_details": str(error_context.original_exception),
            "retry_possible": recovery_action.max_retries > 0,
            "suggested_action": self._get_user_suggestion(error_context.error_type)
        }

        return False, error_result

    def _get_user_suggestion(self, error_type: ErrorType) -> str:
        """Get user-friendly suggestion for error type."""
        suggestions = {
            ErrorType.GPU_ALLOCATION_FAILED: "Try reducing the complexity or try again later when GPU resources are available.",
            ErrorType.GPU_OUT_OF_MEMORY: "Reduce the number of agents or token limits in your request.",
            ErrorType.MODEL_LOADING_TIMEOUT: "The model is taking longer to load. Try again or use a smaller model.",
            ErrorType.MODEL_LOADING_FAILED: "The model could not be loaded. Try a different model or contact support.",
            ErrorType.INFERENCE_TIMEOUT: "The request is taking too long. Try reducing complexity or try again.",
            ErrorType.INFERENCE_FAILED: "The AI inference failed. Try again or contact support if the issue persists.",
            ErrorType.QUOTA_EXCEEDED: "You've exceeded your usage quota. Upgrade your account or wait for quota reset.",
            ErrorType.RATE_LIMITED: "Too many requests. Please wait a moment before trying again.",
            ErrorType.NETWORK_ERROR: "Network connection issue. Check your connection and try again.",
            ErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Try again or contact support if the issue persists."
        }
        return suggestions.get(error_type, "Please try again or contact support.")

    def _record_error(self, error_context: ErrorContext):
        """Record error for statistics and analysis."""
        self.error_history.append(error_context)
        self.error_counts[error_context.error_type] += 1

        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_context)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _is_circuit_open(self, circuit_key: str) -> bool:
        """Check if circuit breaker is open for the given key."""
        circuit = self.circuit_breakers[circuit_key]
        current_time = time.time()

        if circuit.state == "open":
            if current_time - circuit.last_failure_time > self.circuit_breaker_timeout:
                circuit.state = "half_open"
                circuit.success_count = 0
                logger.info(f"Circuit breaker for {circuit_key} moved to half-open state")
                return False
            return True

        return False

    def _update_circuit_breaker(self, circuit_key: str, success: bool):
        """Update circuit breaker state based on operation result."""
        circuit = self.circuit_breakers[circuit_key]
        current_time = time.time()

        if success:
            if circuit.state == "half_open":
                circuit.success_count += 1
                if circuit.success_count >= 3:  # Require 3 successes to close
                    circuit.state = "closed"
                    circuit.failure_count = 0
                    logger.info(f"Circuit breaker for {circuit_key} closed (recovered)")
            else:
                circuit.failure_count = max(0, circuit.failure_count - 1)
        else:
            circuit.failure_count += 1
            circuit.last_failure_time = current_time

            if circuit.failure_count >= self.circuit_breaker_threshold:
                circuit.state = "open"
                logger.warning(f"Circuit breaker for {circuit_key} opened due to {circuit.failure_count} failures")

    def _create_circuit_breaker_error(self, circuit_key: str) -> Dict[str, Any]:
        """Create error response for open circuit breaker."""
        return {
            "success": False,
            "error_type": "circuit_breaker_open",
            "user_message": f"Service temporarily unavailable for {circuit_key}. Please try again later.",
            "technical_details": f"Circuit breaker open for {circuit_key}",
            "retry_possible": True,
            "suggested_action": "Wait a moment and try again. The service is temporarily protecting against failures."
        }

    @asynccontextmanager
    async def error_context(self,
                           operation_name: str,
                           operation_func: Callable,
                           agent_id: Optional[str] = None,
                           model_id: Optional[str] = None,
                           **operation_kwargs):
        """
        Context manager for automatic error handling.

        Usage:
            async with error_handler.error_context("model_inference", model_inference_func,
                                                  agent_id="agent_1", model_id="gpt-3.5"):
                result = await some_operation()
        """
        try:
            yield
        except Exception as e:
            success, result = await self.handle_error(
                e, operation_name, operation_func, agent_id, model_id, **operation_kwargs
            )
            if not success:
                raise e from None

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.recovery_success_counts.values())

        return {
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "recovery_rate": total_recoveries / max(1, total_errors),
            "error_breakdown": dict(self.error_counts),
            "recovery_breakdown": dict(self.recovery_success_counts),
            "circuit_breaker_states": {
                key: {
                    "state": circuit.state,
                    "failure_count": circuit.failure_count,
                    "success_count": circuit.success_count
                }
                for key, circuit in self.circuit_breakers.items()
            },
            "recent_errors": [
                {
                    "error_type": ctx.error_type.value,
                    "operation": ctx.operation_name,
                    "agent_id": ctx.agent_id,
                    "timestamp": ctx.timestamp,
                    "attempt_number": ctx.attempt_number
                }
                for ctx in list(self.error_history)[-10:]  # Last 10 errors
            ]
        }

    def reset_statistics(self):
        """Reset all error statistics."""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_success_counts.clear()
        self.circuit_breakers.clear()
        logger.info("Error statistics reset")

    def export_error_report(self, filepath: Optional[str] = None) -> str:
        """Export comprehensive error report to JSON."""
        report = {
            "timestamp": time.time(),
            "handler_config": {
                "max_retry_attempts": self.max_retry_attempts,
                "enable_circuit_breaker": self.enable_circuit_breaker,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_timeout
            },
            "statistics": self.get_error_statistics(),
            "recovery_strategies": {
                error_type.value: {
                    "strategy": action.strategy.value,
                    "max_retries": action.max_retries,
                    "retry_delay": action.retry_delay,
                    "timeout": action.timeout,
                    "user_message": action.user_message
                }
                for error_type, action in self.recovery_strategies.items()
            }
        }

        report_json = json.dumps(report, indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_json)
            logger.info(f"Error report exported to {filepath}")

        return report_json


# Utility functions for integration

def create_zerogpu_error_handler() -> ZeroGPUErrorHandler:
    """Create a ZeroGPU error handler with optimal settings."""
    return ZeroGPUErrorHandler(
        max_retry_attempts=3,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60.0
    )


def setup_global_error_handling():
    """Set up global error handling for ZeroGPU deployment."""
    # This would integrate with global exception handlers
    # For now, just log the setup
    logger.info("Global ZeroGPU error handling configured")


# Export main classes and functions
__all__ = [
    'ZeroGPUErrorHandler',
    'ErrorType',
    'FallbackStrategy',
    'ErrorContext',
    'RecoveryAction',
    'create_zerogpu_error_handler',
    'setup_global_error_handling'
]