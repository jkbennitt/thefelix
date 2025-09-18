"""
Batch Processing Optimization for Felix Framework on ZeroGPU.

This module provides intelligent batch processing capabilities specifically optimized
for ZeroGPU deployment, maximizing GPU utilization while respecting memory constraints
and maintaining the helix-based coordination architecture.

Key Features:
- Agent task batching with helix-aware grouping
- Dynamic batch size adjustment based on GPU memory
- Parallel processing with efficient GPU resource sharing
- Priority-based scheduling for different agent types
- Memory-aware batch composition and execution
- Real-time performance optimization and monitoring
"""

import asyncio
import logging
import time
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies for different scenarios."""
    MEMORY_OPTIMIZED = "memory_optimized"  # Prioritize memory efficiency
    THROUGHPUT_OPTIMIZED = "throughput_optimized"  # Maximize processing speed
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize response time
    ADAPTIVE = "adaptive"  # Adapt based on current conditions


class AgentPriority(Enum):
    """Priority levels for agent batch processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BatchTask:
    """Individual task within a batch."""
    task_id: str
    agent_id: str
    agent_type: str
    prompt: str
    priority: AgentPriority
    estimated_tokens: int
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    model_preference: Optional[str] = None
    timeout: float = 30.0
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchRequest:
    """Batch of tasks for GPU processing."""
    batch_id: str
    tasks: List[BatchTask]
    total_estimated_tokens: int
    max_memory_mb: float
    strategy: BatchStrategy
    priority_score: float
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None


@dataclass
class BatchResult:
    """Result from batch processing."""
    batch_id: str
    task_results: Dict[str, Any]  # task_id -> result
    processing_time: float
    gpu_memory_used: float
    tokens_processed: int
    success_rate: float
    errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUResourceState:
    """Current GPU resource utilization state."""
    memory_used_mb: float
    memory_total_mb: float
    utilization_percent: float
    temperature: float
    active_models: List[str]
    last_update: float = field(default_factory=time.time)


class ZeroGPUBatchOptimizer:
    """
    Intelligent batch processor for Felix Framework on ZeroGPU.

    Optimizes agent task processing through strategic batching, memory management,
    and parallel execution while maintaining helix-based coordination principles.
    """

    # Configuration constants
    DEFAULT_BATCH_SIZE = 4  # Conservative for ZeroGPU
    MAX_BATCH_SIZE = 8  # Maximum batch size for ZeroGPU
    MIN_BATCH_SIZE = 1  # Minimum viable batch
    MEMORY_SAFETY_MARGIN = 0.1  # 10% memory safety margin
    BATCH_TIMEOUT = 5.0  # Maximum wait time for batch assembly
    PRIORITY_BOOST_FACTOR = 1.5  # Priority multiplier for high-priority tasks

    def __init__(self,
                 gpu_monitor=None,
                 default_strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
                 max_batch_size: int = MAX_BATCH_SIZE,
                 batch_timeout: float = BATCH_TIMEOUT,
                 memory_threshold: float = 0.8,
                 enable_dynamic_sizing: bool = True,
                 max_concurrent_batches: int = 2):
        """
        Initialize batch optimizer.

        Args:
            gpu_monitor: GPU monitoring instance for resource tracking
            default_strategy: Default batching strategy
            max_batch_size: Maximum tasks per batch
            batch_timeout: Maximum wait time for batch assembly
            memory_threshold: GPU memory threshold for batch sizing
            enable_dynamic_sizing: Enable dynamic batch size adjustment
            max_concurrent_batches: Maximum concurrent batch processing
        """
        self.gpu_monitor = gpu_monitor
        self.default_strategy = default_strategy
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.memory_threshold = memory_threshold
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.max_concurrent_batches = max_concurrent_batches

        # Task and batch management
        self.pending_tasks: deque[BatchTask] = deque()
        self.priority_queue: List[Tuple[float, BatchTask]] = []  # Priority heap
        self.active_batches: Dict[str, BatchRequest] = {}
        self.completed_batches: deque[BatchResult] = deque(maxlen=100)

        # Resource tracking
        self.gpu_state = GPUResourceState(0.0, 0.0, 0.0, 0.0, [])
        self.current_memory_usage = 0.0
        self.model_memory_estimates = defaultdict(lambda: 1000.0)  # MB per model

        # Performance metrics
        self.total_tasks_processed = 0
        self.total_batches_processed = 0
        self.average_batch_time = 0.0
        self.memory_efficiency_history = deque(maxlen=50)
        self.throughput_history = deque(maxlen=50)

        # Processing control
        self.is_processing = False
        self.processor_task: Optional[asyncio.Task] = None
        self.batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        logger.info(f"ZeroGPU Batch Optimizer initialized - Strategy: {default_strategy.value}, "
                   f"Max Batch Size: {max_batch_size}")

    async def start_processing(self):
        """Start the batch processing engine."""
        if self.is_processing:
            return

        self.is_processing = True
        self.processor_task = asyncio.create_task(self._batch_processing_loop())
        logger.info("Batch processing started")

    async def stop_processing(self):
        """Stop the batch processing engine."""
        self.is_processing = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        self.thread_pool.shutdown(wait=True)
        logger.info("Batch processing stopped")

    async def submit_task(self,
                         task_id: str,
                         agent_id: str,
                         agent_type: str,
                         prompt: str,
                         priority: AgentPriority = AgentPriority.NORMAL,
                         estimated_tokens: int = 100,
                         **kwargs) -> str:
        """
        Submit a task for batch processing.

        Args:
            task_id: Unique task identifier
            agent_id: Agent submitting the task
            agent_type: Type of agent (research, analysis, synthesis, critic)
            prompt: Task prompt/input
            priority: Task priority level
            estimated_tokens: Estimated token count for resource planning
            **kwargs: Additional task parameters

        Returns:
            Task ID for tracking
        """
        task = BatchTask(
            task_id=task_id,
            agent_id=agent_id,
            agent_type=agent_type,
            prompt=prompt,
            priority=priority,
            estimated_tokens=estimated_tokens,
            **kwargs
        )

        # Add to appropriate queue based on priority
        if priority in [AgentPriority.HIGH, AgentPriority.CRITICAL]:
            priority_score = priority.value * self.PRIORITY_BOOST_FACTOR
            heapq.heappush(self.priority_queue, (-priority_score, task))
        else:
            self.pending_tasks.append(task)

        logger.debug(f"Task {task_id} submitted for agent {agent_id} with priority {priority.value}")
        return task_id

    async def _batch_processing_loop(self):
        """Main batch processing loop."""
        while self.is_processing:
            try:
                # Update GPU resource state
                await self._update_gpu_state()

                # Assemble batch if conditions are met
                batch = await self._assemble_batch()

                if batch and len(batch.tasks) > 0:
                    # Process batch asynchronously
                    asyncio.create_task(self._process_batch(batch))

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1.0)

    async def _update_gpu_state(self):
        """Update current GPU resource state."""
        if self.gpu_monitor:
            try:
                status = self.gpu_monitor.get_resource_status()
                gpu_info = status.get("gpu", {})

                self.gpu_state = GPUResourceState(
                    memory_used_mb=gpu_info.get("memory_mb", {}).get("reserved", 0.0),
                    memory_total_mb=gpu_info.get("memory_mb", {}).get("total", 16000.0),  # Default assumption
                    utilization_percent=gpu_info.get("utilization_percent", 0.0),
                    temperature=0.0,  # Not typically available
                    active_models=status.get("active", {}).get("model_list", [])
                )

            except Exception as e:
                logger.warning(f"Failed to update GPU state: {e}")

    async def _assemble_batch(self) -> Optional[BatchRequest]:
        """Assemble tasks into an optimal batch."""
        if not self.pending_tasks and not self.priority_queue:
            return None

        # Start with high-priority tasks
        selected_tasks = []
        total_tokens = 0

        # Add priority tasks first
        while self.priority_queue and len(selected_tasks) < self.max_batch_size:
            _, task = heapq.heappop(self.priority_queue)
            if self._can_add_to_batch(task, selected_tasks, total_tokens):
                selected_tasks.append(task)
                total_tokens += task.estimated_tokens

        # Fill remaining slots with normal priority tasks
        while self.pending_tasks and len(selected_tasks) < self.max_batch_size:
            task = self.pending_tasks.popleft()
            if self._can_add_to_batch(task, selected_tasks, total_tokens):
                selected_tasks.append(task)
                total_tokens += task.estimated_tokens
            else:
                # Return task to queue if it doesn't fit
                self.pending_tasks.appendleft(task)
                break

        # Create batch if we have tasks
        if selected_tasks:
            return await self._create_batch_request(selected_tasks, total_tokens)

        return None

    def _can_add_to_batch(self, task: BatchTask, current_tasks: List[BatchTask], current_tokens: int) -> bool:
        """Check if task can be added to current batch."""
        # Check batch size limit
        if len(current_tasks) >= self.max_batch_size:
            return False

        # Check memory constraints
        estimated_memory = self._estimate_memory_usage(current_tasks + [task])
        available_memory = self.gpu_state.memory_total_mb * self.memory_threshold
        if estimated_memory > available_memory:
            return False

        # Check token budget
        total_tokens = current_tokens + task.estimated_tokens
        if total_tokens > 8000:  # Conservative token limit for batch
            return False

        # Check model compatibility (prefer same or compatible models)
        if current_tasks:
            current_types = set(t.agent_type for t in current_tasks)
            if len(current_types) > 2:  # Limit model diversity in batch
                return False

        return True

    def _estimate_memory_usage(self, tasks: List[BatchTask]) -> float:
        """Estimate GPU memory usage for a batch of tasks."""
        # Base memory overhead
        base_memory = 500.0  # MB

        # Model memory
        unique_models = set()
        for task in tasks:
            model_id = task.model_preference or f"default_{task.agent_type}"
            unique_models.add(model_id)

        model_memory = sum(self.model_memory_estimates[model] for model in unique_models)

        # Task processing memory (proportional to tokens)
        total_tokens = sum(task.estimated_tokens for task in tasks)
        task_memory = total_tokens * 0.1  # 0.1 MB per token estimate

        # Batch processing overhead
        batch_overhead = len(tasks) * 50.0  # 50 MB per task in batch

        return base_memory + model_memory + task_memory + batch_overhead

    async def _create_batch_request(self, tasks: List[BatchTask], total_tokens: int) -> BatchRequest:
        """Create a batch request from selected tasks."""
        batch_id = f"batch_{int(time.time() * 1000)}"

        # Calculate priority score (average of task priorities)
        avg_priority = sum(task.priority.value for task in tasks) / len(tasks)

        # Estimate memory requirements
        estimated_memory = self._estimate_memory_usage(tasks)

        # Determine strategy
        strategy = await self._select_batch_strategy(tasks, estimated_memory)

        return BatchRequest(
            batch_id=batch_id,
            tasks=tasks,
            total_estimated_tokens=total_tokens,
            max_memory_mb=estimated_memory,
            strategy=strategy,
            priority_score=avg_priority
        )

    async def _select_batch_strategy(self, tasks: List[BatchTask], estimated_memory: float) -> BatchStrategy:
        """Select optimal batch processing strategy."""
        if self.default_strategy != BatchStrategy.ADAPTIVE:
            return self.default_strategy

        # Adaptive strategy selection
        gpu_memory_ratio = estimated_memory / self.gpu_state.memory_total_mb
        gpu_utilization = self.gpu_state.utilization_percent / 100.0

        # High memory usage -> memory optimized
        if gpu_memory_ratio > 0.7:
            return BatchStrategy.MEMORY_OPTIMIZED

        # High priority tasks -> latency optimized
        if any(task.priority == AgentPriority.CRITICAL for task in tasks):
            return BatchStrategy.LATENCY_OPTIMIZED

        # High GPU utilization -> throughput optimized
        if gpu_utilization > 0.6:
            return BatchStrategy.THROUGHPUT_OPTIMIZED

        # Default to memory optimized for ZeroGPU
        return BatchStrategy.MEMORY_OPTIMIZED

    async def _process_batch(self, batch: BatchRequest):
        """Process a batch of tasks."""
        async with self.batch_semaphore:
            start_time = time.time()
            batch_id = batch.batch_id

            logger.info(f"Processing batch {batch_id} with {len(batch.tasks)} tasks "
                       f"(strategy: {batch.strategy.value})")

            try:
                # Add to active batches
                self.active_batches[batch_id] = batch

                # Apply pre-processing optimizations
                await self._optimize_batch_for_strategy(batch)

                # Process tasks based on strategy
                if batch.strategy == BatchStrategy.MEMORY_OPTIMIZED:
                    result = await self._process_memory_optimized(batch)
                elif batch.strategy == BatchStrategy.THROUGHPUT_OPTIMIZED:
                    result = await self._process_throughput_optimized(batch)
                elif batch.strategy == BatchStrategy.LATENCY_OPTIMIZED:
                    result = await self._process_latency_optimized(batch)
                else:
                    result = await self._process_default(batch)

                # Record performance metrics
                processing_time = time.time() - start_time
                await self._record_batch_performance(batch, result, processing_time)

                logger.info(f"Batch {batch_id} completed in {processing_time:.2f}s "
                           f"(success rate: {result.success_rate:.1%})")

            except Exception as e:
                logger.error(f"Batch {batch_id} processing failed: {e}")
                # Create error result
                result = BatchResult(
                    batch_id=batch_id,
                    task_results={},
                    processing_time=time.time() - start_time,
                    gpu_memory_used=0.0,
                    tokens_processed=0,
                    success_rate=0.0,
                    errors=[{"error": str(e), "timestamp": time.time()}]
                )

            finally:
                # Clean up
                self.active_batches.pop(batch_id, None)
                self.completed_batches.append(result)

                # Trigger cleanup if memory is high
                if self.gpu_state.memory_used_mb > self.gpu_state.memory_total_mb * 0.8:
                    await self._cleanup_gpu_memory()

    async def _optimize_batch_for_strategy(self, batch: BatchRequest):
        """Apply strategy-specific optimizations."""
        if batch.strategy == BatchStrategy.MEMORY_OPTIMIZED:
            # Sort tasks by estimated memory usage (smallest first)
            batch.tasks.sort(key=lambda t: t.estimated_tokens)

        elif batch.strategy == BatchStrategy.LATENCY_OPTIMIZED:
            # Sort by priority (highest first)
            batch.tasks.sort(key=lambda t: t.priority.value, reverse=True)

        elif batch.strategy == BatchStrategy.THROUGHPUT_OPTIMIZED:
            # Group by agent type for model efficiency
            batch.tasks.sort(key=lambda t: t.agent_type)

    async def _process_memory_optimized(self, batch: BatchRequest) -> BatchResult:
        """Process batch with memory optimization priority."""
        results = {}
        total_tokens = 0
        successful_tasks = 0
        errors = []

        # Process tasks sequentially to minimize memory usage
        for task in batch.tasks:
            try:
                # Check memory before processing
                if self.gpu_state.memory_used_mb > self.gpu_state.memory_total_mb * 0.9:
                    await self._cleanup_gpu_memory()

                # Process single task
                result = await self._process_single_task(task)
                results[task.task_id] = result
                total_tokens += task.estimated_tokens
                successful_tasks += 1

                # Clear intermediate results to save memory
                if hasattr(result, 'intermediate_data'):
                    delattr(result, 'intermediate_data')

            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                errors.append({
                    "task_id": task.task_id,
                    "error": str(e),
                    "timestamp": time.time()
                })

        return BatchResult(
            batch_id=batch.batch_id,
            task_results=results,
            processing_time=0.0,  # Will be set by caller
            gpu_memory_used=self.gpu_state.memory_used_mb,
            tokens_processed=total_tokens,
            success_rate=successful_tasks / len(batch.tasks),
            errors=errors
        )

    async def _process_throughput_optimized(self, batch: BatchRequest) -> BatchResult:
        """Process batch with throughput optimization priority."""
        results = {}
        total_tokens = 0
        successful_tasks = 0
        errors = []

        # Group tasks by agent type for parallel processing
        task_groups = defaultdict(list)
        for task in batch.tasks:
            task_groups[task.agent_type].append(task)

        # Process groups concurrently
        group_tasks = []
        for agent_type, tasks in task_groups.items():
            group_task = asyncio.create_task(
                self._process_task_group(tasks, f"group_{agent_type}")
            )
            group_tasks.append(group_task)

        # Await all groups
        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

        # Aggregate results
        for group_result in group_results:
            if isinstance(group_result, Exception):
                errors.append({
                    "error": str(group_result),
                    "timestamp": time.time()
                })
            else:
                results.update(group_result.get("results", {}))
                total_tokens += group_result.get("tokens", 0)
                successful_tasks += group_result.get("successful", 0)

        return BatchResult(
            batch_id=batch.batch_id,
            task_results=results,
            processing_time=0.0,
            gpu_memory_used=self.gpu_state.memory_used_mb,
            tokens_processed=total_tokens,
            success_rate=successful_tasks / len(batch.tasks),
            errors=errors
        )

    async def _process_latency_optimized(self, batch: BatchRequest) -> BatchResult:
        """Process batch with latency optimization priority."""
        # Process highest priority tasks first, with immediate execution
        results = {}
        total_tokens = 0
        successful_tasks = 0
        errors = []

        # Sort by priority
        priority_tasks = sorted(batch.tasks, key=lambda t: t.priority.value, reverse=True)

        # Process with adaptive concurrency based on priority
        for i, task in enumerate(priority_tasks):
            try:
                # Higher priority tasks get immediate processing
                if task.priority in [AgentPriority.CRITICAL, AgentPriority.HIGH]:
                    result = await self._process_single_task(task)
                else:
                    # Lower priority tasks can be batched
                    remaining_tasks = priority_tasks[i:]
                    if len(remaining_tasks) > 1:
                        group_result = await self._process_task_group(remaining_tasks[:3], "low_priority_group")
                        results.update(group_result.get("results", {}))
                        total_tokens += group_result.get("tokens", 0)
                        successful_tasks += group_result.get("successful", 0)
                        break
                    else:
                        result = await self._process_single_task(task)

                results[task.task_id] = result
                total_tokens += task.estimated_tokens
                successful_tasks += 1

            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                errors.append({
                    "task_id": task.task_id,
                    "error": str(e),
                    "timestamp": time.time()
                })

        return BatchResult(
            batch_id=batch.batch_id,
            task_results=results,
            processing_time=0.0,
            gpu_memory_used=self.gpu_state.memory_used_mb,
            tokens_processed=total_tokens,
            success_rate=successful_tasks / len(batch.tasks),
            errors=errors
        )

    async def _process_default(self, batch: BatchRequest) -> BatchResult:
        """Default batch processing strategy."""
        return await self._process_memory_optimized(batch)

    async def _process_single_task(self, task: BatchTask) -> Dict[str, Any]:
        """Process a single task."""
        # This would integrate with the actual LLM client
        # For now, return a mock result
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "task_id": task.task_id,
            "content": f"Processed result for {task.agent_type} agent",
            "tokens_used": task.estimated_tokens,
            "success": True,
            "timestamp": time.time()
        }

    async def _process_task_group(self, tasks: List[BatchTask], group_name: str) -> Dict[str, Any]:
        """Process a group of tasks concurrently."""
        results = {}
        total_tokens = 0
        successful_tasks = 0

        # Process tasks concurrently within the group
        task_futures = [self._process_single_task(task) for task in tasks]
        task_results = await asyncio.gather(*task_futures, return_exceptions=True)

        for task, result in zip(tasks, task_results):
            if isinstance(result, Exception):
                logger.error(f"Task {task.task_id} in group {group_name} failed: {result}")
            else:
                results[task.task_id] = result
                total_tokens += task.estimated_tokens
                successful_tasks += 1

        return {
            "results": results,
            "tokens": total_tokens,
            "successful": successful_tasks
        }

    async def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if hasattr(self, 'gpu_monitor') and self.gpu_monitor:
            # Use monitor's cleanup if available
            logger.info("Triggering GPU memory cleanup")
            # Would call gpu_monitor._emergency_memory_cleanup()
        else:
            # Basic cleanup
            gc.collect()

    async def _record_batch_performance(self, batch: BatchRequest, result: BatchResult, processing_time: float):
        """Record batch performance metrics."""
        result.processing_time = processing_time

        # Update global statistics
        self.total_tasks_processed += len(batch.tasks)
        self.total_batches_processed += 1

        # Update average batch time
        self.average_batch_time = (
            (self.average_batch_time * (self.total_batches_processed - 1) + processing_time)
            / self.total_batches_processed
        )

        # Memory efficiency
        memory_efficiency = result.tokens_processed / max(1.0, result.gpu_memory_used)
        self.memory_efficiency_history.append(memory_efficiency)

        # Throughput
        throughput = len(batch.tasks) / processing_time
        self.throughput_history.append(throughput)

        # Log performance summary
        logger.info(f"Batch performance - Tasks: {len(batch.tasks)}, "
                   f"Time: {processing_time:.2f}s, "
                   f"Memory: {result.gpu_memory_used:.0f}MB, "
                   f"Throughput: {throughput:.1f} tasks/s")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "total_batches_processed": self.total_batches_processed,
            "average_batch_time": self.average_batch_time,
            "current_queue_size": len(self.pending_tasks) + len(self.priority_queue),
            "active_batches": len(self.active_batches),
            "memory_efficiency": {
                "current": self.memory_efficiency_history[-1] if self.memory_efficiency_history else 0.0,
                "average": sum(self.memory_efficiency_history) / max(1, len(self.memory_efficiency_history)),
                "history_size": len(self.memory_efficiency_history)
            },
            "throughput": {
                "current": self.throughput_history[-1] if self.throughput_history else 0.0,
                "average": sum(self.throughput_history) / max(1, len(self.throughput_history)),
                "peak": max(self.throughput_history) if self.throughput_history else 0.0
            },
            "gpu_state": {
                "memory_used_mb": self.gpu_state.memory_used_mb,
                "memory_total_mb": self.gpu_state.memory_total_mb,
                "memory_utilization": self.gpu_state.memory_used_mb / max(1.0, self.gpu_state.memory_total_mb),
                "gpu_utilization": self.gpu_state.utilization_percent
            }
        }

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        priority_tasks = len(self.priority_queue)
        normal_tasks = len(self.pending_tasks)

        return {
            "total_queued": priority_tasks + normal_tasks,
            "priority_queue": priority_tasks,
            "normal_queue": normal_tasks,
            "active_batches": len(self.active_batches),
            "processing": self.is_processing,
            "estimated_wait_time": self._estimate_wait_time()
        }

    def _estimate_wait_time(self) -> float:
        """Estimate wait time for new tasks."""
        if not self.throughput_history:
            return 10.0  # Default estimate

        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        total_queued = len(self.pending_tasks) + len(self.priority_queue)

        return total_queued / max(0.1, avg_throughput)


# Utility functions

def create_zerogpu_batch_optimizer(gpu_monitor=None,
                                 strategy: BatchStrategy = BatchStrategy.ADAPTIVE) -> ZeroGPUBatchOptimizer:
    """Create a ZeroGPU batch optimizer with optimal settings."""
    return ZeroGPUBatchOptimizer(
        gpu_monitor=gpu_monitor,
        default_strategy=strategy,
        max_batch_size=6,  # Conservative for ZeroGPU
        batch_timeout=3.0,  # Quick batching for responsiveness
        memory_threshold=0.75,  # Conservative memory usage
        enable_dynamic_sizing=True,
        max_concurrent_batches=2  # Limit concurrency for ZeroGPU
    )


# Export main classes and functions
__all__ = [
    'ZeroGPUBatchOptimizer',
    'BatchTask',
    'BatchRequest',
    'BatchResult',
    'BatchStrategy',
    'AgentPriority',
    'GPUResourceState',
    'create_zerogpu_batch_optimizer'
]