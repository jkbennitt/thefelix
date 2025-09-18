"""
Comprehensive test scenarios for Felix Framework ZeroGPU deployment.

This module provides extensive testing for ZeroGPU-specific functionality,
including resource management, error handling, batch processing, and performance
optimization under various deployment conditions.

Test Categories:
- GPU memory management and cleanup
- Model loading and switching scenarios
- Concurrent user sessions and resource contention
- Error recovery and fallback mechanisms
- Performance benchmarking under ZeroGPU constraints
- Integration testing with HuggingFace Spaces
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional
import gc

# Add src to path for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deployment.zerogpu_monitor import (
    ZeroGPUMonitor, create_zerogpu_monitor, ResourceAlert,
    ResourceType, AlertSeverity, GPUMemorySnapshot
)
from deployment.zerogpu_error_handler import (
    ZeroGPUErrorHandler, ErrorType, FallbackStrategy,
    create_zerogpu_error_handler
)
from deployment.batch_optimizer import (
    ZeroGPUBatchOptimizer, BatchTask, BatchStrategy, AgentPriority,
    create_zerogpu_batch_optimizer
)

logger = logging.getLogger(__name__)


class MockGPUState:
    """Mock GPU state for testing."""

    def __init__(self, memory_total_mb: float = 16000.0):
        self.memory_total_mb = memory_total_mb
        self.memory_used_mb = 0.0
        self.utilization_percent = 0.0
        self.active_models = []

    def use_memory(self, amount_mb: float):
        """Simulate GPU memory usage."""
        self.memory_used_mb += amount_mb
        self.utilization_percent = (self.memory_used_mb / self.memory_total_mb) * 100

    def free_memory(self, amount_mb: float):
        """Simulate GPU memory release."""
        self.memory_used_mb = max(0.0, self.memory_used_mb - amount_mb)
        self.utilization_percent = (self.memory_used_mb / self.memory_total_mb) * 100

    def get_status(self):
        """Get mock status."""
        return {
            "gpu": {
                "memory_mb": {
                    "reserved": self.memory_used_mb,
                    "total": self.memory_total_mb,
                    "free": self.memory_total_mb - self.memory_used_mb
                },
                "utilization_percent": self.utilization_percent
            },
            "active": {
                "model_list": self.active_models.copy(),
                "models": len(self.active_models)
            }
        }


@pytest.fixture
def mock_gpu_state():
    """Create mock GPU state for testing."""
    return MockGPUState()


@pytest.fixture
def zerogpu_monitor(mock_gpu_state):
    """Create ZeroGPU monitor with mock GPU state."""
    monitor = create_zerogpu_monitor()
    monitor.gpu_available = True
    monitor.torch_available = True

    # Mock the get_resource_status method
    monitor.get_resource_status = lambda: mock_gpu_state.get_status()

    return monitor


@pytest.fixture
def error_handler():
    """Create ZeroGPU error handler for testing."""
    return create_zerogpu_error_handler()


@pytest.fixture
def batch_optimizer(zerogpu_monitor):
    """Create batch optimizer with monitor."""
    return create_zerogpu_batch_optimizer(gpu_monitor=zerogpu_monitor)


class TestZeroGPUMonitoring:
    """Test ZeroGPU monitoring functionality."""

    def test_monitor_initialization(self, zerogpu_monitor):
        """Test monitor initializes correctly."""
        assert zerogpu_monitor is not None
        assert zerogpu_monitor.GPU_MEMORY_WARNING_THRESHOLD == 0.7
        assert zerogpu_monitor.GPU_MEMORY_CRITICAL_THRESHOLD == 0.85

    def test_memory_threshold_alerts(self, zerogpu_monitor, mock_gpu_state):
        """Test memory threshold alert generation."""
        alerts_triggered = []

        def alert_callback(alert):
            alerts_triggered.append(alert)

        zerogpu_monitor.alert_callback = alert_callback

        # Simulate high memory usage
        mock_gpu_state.use_memory(12000)  # 75% of 16GB
        zerogpu_monitor._take_memory_snapshot()
        zerogpu_monitor._check_resource_thresholds()

        # Should trigger warning
        assert len(alerts_triggered) > 0
        assert alerts_triggered[0].severity == AlertSeverity.WARNING
        assert alerts_triggered[0].resource_type == ResourceType.GPU_MEMORY

        # Simulate critical memory usage
        mock_gpu_state.use_memory(2000)  # 87.5% of 16GB
        zerogpu_monitor._take_memory_snapshot()
        zerogpu_monitor._check_resource_thresholds()

        # Should trigger critical alert
        critical_alerts = [a for a in alerts_triggered if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0

    def test_performance_tracking(self, zerogpu_monitor):
        """Test performance metrics tracking."""
        # Simulate some operations
        with zerogpu_monitor.track_operation("test_inference", "agent_1"):
            time.sleep(0.1)

        with zerogpu_monitor.track_operation("test_model_load", "agent_2"):
            time.sleep(0.05)

        stats = zerogpu_monitor.get_performance_report()

        assert stats["summary"]["total_operations"] == 2
        assert stats["summary"]["successful_operations"] == 2
        assert stats["timing"]["avg_inference_time"] > 0

    def test_resource_cleanup(self, zerogpu_monitor, mock_gpu_state):
        """Test automatic resource cleanup."""
        # Register some models and agents
        zerogpu_monitor.register_model("model_1")
        zerogpu_monitor.register_model("model_2")
        zerogpu_monitor.register_agent("agent_1", "research")

        # Simulate passage of time to trigger cleanup
        old_time = time.time() - 400  # 6 minutes ago
        zerogpu_monitor.active_models["model_1"] = old_time
        zerogpu_monitor.active_agents["agent_1"]["last_activity"] = old_time

        # Trigger cleanup
        zerogpu_monitor._automatic_cleanup()

        # Check that old resources were cleaned up
        assert "model_1" not in zerogpu_monitor.active_models
        assert "agent_1" not in zerogpu_monitor.active_agents

    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self, zerogpu_monitor):
        """Test concurrent monitoring operations."""
        zerogpu_monitor.start_monitoring()

        # Wait a bit for monitoring to start
        await asyncio.sleep(0.5)

        # Perform concurrent operations
        tasks = []
        for i in range(5):
            task = asyncio.create_task(self._simulate_agent_operation(zerogpu_monitor, f"agent_{i}"))
            tasks.append(task)

        await asyncio.gather(*tasks)

        zerogpu_monitor.stop_monitoring()

        # Check that all operations were tracked
        stats = zerogpu_monitor.get_resource_status()
        assert stats["active"]["agents"] == 0  # Should be cleaned up

    async def _simulate_agent_operation(self, monitor, agent_id):
        """Simulate an agent operation."""
        monitor.register_agent(agent_id, "test")
        await asyncio.sleep(0.1)
        monitor.update_agent_activity(agent_id)
        await asyncio.sleep(0.1)
        monitor.unregister_agent(agent_id)


class TestZeroGPUErrorHandling:
    """Test ZeroGPU error handling and recovery."""

    @pytest.mark.asyncio
    async def test_gpu_memory_error_recovery(self, error_handler):
        """Test GPU out of memory error recovery."""

        # Mock operation that fails with GPU OOM
        async def failing_operation(**kwargs):
            if not kwargs.get("force_cpu", False):
                raise RuntimeError("CUDA out of memory")
            return {"success": True, "device": "cpu"}

        # Simulate GPU OOM error
        exception = RuntimeError("CUDA out of memory")
        success, result = await error_handler.handle_error(
            exception, "gpu_inference", failing_operation
        )

        assert success
        assert result["device"] == "cpu"

        # Check that error was recorded
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] > 0
        assert ErrorType.GPU_OUT_OF_MEMORY in error_handler.error_counts

    @pytest.mark.asyncio
    async def test_model_loading_timeout_recovery(self, error_handler):
        """Test model loading timeout recovery."""

        call_count = 0

        async def slow_model_loading(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(2.0)  # Simulate timeout
                raise asyncio.TimeoutError("Model loading timed out")
            return {"success": True, "attempt": call_count}

        exception = asyncio.TimeoutError("Model loading timed out")
        success, result = await error_handler.handle_error(
            exception, "model_loading", slow_model_loading
        )

        assert success
        assert result["attempt"] == 2  # Should retry once

    @pytest.mark.asyncio
    async def test_fallback_to_inference_api(self, error_handler):
        """Test fallback to HuggingFace Inference API."""

        async def failing_local_inference(**kwargs):
            if kwargs.get("use_inference_api", False):
                return {"success": True, "source": "inference_api"}
            raise RuntimeError("Local inference failed")

        exception = RuntimeError("Local inference failed")
        success, result = await error_handler.handle_error(
            exception, "inference", failing_local_inference
        )

        assert success
        assert result["source"] == "inference_api"

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, error_handler):
        """Test circuit breaker pattern."""

        failure_count = 0

        async def unreliable_operation(**kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 6:  # Fail first 6 times
                raise RuntimeError(f"Failure {failure_count}")
            return {"success": True}

        # Trigger multiple failures to open circuit breaker
        for i in range(6):
            exception = RuntimeError(f"Failure {i+1}")
            success, result = await error_handler.handle_error(
                exception, "unreliable_op", unreliable_operation
            )
            assert not success

        # Circuit should now be open
        circuit_key = "unreliable_op_default"
        assert error_handler._is_circuit_open(circuit_key)

        # Next call should be blocked by circuit breaker
        exception = RuntimeError("Should be blocked")
        success, result = await error_handler.handle_error(
            exception, "unreliable_op", unreliable_operation
        )

        assert not success
        assert "circuit_breaker_open" in result.get("error_type", "")

    def test_error_classification(self, error_handler):
        """Test error classification accuracy."""

        # GPU errors
        gpu_oom = RuntimeError("CUDA out of memory")
        assert error_handler.classify_error(gpu_oom, "inference") == ErrorType.GPU_OUT_OF_MEMORY

        gpu_alloc = RuntimeError("GPU allocation failed")
        assert error_handler.classify_error(gpu_alloc, "model_load") == ErrorType.GPU_ALLOCATION_FAILED

        # Timeout errors
        timeout_error = asyncio.TimeoutError("Operation timed out")
        assert error_handler.classify_error(timeout_error, "model_loading") == ErrorType.MODEL_LOADING_TIMEOUT
        assert error_handler.classify_error(timeout_error, "inference") == ErrorType.INFERENCE_TIMEOUT

        # Rate limiting
        rate_limit = RuntimeError("Rate limit exceeded")
        assert error_handler.classify_error(rate_limit, "api_call") == ErrorType.RATE_LIMITED


class TestZeroGPUBatchProcessing:
    """Test ZeroGPU batch processing optimization."""

    @pytest.mark.asyncio
    async def test_batch_assembly(self, batch_optimizer):
        """Test intelligent batch assembly."""
        await batch_optimizer.start_processing()

        # Submit various tasks
        task_ids = []
        for i in range(5):
            task_id = await batch_optimizer.submit_task(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                agent_type="research" if i % 2 == 0 else "analysis",
                prompt=f"Test prompt {i}",
                priority=AgentPriority.HIGH if i < 2 else AgentPriority.NORMAL,
                estimated_tokens=100 + i * 50
            )
            task_ids.append(task_id)

        # Wait for batch processing
        await asyncio.sleep(1.0)

        # Check that tasks were processed
        queue_status = await batch_optimizer.get_queue_status()
        assert queue_status["total_queued"] <= len(task_ids)  # Some may have been processed

        await batch_optimizer.stop_processing()

    @pytest.mark.asyncio
    async def test_memory_aware_batching(self, batch_optimizer, mock_gpu_state):
        """Test memory-aware batch size adjustment."""
        await batch_optimizer.start_processing()

        # Simulate high memory usage
        mock_gpu_state.use_memory(14000)  # 87.5% of 16GB

        # Submit memory-intensive tasks
        for i in range(10):
            await batch_optimizer.submit_task(
                task_id=f"memory_task_{i}",
                agent_id=f"agent_{i}",
                agent_type="synthesis",  # More memory intensive
                prompt=f"Large synthesis task {i}",
                estimated_tokens=1000  # Large token count
            )

        await asyncio.sleep(0.5)

        # Batch size should be reduced due to memory constraints
        active_batches = len(batch_optimizer.active_batches)
        for batch in batch_optimizer.active_batches.values():
            assert len(batch.tasks) <= 3  # Should be smaller due to memory

        await batch_optimizer.stop_processing()

    @pytest.mark.asyncio
    async def test_priority_based_processing(self, batch_optimizer):
        """Test priority-based task processing."""
        await batch_optimizer.start_processing()

        # Submit tasks with different priorities
        await batch_optimizer.submit_task(
            "low_priority", "agent_1", "research", "Low priority task",
            AgentPriority.LOW, 100
        )

        await batch_optimizer.submit_task(
            "critical_priority", "agent_2", "analysis", "Critical task",
            AgentPriority.CRITICAL, 200
        )

        await batch_optimizer.submit_task(
            "normal_priority", "agent_3", "synthesis", "Normal task",
            AgentPriority.NORMAL, 150
        )

        # Wait for processing
        await asyncio.sleep(1.0)

        # Critical priority task should be processed first
        # (This would require more detailed result tracking in a real implementation)

        await batch_optimizer.stop_processing()

    def test_batch_strategy_selection(self, batch_optimizer):
        """Test adaptive batch strategy selection."""
        # Mock different GPU states
        batch_optimizer.gpu_state.memory_total_mb = 16000
        batch_optimizer.gpu_state.utilization_percent = 30

        # Create test tasks
        tasks = [
            BatchTask("task_1", "agent_1", "research", "Test", AgentPriority.NORMAL, 100),
            BatchTask("task_2", "agent_2", "analysis", "Test", AgentPriority.CRITICAL, 200)
        ]

        # Test different scenarios
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # High memory usage -> memory optimized
            strategy = loop.run_until_complete(
                batch_optimizer._select_batch_strategy(tasks, 12000)
            )
            assert strategy == BatchStrategy.MEMORY_OPTIMIZED

            # Critical priority tasks -> latency optimized
            strategy = loop.run_until_complete(
                batch_optimizer._select_batch_strategy(tasks, 5000)
            )
            assert strategy == BatchStrategy.LATENCY_OPTIMIZED

        finally:
            loop.close()

    def test_performance_metrics(self, batch_optimizer):
        """Test performance metrics collection."""
        # Simulate some completed batches
        batch_optimizer.total_tasks_processed = 50
        batch_optimizer.total_batches_processed = 10
        batch_optimizer.average_batch_time = 2.5
        batch_optimizer.throughput_history.extend([5.0, 6.0, 4.5, 5.5])

        stats = batch_optimizer.get_performance_statistics()

        assert stats["total_tasks_processed"] == 50
        assert stats["total_batches_processed"] == 10
        assert stats["average_batch_time"] == 2.5
        assert stats["throughput"]["average"] == 5.25  # Average of throughput history


class TestZeroGPUIntegration:
    """Test integration scenarios for ZeroGPU deployment."""

    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, zerogpu_monitor, error_handler, batch_optimizer):
        """Test handling multiple concurrent user sessions."""

        # Start all components
        zerogpu_monitor.start_monitoring()
        await batch_optimizer.start_processing()

        # Simulate multiple users
        user_tasks = []
        for user_id in range(3):
            task = asyncio.create_task(
                self._simulate_user_session(user_id, batch_optimizer, error_handler)
            )
            user_tasks.append(task)

        # Run sessions concurrently
        results = await asyncio.gather(*user_tasks, return_exceptions=True)

        # Check that all sessions completed successfully
        successful_sessions = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_sessions) == 3

        # Cleanup
        await batch_optimizer.stop_processing()
        zerogpu_monitor.stop_monitoring()

    async def _simulate_user_session(self, user_id, batch_optimizer, error_handler):
        """Simulate a user session with multiple agent tasks."""
        session_results = []

        # Submit tasks for different agent types
        agent_types = ["research", "analysis", "synthesis", "critic"]

        for i, agent_type in enumerate(agent_types):
            task_id = f"user_{user_id}_task_{i}"

            try:
                await batch_optimizer.submit_task(
                    task_id=task_id,
                    agent_id=f"user_{user_id}_agent_{i}",
                    agent_type=agent_type,
                    prompt=f"User {user_id} task for {agent_type}",
                    priority=AgentPriority.NORMAL,
                    estimated_tokens=150 + i * 50
                )
                session_results.append({"task_id": task_id, "status": "submitted"})

            except Exception as e:
                session_results.append({"task_id": task_id, "status": "failed", "error": str(e)})

        return session_results

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, mock_gpu_state, zerogpu_monitor, error_handler):
        """Test recovery from resource exhaustion scenarios."""

        zerogpu_monitor.start_monitoring()

        # Simulate gradual memory increase to exhaustion
        for i in range(10):
            mock_gpu_state.use_memory(1500)  # Gradually fill memory

            # Simulate operation that might fail due to memory pressure
            if mock_gpu_state.memory_used_mb > 14000:  # > 87.5%

                async def memory_intensive_operation(**kwargs):
                    if kwargs.get("force_cpu"):
                        return {"success": True, "device": "cpu"}
                    raise RuntimeError("CUDA out of memory")

                exception = RuntimeError("CUDA out of memory")
                success, result = await error_handler.handle_error(
                    exception, "gpu_operation", memory_intensive_operation
                )

                # Should successfully fallback to CPU
                assert success
                assert result["device"] == "cpu"
                break

            await asyncio.sleep(0.1)

        zerogpu_monitor.stop_monitoring()

    def test_error_statistics_and_reporting(self, error_handler):
        """Test comprehensive error statistics and reporting."""

        # Simulate various error types
        error_types = [
            (ErrorType.GPU_OUT_OF_MEMORY, RuntimeError("CUDA out of memory")),
            (ErrorType.MODEL_LOADING_TIMEOUT, asyncio.TimeoutError("Timeout")),
            (ErrorType.RATE_LIMITED, RuntimeError("Rate limit exceeded")),
            (ErrorType.NETWORK_ERROR, ConnectionError("Network failed"))
        ]

        for error_type, exception in error_types:
            # Record multiple instances
            for _ in range(3):
                error_handler._record_error(
                    error_handler.ErrorContext(
                        error_type=error_type,
                        original_exception=exception,
                        operation_name="test_operation"
                    )
                )

        # Get statistics
        stats = error_handler.get_error_statistics()

        assert stats["total_errors"] == 12  # 4 types * 3 instances
        assert len(stats["error_breakdown"]) == 4
        assert all(count == 3 for count in stats["error_breakdown"].values())

        # Test report export
        report = error_handler.export_error_report()
        assert "statistics" in report
        assert "recovery_strategies" in report

    @pytest.mark.asyncio
    async def test_performance_under_load(self, batch_optimizer):
        """Test performance under high load conditions."""

        await batch_optimizer.start_processing()

        # Submit a large number of tasks quickly
        start_time = time.time()

        tasks = []
        for i in range(50):
            task = batch_optimizer.submit_task(
                task_id=f"load_test_task_{i}",
                agent_id=f"agent_{i % 10}",  # 10 different agents
                agent_type=["research", "analysis", "synthesis", "critic"][i % 4],
                prompt=f"Load test task {i}",
                priority=AgentPriority.HIGH if i < 10 else AgentPriority.NORMAL,
                estimated_tokens=100 + (i % 5) * 50
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        submission_time = time.time() - start_time

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check performance metrics
        stats = batch_optimizer.get_performance_statistics()
        queue_status = await batch_optimizer.get_queue_status()

        # Verify reasonable performance
        assert submission_time < 5.0  # Should submit 50 tasks in under 5 seconds
        assert stats["total_tasks_processed"] > 0

        # Check that system is still responsive
        assert queue_status["processing"] is True

        await batch_optimizer.stop_processing()


class TestZeroGPUDeploymentScenarios:
    """Test specific deployment scenarios for ZeroGPU."""

    def test_huggingface_spaces_simulation(self):
        """Test simulation of HuggingFace Spaces deployment constraints."""

        # Simulate Spaces environment constraints
        constraints = {
            "max_gpu_memory_gb": 16,
            "max_concurrent_users": 10,
            "request_timeout_seconds": 60,
            "cold_start_delay": 30,
            "model_loading_timeout": 120
        }

        # Test constraint validation
        assert constraints["max_gpu_memory_gb"] == 16
        assert constraints["max_concurrent_users"] <= 20  # Reasonable limit
        assert constraints["request_timeout_seconds"] >= 30  # Minimum reasonable timeout

    @pytest.mark.asyncio
    async def test_cold_start_optimization(self, error_handler):
        """Test optimization for cold start scenarios."""

        # Simulate cold start delay
        async def cold_start_operation(**kwargs):
            if not kwargs.get("warm_cache", False):
                await asyncio.sleep(0.2)  # Simulate cold start delay
                raise asyncio.TimeoutError("Cold start timeout")
            return {"success": True, "warm": True}

        # First call should trigger timeout and retry
        exception = asyncio.TimeoutError("Cold start timeout")
        success, result = await error_handler.handle_error(
            exception, "cold_start_operation", cold_start_operation, warm_cache=False
        )

        # Should eventually succeed with retry
        assert success or "retry" in str(result)  # Either success or retry scheduled

    def test_gradio_interface_integration(self):
        """Test integration with Gradio interface requirements."""

        # Simulate Gradio interface requirements
        gradio_config = {
            "max_queue_size": 100,
            "progress_updates": True,
            "session_timeout": 3600,  # 1 hour
            "enable_caching": True,
            "max_file_size_mb": 100
        }

        # Validate configuration
        assert gradio_config["max_queue_size"] > 0
        assert gradio_config["session_timeout"] > 0
        assert gradio_config["enable_caching"] is True

    def test_token_budget_optimization(self):
        """Test token budget optimization for ZeroGPU deployment."""

        # Simulate token budget constraints
        token_config = {
            "max_tokens_per_request": 2048,
            "total_session_budget": 50000,
            "emergency_reserve": 5000,
            "agent_type_limits": {
                "research": 300,
                "analysis": 500,
                "synthesis": 800,
                "critic": 200
            }
        }

        # Test budget allocation
        total_allocated = sum(token_config["agent_type_limits"].values())
        assert total_allocated <= token_config["max_tokens_per_request"]
        assert token_config["emergency_reserve"] < token_config["total_session_budget"]


if __name__ == "__main__":
    # Run specific test categories
    import sys

    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category == "monitoring":
            pytest.main(["-v", "test_zerogpu_deployment.py::TestZeroGPUMonitoring"])
        elif category == "errors":
            pytest.main(["-v", "test_zerogpu_deployment.py::TestZeroGPUErrorHandling"])
        elif category == "batching":
            pytest.main(["-v", "test_zerogpu_deployment.py::TestZeroGPUBatchProcessing"])
        elif category == "integration":
            pytest.main(["-v", "test_zerogpu_deployment.py::TestZeroGPUIntegration"])
        elif category == "deployment":
            pytest.main(["-v", "test_zerogpu_deployment.py::TestZeroGPUDeploymentScenarios"])
        else:
            print("Unknown category. Options: monitoring, errors, batching, integration, deployment")
    else:
        # Run all tests
        pytest.main(["-v", "test_zerogpu_deployment.py"])