"""
Unit tests for Gradio adapter components.

This module tests the Felix Framework's Gradio adapter for:
- Thread safety with multiple concurrent users
- Session management and cleanup
- Progress tracking accuracy
- Cache functionality
- GPU resource management
- Error handling and fallbacks
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gradio_interface.felix_gradio_adapter import (
    FelixGradioAdapter, ComplexityLevel, SessionState, ResultCache
)
from gradio_interface.progress_tracker import (
    ProgressTracker, GradioProgressAdapter
)
from gradio_interface.gpu_manager import GPUResourceManager
from gradio_interface.helix_cache import HelixCache


class TestSessionManagement:
    """Test session management functionality."""

    def test_create_session(self):
        """Test session creation."""
        adapter = FelixGradioAdapter(max_sessions=10)
        session_id = adapter.create_session()

        assert session_id is not None
        assert len(session_id) > 0

        session = adapter.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert not session.is_processing

    def test_max_sessions_limit(self):
        """Test maximum sessions enforcement."""
        adapter = FelixGradioAdapter(max_sessions=3, session_timeout=3600)

        # Create maximum sessions
        session_ids = []
        for _ in range(3):
            sid = adapter.create_session()
            session_ids.append(sid)

        # Creating one more should evict the oldest
        new_sid = adapter.create_session()
        assert new_sid is not None

        # Check that we still have max sessions
        assert len(adapter.sessions) == 3

        # Oldest session should be removed
        assert adapter.get_session(session_ids[0]) is None

    def test_session_cleanup_on_timeout(self):
        """Test automatic session cleanup on timeout."""
        adapter = FelixGradioAdapter(session_timeout=0.1)  # 100ms timeout

        session_id = adapter.create_session()
        assert adapter.get_session(session_id) is not None

        # Wait for timeout
        time.sleep(0.2)

        # Create new session to trigger cleanup
        adapter.create_session()

        # Old session should be cleaned up
        assert adapter.get_session(session_id) is None

    def test_concurrent_session_access(self):
        """Test thread safety of session management."""
        adapter = FelixGradioAdapter(max_sessions=50)
        session_ids = []
        errors = []

        def create_and_access():
            try:
                sid = adapter.create_session()
                session_ids.append(sid)

                # Access session multiple times
                for _ in range(10):
                    session = adapter.get_session(sid)
                    assert session is not None

                # Cleanup
                adapter.cleanup_session(sid)

            except Exception as e:
                errors.append(str(e))

        # Run concurrent threads
        threads = []
        for _ in range(20):
            t = threading.Thread(target=create_and_access)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0
        assert len(session_ids) == 20


class TestProgressTracking:
    """Test progress tracking functionality."""

    def test_basic_progress_tracking(self):
        """Test basic progress tracking operations."""
        tracker = ProgressTracker()

        with tracker.track_operation("test_op") as op:
            assert op.current_progress == 0.0

            op.update(50, "Half way")
            state = tracker.get_progress("test_op")
            assert state.current_progress == 50.0
            assert state.message == "Half way"

            op.update(100, "Complete")
            assert op.current_progress == 100.0

        # Operation should be in history
        assert "test_op" not in tracker._active_operations
        assert len(tracker._history) == 1

    def test_nested_progress_operations(self):
        """Test nested progress tracking."""
        tracker = ProgressTracker()

        with tracker.track_operation("outer") as outer_op:
            outer_op.update(25, "Outer started")

            with tracker.track_operation("inner") as inner_op:
                inner_op.update(50, "Inner processing")

                # Both should be active
                active = tracker.get_all_active()
                assert len(active) == 2
                assert "outer" in active
                assert "inner" in active

            # Inner should be completed
            assert "inner" not in tracker._active_operations

            outer_op.update(100, "Outer complete")

        # Both should be in history
        assert len(tracker._history) == 2

    def test_progress_callbacks(self):
        """Test progress callback notifications."""
        tracker = ProgressTracker()
        callback_data = []

        def callback(op_id, progress, message):
            callback_data.append((op_id, progress, message))

        tracker.register_callback(callback)

        with tracker.track_operation("test") as op:
            op.update(50, "Processing")
            op.update(100, "Done")

        # Check callbacks were called
        assert len(callback_data) >= 2
        assert any(d[1] == 50.0 for d in callback_data)
        assert any(d[1] == 100.0 for d in callback_data)

    def test_gradio_progress_adapter(self):
        """Test Gradio progress adapter."""
        gradio_progress = Mock()
        adapter = GradioProgressAdapter(gradio_progress)

        with adapter.track("operation") as op:
            op.update(25, "Starting")
            op.update(75, "Almost done")

        # Check Gradio progress was called
        assert gradio_progress.call_count >= 2
        calls = gradio_progress.call_args_list
        assert any(0.25 in str(call) for call in calls)
        assert any(0.75 in str(call) for call in calls)


class TestResultCache:
    """Test result caching functionality."""

    def test_basic_caching(self):
        """Test basic cache operations."""
        cache = ResultCache(max_size=10)

        result = {"content": "Test content", "success": True}
        cache.set("test topic", "medium", result)

        # Retrieve from cache
        cached = cache.get("test topic", "medium")
        assert cached is not None
        assert cached["content"] == "Test content"

        # Different parameters should not match
        cached2 = cache.get("test topic", "complex")
        assert cached2 is None

    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        cache = ResultCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.set(f"topic{i}", "medium", {"id": i})

        # Access first item to make it recently used
        cache.get("topic0", "medium")

        # Add new item, should evict topic1 (least recently used)
        cache.set("topic3", "medium", {"id": 3})

        # Check eviction
        assert cache.get("topic0", "medium") is not None
        assert cache.get("topic1", "medium") is None  # Evicted
        assert cache.get("topic2", "medium") is not None
        assert cache.get("topic3", "medium") is not None

    def test_cache_thread_safety(self):
        """Test thread-safe cache access."""
        cache = ResultCache(max_size=100)
        errors = []

        def cache_operations():
            try:
                for i in range(50):
                    # Set and get operations
                    cache.set(f"topic{i}", "medium", {"id": i})
                    result = cache.get(f"topic{i}", "medium")
                    assert result["id"] == i
            except Exception as e:
                errors.append(str(e))

        # Run concurrent threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=cache_operations)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


class TestGPUResourceManager:
    """Test GPU resource management."""

    def test_gpu_detection(self):
        """Test GPU availability detection."""
        manager = GPUResourceManager(enable_gpu=True)

        # Should detect GPU availability (or lack thereof)
        assert isinstance(manager.enable_gpu, bool)

        stats = manager.get_statistics()
        assert "gpu_enabled" in stats
        assert "is_zerogpu" in stats

    def test_resource_allocation(self):
        """Test GPU resource allocation."""
        manager = GPUResourceManager(enable_gpu=False)  # Force CPU for testing

        with manager.acquire_resources(priority="normal", memory_required=2048) as ctx:
            assert ctx is not None
            assert not ctx.is_gpu  # Should be CPU since we disabled GPU

            # Device string should be CPU
            device_str = manager.get_device_string(ctx)
            assert device_str == "cpu"

    def test_concurrent_resource_limits(self):
        """Test concurrent resource allocation limits."""
        manager = GPUResourceManager(enable_gpu=False, max_concurrent=2)

        contexts = []

        # Allocate max resources
        for _ in range(2):
            ctx = manager._allocate_resources("normal", 1024)
            contexts.append(ctx)

        # Next allocation should fall back to CPU
        ctx3 = manager._allocate_resources("normal", 1024)
        assert not ctx3.is_gpu

        # Release one
        manager._release_resources(contexts[0])

        # Now should be able to allocate again
        ctx4 = manager._allocate_resources("normal", 1024)
        # Still CPU since we disabled GPU
        assert not ctx4.is_gpu

        # Cleanup
        for ctx in contexts[1:]:
            manager._release_resources(ctx)

    def test_priority_levels(self):
        """Test different priority levels."""
        manager = GPUResourceManager(enable_gpu=False)

        for priority in ["high", "normal", "low"]:
            with manager.acquire_resources(priority=priority) as ctx:
                assert ctx.priority == priority

                # Check limits are applied
                limits = manager.PRIORITY_LEVELS[priority]
                assert ctx.memory_allocated <= limits["memory_limit"]


class TestHelixCache:
    """Test helix caching system."""

    def test_cache_warming(self):
        """Test cache pre-warming with common configurations."""
        cache = HelixCache()

        # Common configurations should be pre-cached
        stats = cache.get_statistics()
        assert stats["cache_size"] > 0

    def test_helix_data_caching(self):
        """Test helix data computation and caching."""
        cache = HelixCache()

        # Get helix data
        entry = cache.get_helix_data(10.0, 0.01, 20.0, 20)
        assert entry is not None
        assert entry.positions.shape[0] == cache.precompute_steps

        # Second request should use cache
        entry2 = cache.get_helix_data(10.0, 0.01, 20.0, 20)
        assert entry2 is entry  # Same object

    def test_cached_helix_geometry(self):
        """Test cached helix geometry wrapper."""
        cache = HelixCache()

        cached_helix = cache.get_cached_helix(10.0, 0.01, 20.0, 20)
        assert cached_helix is not None

        # Test position calculation
        pos = cached_helix.get_position(0.5)
        assert len(pos) == 3
        assert all(isinstance(p, float) for p in pos)

        # Test radius calculation
        radius = cached_helix.get_radius(10.0)
        assert isinstance(radius, float)
        assert radius > 0

    def test_position_interpolation(self):
        """Test position interpolation accuracy."""
        cache = HelixCache()

        entry = cache.get_helix_data(10.0, 0.01, 20.0, 20)

        # Test interpolation at various points
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pos = cache.interpolate_position(entry, t)
            assert len(pos) == 3
            assert all(isinstance(p, float) for p in pos)


class TestGradioIntegration:
    """Integration tests for Gradio components."""

    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Test async processing compatibility."""
        adapter = FelixGradioAdapter()
        session_id = adapter.create_session()

        # Mock process_blog_request as async generator
        async def mock_process():
            yield ("Starting", 10.0)
            await asyncio.sleep(0.01)
            yield ("Processing", 50.0)
            await asyncio.sleep(0.01)
            yield ("Completing", 90.0)
            return {"success": True, "content": "Test"}

        # Should handle async operations
        result = None
        async for status, progress in mock_process():
            if isinstance(status, dict):
                result = status

        assert result is not None

    def test_complexity_configurations(self):
        """Test different complexity level configurations."""
        adapter = FelixGradioAdapter()

        for complexity in ComplexityLevel:
            config = adapter.COMPLEXITY_CONFIG[complexity]

            assert "num_agents" in config
            assert "helix_turns" in config
            assert "max_tokens" in config
            assert "simulation_time" in config
            assert "timeout" in config

            # Check reasonable values
            assert 1 <= config["num_agents"] <= 30
            assert 1 <= config["helix_turns"] <= 50
            assert 50 <= config["max_tokens"] <= 2000
            assert 0.1 <= config["simulation_time"] <= 1.0
            assert 5 <= config["timeout"] <= 120


def test_thread_safety_stress():
    """Stress test for thread safety with high concurrency."""
    adapter = FelixGradioAdapter(max_sessions=100)
    errors = []
    results = []

    def worker(worker_id):
        try:
            # Create session
            session_id = adapter.create_session()

            # Simulate processing
            for status, progress in adapter.process_blog_request(
                topic=f"Test topic {worker_id}",
                complexity="demo",
                session_id=session_id,
                use_cache=False
            ):
                if isinstance(status, dict):
                    results.append(status)
                    break

            # Cleanup
            adapter.cleanup_session(session_id)

        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")

    # Run many concurrent workers
    threads = []
    for i in range(50):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all to complete
    for t in threads:
        t.join(timeout=30)

    # Check results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) > 0, "No results generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])