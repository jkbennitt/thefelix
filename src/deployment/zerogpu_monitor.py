"""
ZeroGPU Deployment Monitoring and Resource Management for Felix Framework.

This module provides specialized monitoring, optimization, and error handling
for deploying Felix Framework on Hugging Face ZeroGPU infrastructure.

Key Features:
- GPU memory usage tracking and alerts
- Model loading/unloading optimization
- Batch processing management
- Resource cleanup automation
- Graceful degradation to CPU/Inference API
- Performance profiling and analytics
- Error recovery strategies
"""

import os
import gc
import time
import asyncio
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import deque, defaultdict
import psutil
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources to monitor."""
    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    CPU_USAGE = "cpu_usage"
    NETWORK = "network"
    DISK = "disk"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResourceAlert:
    """Resource usage alert."""
    resource_type: ResourceType
    severity: AlertSeverity
    threshold: float
    current_value: float
    message: str
    timestamp: float
    agent_context: Optional[str] = None


@dataclass
class GPUMemorySnapshot:
    """GPU memory usage snapshot."""
    timestamp: float
    allocated_mb: float
    cached_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    active_models: List[str] = field(default_factory=list)
    agent_count: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for ZeroGPU operations."""
    model_load_time: float = 0.0
    inference_time: float = 0.0
    batch_processing_time: float = 0.0
    memory_cleanup_time: float = 0.0
    total_operation_time: float = 0.0
    tokens_processed: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    memory_efficiency: float = 0.0


class ZeroGPUMonitor:
    """
    ZeroGPU resource monitor and optimizer for Felix Framework.

    Provides real-time monitoring, optimization strategies, and error handling
    specifically designed for ZeroGPU deployment constraints.
    """

    # ZeroGPU-specific thresholds
    GPU_MEMORY_WARNING_THRESHOLD = 0.7  # 70% usage
    GPU_MEMORY_CRITICAL_THRESHOLD = 0.85  # 85% usage
    CPU_MEMORY_WARNING_THRESHOLD = 0.8   # 80% usage
    CPU_MEMORY_CRITICAL_THRESHOLD = 0.9  # 90% usage

    # Performance targets for ZeroGPU
    TARGET_INFERENCE_TIME = 5.0  # 5 seconds max
    TARGET_MODEL_LOAD_TIME = 10.0  # 10 seconds max
    MAX_CONCURRENT_AGENTS = 8  # Conservative limit for ZeroGPU

    def __init__(self,
                 enable_gpu_monitoring: bool = True,
                 alert_callback: Optional[Callable[[ResourceAlert], None]] = None,
                 cleanup_interval: float = 30.0,
                 performance_log_interval: float = 60.0):
        """
        Initialize ZeroGPU monitor.

        Args:
            enable_gpu_monitoring: Enable GPU monitoring (requires torch)
            alert_callback: Callback function for resource alerts
            cleanup_interval: Automatic cleanup interval in seconds
            performance_log_interval: Performance logging interval in seconds
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.alert_callback = alert_callback
        self.cleanup_interval = cleanup_interval
        self.performance_log_interval = performance_log_interval

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alerts: deque[ResourceAlert] = deque(maxlen=100)
        self.performance_history: deque[PerformanceMetrics] = deque(maxlen=1000)

        # GPU monitoring
        self.gpu_available = False
        self.torch_available = False
        self._init_gpu_monitoring()

        # Resource tracking
        self.memory_snapshots: deque[GPUMemorySnapshot] = deque(maxlen=200)
        self.active_models: Dict[str, float] = {}  # model_id -> load_time
        self.active_agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata

        # Performance tracking
        self.operation_start_times: Dict[str, float] = {}
        self.batch_operations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Cleanup tracking
        self.last_cleanup: float = time.time()
        self.cleanup_stats = {
            "models_unloaded": 0,
            "memory_freed_mb": 0.0,
            "last_cleanup_time": 0.0
        }

        logger.info(f"ZeroGPU Monitor initialized - GPU: {self.gpu_available}, Torch: {self.torch_available}")

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities."""
        try:
            import torch
            self.torch_available = True
            self.gpu_available = torch.cuda.is_available()

            if self.gpu_available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning("CUDA not available - GPU monitoring disabled")

        except ImportError:
            logger.warning("PyTorch not available - GPU monitoring disabled")
            self.torch_available = False
            self.gpu_available = False

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("ZeroGPU monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("ZeroGPU monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_performance_log = time.time()

        while self.is_monitoring:
            try:
                current_time = time.time()

                # Take resource snapshots
                self._take_memory_snapshot()
                self._check_resource_thresholds()

                # Automatic cleanup
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self._automatic_cleanup()
                    self.last_cleanup = current_time

                # Performance logging
                if current_time - last_performance_log > self.performance_log_interval:
                    self._log_performance_summary()
                    last_performance_log = current_time

                time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Longer sleep on error

    def _take_memory_snapshot(self):
        """Take a memory usage snapshot."""
        timestamp = time.time()

        # CPU memory
        cpu_memory = psutil.virtual_memory()

        # GPU memory (if available)
        gpu_allocated = gpu_cached = gpu_reserved = gpu_free = gpu_total = 0.0
        gpu_utilization = 0.0

        if self.gpu_available and self.torch_available:
            try:
                import torch
                gpu_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_cached = torch.cuda.memory_cached() / 1024**2  # MB
                gpu_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
                gpu_free = gpu_total - gpu_reserved
                gpu_utilization = (gpu_reserved / gpu_total) * 100 if gpu_total > 0 else 0.0
            except Exception as e:
                logger.warning(f"GPU memory snapshot failed: {e}")

        snapshot = GPUMemorySnapshot(
            timestamp=timestamp,
            allocated_mb=gpu_allocated,
            cached_mb=gpu_cached,
            reserved_mb=gpu_reserved,
            free_mb=gpu_free,
            total_mb=gpu_total,
            utilization_percent=gpu_utilization,
            active_models=list(self.active_models.keys()),
            agent_count=len(self.active_agents)
        )

        self.memory_snapshots.append(snapshot)

    def _check_resource_thresholds(self):
        """Check resource usage against thresholds and create alerts."""
        if not self.memory_snapshots:
            return

        latest = self.memory_snapshots[-1]

        # GPU memory alerts
        if self.gpu_available and latest.total_mb > 0:
            gpu_usage_ratio = latest.utilization_percent / 100.0

            if gpu_usage_ratio >= self.GPU_MEMORY_CRITICAL_THRESHOLD:
                self._create_alert(
                    ResourceType.GPU_MEMORY,
                    AlertSeverity.CRITICAL,
                    self.GPU_MEMORY_CRITICAL_THRESHOLD,
                    gpu_usage_ratio,
                    f"Critical GPU memory usage: {gpu_usage_ratio:.1%} ({latest.reserved_mb:.0f}MB/{latest.total_mb:.0f}MB)"
                )
            elif gpu_usage_ratio >= self.GPU_MEMORY_WARNING_THRESHOLD:
                self._create_alert(
                    ResourceType.GPU_MEMORY,
                    AlertSeverity.WARNING,
                    self.GPU_MEMORY_WARNING_THRESHOLD,
                    gpu_usage_ratio,
                    f"High GPU memory usage: {gpu_usage_ratio:.1%} ({latest.reserved_mb:.0f}MB/{latest.total_mb:.0f}MB)"
                )

        # CPU memory alerts
        cpu_memory = psutil.virtual_memory()
        cpu_usage_ratio = cpu_memory.percent / 100.0

        if cpu_usage_ratio >= self.CPU_MEMORY_CRITICAL_THRESHOLD:
            self._create_alert(
                ResourceType.CPU_MEMORY,
                AlertSeverity.CRITICAL,
                self.CPU_MEMORY_CRITICAL_THRESHOLD,
                cpu_usage_ratio,
                f"Critical CPU memory usage: {cpu_usage_ratio:.1%} ({cpu_memory.used / 1024**3:.1f}GB/{cpu_memory.total / 1024**3:.1f}GB)"
            )
        elif cpu_usage_ratio >= self.CPU_MEMORY_WARNING_THRESHOLD:
            self._create_alert(
                ResourceType.CPU_MEMORY,
                AlertSeverity.WARNING,
                self.CPU_MEMORY_WARNING_THRESHOLD,
                cpu_usage_ratio,
                f"High CPU memory usage: {cpu_usage_ratio:.1%} ({cpu_memory.used / 1024**3:.1f}GB/{cpu_memory.total / 1024**3:.1f}GB)"
            )

    def _create_alert(self, resource_type: ResourceType, severity: AlertSeverity,
                     threshold: float, current_value: float, message: str,
                     agent_context: Optional[str] = None):
        """Create and process a resource alert."""
        alert = ResourceAlert(
            resource_type=resource_type,
            severity=severity,
            threshold=threshold,
            current_value=current_value,
            message=message,
            timestamp=time.time(),
            agent_context=agent_context
        )

        self.alerts.append(alert)

        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }[severity]

        logger.log(log_level, f"Resource Alert [{severity.value.upper()}]: {message}")

        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Trigger automatic actions for critical alerts
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            self._handle_critical_alert(alert)

    def _handle_critical_alert(self, alert: ResourceAlert):
        """Handle critical resource alerts with automatic actions."""
        if alert.resource_type == ResourceType.GPU_MEMORY:
            logger.warning("Critical GPU memory - triggering emergency cleanup")
            self._emergency_memory_cleanup()
        elif alert.resource_type == ResourceType.CPU_MEMORY:
            logger.warning("Critical CPU memory - reducing agent load")
            self._reduce_agent_load()

    def _emergency_memory_cleanup(self):
        """Emergency GPU memory cleanup."""
        cleaned_mb = 0.0

        if self.gpu_available and self.torch_available:
            try:
                import torch

                # Clear cache
                torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                # Unload oldest models
                if self.active_models:
                    models_to_remove = sorted(self.active_models.items(), key=lambda x: x[1])[:2]
                    for model_id, _ in models_to_remove:
                        self._unload_model(model_id)
                        cleaned_mb += 100  # Estimate

                logger.info(f"Emergency cleanup completed - freed ~{cleaned_mb:.0f}MB")

            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")

    def _reduce_agent_load(self):
        """Reduce agent load by terminating non-critical agents."""
        if len(self.active_agents) > self.MAX_CONCURRENT_AGENTS // 2:
            # Terminate oldest agents
            agents_to_remove = list(self.active_agents.keys())[:-self.MAX_CONCURRENT_AGENTS // 2]
            for agent_id in agents_to_remove:
                self.unregister_agent(agent_id)
                logger.info(f"Terminated agent {agent_id} due to resource pressure")

    def _automatic_cleanup(self):
        """Perform automatic cleanup of unused resources."""
        current_time = time.time()
        cleanup_start = time.time()

        cleaned_models = 0
        freed_memory_mb = 0.0

        # Remove old models (not used in last 5 minutes)
        models_to_remove = []
        for model_id, load_time in self.active_models.items():
            if current_time - load_time > 300:  # 5 minutes
                models_to_remove.append(model_id)

        for model_id in models_to_remove:
            if self._unload_model(model_id):
                cleaned_models += 1
                freed_memory_mb += 100  # Estimate

        # Remove inactive agents (not updated in last 2 minutes)
        agents_to_remove = []
        for agent_id, metadata in self.active_agents.items():
            if current_time - metadata.get("last_activity", 0) > 120:  # 2 minutes
                agents_to_remove.append(agent_id)

        for agent_id in agents_to_remove:
            self.unregister_agent(agent_id)

        # Clear PyTorch cache if available
        if self.gpu_available and self.torch_available:
            try:
                import torch
                torch.cuda.empty_cache()
                freed_memory_mb += 50  # Estimate
            except Exception as e:
                logger.warning(f"PyTorch cache clear failed: {e}")

        # Force garbage collection
        gc.collect()

        cleanup_time = time.time() - cleanup_start

        # Update cleanup stats
        self.cleanup_stats.update({
            "models_unloaded": self.cleanup_stats["models_unloaded"] + cleaned_models,
            "memory_freed_mb": self.cleanup_stats["memory_freed_mb"] + freed_memory_mb,
            "last_cleanup_time": cleanup_time
        })

        if cleaned_models > 0 or len(agents_to_remove) > 0:
            logger.info(f"Automatic cleanup: {cleaned_models} models, {len(agents_to_remove)} agents, "
                       f"~{freed_memory_mb:.0f}MB freed in {cleanup_time:.2f}s")

    def _unload_model(self, model_id: str) -> bool:
        """Unload a specific model from memory."""
        if model_id in self.active_models:
            try:
                # Model-specific cleanup would go here
                # For now, just remove from tracking
                del self.active_models[model_id]
                logger.debug(f"Unloaded model: {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {e}")
        return False

    def _log_performance_summary(self):
        """Log performance summary."""
        if not self.performance_history:
            return

        recent_metrics = list(self.performance_history)[-10:]  # Last 10 operations

        if recent_metrics:
            avg_inference_time = sum(m.inference_time for m in recent_metrics) / len(recent_metrics)
            avg_model_load_time = sum(m.model_load_time for m in recent_metrics) / len(recent_metrics)
            total_tokens = sum(m.tokens_processed for m in recent_metrics)
            success_rate = sum(m.successful_operations for m in recent_metrics) / max(1, len(recent_metrics))

            latest_snapshot = self.memory_snapshots[-1] if self.memory_snapshots else None
            gpu_usage = latest_snapshot.utilization_percent if latest_snapshot else 0.0

            logger.info(
                f"Performance Summary - "
                f"Inference: {avg_inference_time:.2f}s avg, "
                f"Model Load: {avg_model_load_time:.2f}s avg, "
                f"Success Rate: {success_rate:.1%}, "
                f"Tokens: {total_tokens}, "
                f"GPU: {gpu_usage:.1f}%, "
                f"Active: {len(self.active_agents)} agents, {len(self.active_models)} models"
            )

    @contextmanager
    def track_operation(self, operation_name: str, agent_id: Optional[str] = None):
        """Context manager to track operation performance."""
        start_time = time.time()
        operation_id = f"{operation_name}_{agent_id}_{start_time}"

        self.operation_start_times[operation_id] = start_time

        try:
            yield operation_id
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Record performance metric
            metrics = PerformanceMetrics(
                total_operation_time=duration,
                successful_operations=1,
                failed_operations=0
            )

            # Operation-specific metrics
            if "inference" in operation_name.lower():
                metrics.inference_time = duration
            elif "load" in operation_name.lower():
                metrics.model_load_time = duration
            elif "batch" in operation_name.lower():
                metrics.batch_processing_time = duration
            elif "cleanup" in operation_name.lower():
                metrics.memory_cleanup_time = duration

            self.performance_history.append(metrics)

            # Remove from tracking
            self.operation_start_times.pop(operation_id, None)

            # Log slow operations
            if duration > self.TARGET_INFERENCE_TIME:
                logger.warning(f"Slow operation: {operation_name} took {duration:.2f}s (target: {self.TARGET_INFERENCE_TIME}s)")

    def register_model(self, model_id: str) -> bool:
        """Register a loaded model."""
        if len(self.active_models) >= 3:  # Conservative limit for ZeroGPU
            logger.warning(f"Model limit reached - cannot load {model_id}")
            return False

        self.active_models[model_id] = time.time()
        logger.debug(f"Registered model: {model_id}")
        return True

    def register_agent(self, agent_id: str, agent_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Register an active agent."""
        if len(self.active_agents) >= self.MAX_CONCURRENT_AGENTS:
            logger.warning(f"Agent limit reached - cannot register {agent_id}")
            return False

        self.active_agents[agent_id] = {
            "agent_type": agent_type,
            "registered_at": time.time(),
            "last_activity": time.time(),
            **(metadata or {})
        }
        logger.debug(f"Registered agent: {agent_id} ({agent_type})")
        return True

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            logger.debug(f"Unregistered agent: {agent_id}")

    def update_agent_activity(self, agent_id: str):
        """Update agent's last activity timestamp."""
        if agent_id in self.active_agents:
            self.active_agents[agent_id]["last_activity"] = time.time()

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        latest_snapshot = self.memory_snapshots[-1] if self.memory_snapshots else None
        cpu_memory = psutil.virtual_memory()

        return {
            "timestamp": time.time(),
            "gpu": {
                "available": self.gpu_available,
                "memory_mb": {
                    "allocated": latest_snapshot.allocated_mb if latest_snapshot else 0,
                    "cached": latest_snapshot.cached_mb if latest_snapshot else 0,
                    "reserved": latest_snapshot.reserved_mb if latest_snapshot else 0,
                    "free": latest_snapshot.free_mb if latest_snapshot else 0,
                    "total": latest_snapshot.total_mb if latest_snapshot else 0,
                },
                "utilization_percent": latest_snapshot.utilization_percent if latest_snapshot else 0,
            },
            "cpu": {
                "memory_percent": cpu_memory.percent,
                "memory_gb": {
                    "used": cpu_memory.used / 1024**3,
                    "available": cpu_memory.available / 1024**3,
                    "total": cpu_memory.total / 1024**3,
                }
            },
            "active": {
                "models": len(self.active_models),
                "agents": len(self.active_agents),
                "model_list": list(self.active_models.keys()),
                "agent_types": [meta["agent_type"] for meta in self.active_agents.values()]
            },
            "alerts": {
                "total": len(self.alerts),
                "critical": len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
                "warnings": len([a for a in self.alerts if a.severity == AlertSeverity.WARNING])
            },
            "cleanup_stats": self.cleanup_stats.copy()
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        recent_metrics = list(self.performance_history)[-50:]  # Last 50 operations

        total_ops = len(recent_metrics)
        successful_ops = sum(m.successful_operations for m in recent_metrics)
        failed_ops = sum(m.failed_operations for m in recent_metrics)

        inference_times = [m.inference_time for m in recent_metrics if m.inference_time > 0]
        model_load_times = [m.model_load_time for m in recent_metrics if m.model_load_time > 0]

        return {
            "summary": {
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "success_rate": successful_ops / max(1, total_ops),
                "total_tokens_processed": sum(m.tokens_processed for m in recent_metrics)
            },
            "timing": {
                "avg_inference_time": sum(inference_times) / max(1, len(inference_times)),
                "max_inference_time": max(inference_times) if inference_times else 0,
                "avg_model_load_time": sum(model_load_times) / max(1, len(model_load_times)),
                "max_model_load_time": max(model_load_times) if model_load_times else 0,
                "target_inference_time": self.TARGET_INFERENCE_TIME,
                "target_model_load_time": self.TARGET_MODEL_LOAD_TIME
            },
            "efficiency": {
                "operations_per_minute": total_ops / max(1, (time.time() - recent_metrics[0].total_operation_time) / 60),
                "avg_memory_efficiency": sum(m.memory_efficiency for m in recent_metrics) / max(1, total_ops),
                "cleanup_frequency": self.cleanup_interval
            }
        }

    def export_diagnostics(self, filepath: Optional[str] = None) -> str:
        """Export comprehensive diagnostics to JSON."""
        diagnostics = {
            "timestamp": time.time(),
            "monitor_config": {
                "gpu_monitoring_enabled": self.enable_gpu_monitoring,
                "gpu_available": self.gpu_available,
                "torch_available": self.torch_available,
                "cleanup_interval": self.cleanup_interval,
                "performance_log_interval": self.performance_log_interval
            },
            "resource_status": self.get_resource_status(),
            "performance_report": self.get_performance_report(),
            "recent_alerts": [
                {
                    "resource_type": alert.resource_type.value,
                    "severity": alert.severity.value,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "agent_context": alert.agent_context
                }
                for alert in list(self.alerts)[-20:]  # Last 20 alerts
            ],
            "memory_snapshots": [
                {
                    "timestamp": snap.timestamp,
                    "gpu_utilization_percent": snap.utilization_percent,
                    "gpu_allocated_mb": snap.allocated_mb,
                    "gpu_free_mb": snap.free_mb,
                    "active_models": snap.active_models,
                    "agent_count": snap.agent_count
                }
                for snap in list(self.memory_snapshots)[-20:]  # Last 20 snapshots
            ]
        }

        diagnostics_json = json.dumps(diagnostics, indent=2)

        if filepath:
            Path(filepath).write_text(diagnostics_json)
            logger.info(f"Diagnostics exported to {filepath}")

        return diagnostics_json


# Utility functions for ZeroGPU optimization

def create_zerogpu_monitor(alert_callback: Optional[Callable] = None) -> ZeroGPUMonitor:
    """Create a ZeroGPU monitor with optimal settings."""

    def default_alert_callback(alert: ResourceAlert):
        """Default alert handler that logs critical issues."""
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            logger.error(f"ZeroGPU Alert: {alert.message}")
        else:
            logger.warning(f"ZeroGPU Warning: {alert.message}")

    monitor = ZeroGPUMonitor(
        enable_gpu_monitoring=True,
        alert_callback=alert_callback or default_alert_callback,
        cleanup_interval=30.0,  # Aggressive cleanup for ZeroGPU
        performance_log_interval=60.0
    )

    monitor.start_monitoring()
    return monitor


def optimize_for_zerogpu():
    """Apply system-wide optimizations for ZeroGPU deployment."""
    # Set environment variables for optimal ZeroGPU performance
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    # Force aggressive garbage collection
    gc.set_threshold(100, 10, 10)  # More aggressive GC

    logger.info("Applied ZeroGPU optimizations")


# Export main classes and functions
__all__ = [
    'ZeroGPUMonitor',
    'ResourceAlert',
    'GPUMemorySnapshot',
    'PerformanceMetrics',
    'ResourceType',
    'AlertSeverity',
    'create_zerogpu_monitor',
    'optimize_for_zerogpu'
]