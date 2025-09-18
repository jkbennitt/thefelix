"""
Scalable Architecture Configuration for Felix Framework HF Pro Deployment

This module provides comprehensive scalability configurations and optimization
strategies for high-load deployments on HuggingFace Pro accounts with ZeroGPU.

Features:
- Auto-scaling configuration for increased user loads
- Load balancing strategies for multi-instance deployments
- Queue management for request buffering
- Circuit breaker patterns for resilience
- Resource pooling and connection management
- Horizontal scaling with HF Spaces replication
- Performance optimization for concurrent users
- Adaptive resource allocation based on demand
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling operation modes."""
    MANUAL = "manual"
    AUTO = "auto"
    SCHEDULED = "scheduled"
    REACTIVE = "reactive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"


class HealthStatus(Enum):
    """Instance health statuses."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    concurrent_users: int
    queue_length: int
    avg_response_time: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    request_rate: float
    error_rate: float
    cost_per_request: float


@dataclass
class InstanceConfig:
    """Configuration for a Felix Framework instance."""
    instance_id: str
    endpoint_url: str
    weight: float = 1.0
    max_concurrent_requests: int = 10
    health_check_url: str = "/health"
    timeout: float = 30.0
    last_health_check: Optional[datetime] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_count: int = 1
    scale_down_count: int = 1
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for resilient service calls."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1

            result = await func(*args, **kwargs)

            # Success
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if (self.state == CircuitBreakerState.CLOSED and
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN

            elif (self.state == CircuitBreakerState.HALF_OPEN and
                  self.half_open_calls >= self.config.half_open_max_calls):
                self.state = CircuitBreakerState.OPEN

            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False

        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout


class LoadBalancer:
    """Load balancer for distributing requests across Felix instances."""

    def __init__(self,
                 instances: List[InstanceConfig],
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
                 health_check_interval: int = 30):
        """
        Initialize load balancer.

        Args:
            instances: List of instance configurations
            strategy: Load balancing strategy
            health_check_interval: Health check interval in seconds
        """
        self.instances = {inst.instance_id: inst for inst in instances}
        self.strategy = strategy
        self.health_check_interval = health_check_interval

        # Request tracking
        self.current_index = 0
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)

        # Circuit breakers for each instance
        self.circuit_breakers = {
            inst.instance_id: CircuitBreaker(CircuitBreakerConfig())
            for inst in instances
        }

        # Health checking
        self.health_check_task: Optional[asyncio.Task] = None

        logger.info(f"Load balancer initialized with {len(instances)} instances")

    async def start(self):
        """Start load balancer services."""
        if not self.health_check_task:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Load balancer started")

    async def stop(self):
        """Stop load balancer services."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("Load balancer stopped")

    async def select_instance(self) -> Optional[InstanceConfig]:
        """Select best instance based on strategy."""
        healthy_instances = [
            inst for inst in self.instances.values()
            if inst.health_status == HealthStatus.HEALTHY
        ]

        if not healthy_instances:
            logger.warning("No healthy instances available")
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(healthy_instances)
        else:
            return healthy_instances[0]

    def _round_robin_selection(self, instances: List[InstanceConfig]) -> InstanceConfig:
        """Round-robin instance selection."""
        self.current_index = (self.current_index + 1) % len(instances)
        return instances[self.current_index]

    def _least_connections_selection(self, instances: List[InstanceConfig]) -> InstanceConfig:
        """Select instance with least connections."""
        return min(instances, key=lambda x: x.current_connections)

    def _weighted_round_robin_selection(self, instances: List[InstanceConfig]) -> InstanceConfig:
        """Weighted round-robin selection."""
        total_weight = sum(inst.weight for inst in instances)
        weighted_instances = []

        for inst in instances:
            count = int(inst.weight / total_weight * 100)
            weighted_instances.extend([inst] * max(1, count))

        self.current_index = (self.current_index + 1) % len(weighted_instances)
        return weighted_instances[self.current_index]

    def _least_response_time_selection(self, instances: List[InstanceConfig]) -> InstanceConfig:
        """Select instance with lowest response time."""
        return min(instances, key=lambda x: x.avg_response_time)

    def _resource_based_selection(self, instances: List[InstanceConfig]) -> InstanceConfig:
        """Select instance based on resource utilization."""
        def resource_score(inst: InstanceConfig) -> float:
            # Lower score = better choice
            connections_score = inst.current_connections / inst.max_concurrent_requests
            response_time_score = min(inst.avg_response_time / 5.0, 1.0)  # Normalize to 5s max
            return (connections_score * 0.6) + (response_time_score * 0.4)

        return min(instances, key=resource_score)

    async def execute_request(self, instance: InstanceConfig, request_func: Callable, *args, **kwargs):
        """Execute request through circuit breaker."""
        circuit_breaker = self.circuit_breakers[instance.instance_id]

        instance.current_connections += 1
        start_time = time.time()

        try:
            result = await circuit_breaker.call(request_func, *args, **kwargs)

            # Update metrics
            response_time = time.time() - start_time
            self.response_times[instance.instance_id].append(response_time)
            if len(self.response_times[instance.instance_id]) > 100:
                self.response_times[instance.instance_id] = self.response_times[instance.instance_id][-100:]

            instance.avg_response_time = statistics.mean(self.response_times[instance.instance_id])
            instance.total_requests += 1

            return result

        finally:
            instance.current_connections -= 1

    async def _health_check_loop(self):
        """Periodic health check for all instances."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_instances(self):
        """Check health of all instances."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._check_instance_health(session, instance)
                for instance in self.instances.values()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_instance_health(self, session: aiohttp.ClientSession, instance: InstanceConfig):
        """Check health of a single instance."""
        try:
            health_url = f"{instance.endpoint_url.rstrip('/')}{instance.health_check_url}"
            async with session.get(health_url, timeout=10) as response:
                if response.status == 200:
                    instance.health_status = HealthStatus.HEALTHY
                else:
                    instance.health_status = HealthStatus.DEGRADED

        except Exception as e:
            logger.warning(f"Health check failed for {instance.instance_id}: {e}")
            instance.health_status = HealthStatus.UNHEALTHY

        instance.last_health_check = datetime.now()

    def get_instance_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(inst.total_requests for inst in self.instances.values())
        healthy_count = sum(1 for inst in self.instances.values()
                          if inst.health_status == HealthStatus.HEALTHY)

        return {
            "total_instances": len(self.instances),
            "healthy_instances": healthy_count,
            "total_requests": total_requests,
            "strategy": self.strategy.value,
            "instances": {
                inst.instance_id: {
                    "health_status": inst.health_status.value,
                    "current_connections": inst.current_connections,
                    "total_requests": inst.total_requests,
                    "avg_response_time": inst.avg_response_time,
                    "circuit_breaker_state": self.circuit_breakers[inst.instance_id].state.value
                }
                for inst in self.instances.values()
            }
        }


class AutoScaler:
    """Auto-scaling manager for Felix Framework deployments."""

    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 10,
                 scaling_rules: Optional[List[ScalingRule]] = None,
                 scaling_mode: ScalingMode = ScalingMode.AUTO,
                 metrics_window_minutes: int = 5):
        """
        Initialize auto-scaler.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            scaling_rules: List of scaling rules
            scaling_mode: Scaling operation mode
            metrics_window_minutes: Metrics evaluation window
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scaling_mode = scaling_mode
        self.metrics_window_minutes = metrics_window_minutes

        # Default scaling rules
        self.scaling_rules = scaling_rules or [
            ScalingRule(
                name="cpu_scale_up",
                metric_name="cpu_utilization",
                threshold_up=70.0,
                threshold_down=30.0,
                scale_up_count=1,
                cooldown_minutes=3
            ),
            ScalingRule(
                name="queue_scale_up",
                metric_name="queue_length",
                threshold_up=20.0,
                threshold_down=5.0,
                scale_up_count=2,
                cooldown_minutes=2
            ),
            ScalingRule(
                name="response_time_scale_up",
                metric_name="avg_response_time",
                threshold_up=5.0,
                threshold_down=2.0,
                scale_up_count=1,
                cooldown_minutes=3
            )
        ]

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.current_instances = 1
        self.scaling_events: deque = deque(maxlen=100)

        # Scaling callbacks
        self.scale_up_callback: Optional[Callable] = None
        self.scale_down_callback: Optional[Callable] = None

        logger.info("Auto-scaler initialized")

    def set_scaling_callbacks(self,
                            scale_up_callback: Callable[[int], None],
                            scale_down_callback: Callable[[int], None]):
        """Set callbacks for scaling operations."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback

    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics for scaling evaluation."""
        self.metrics_history.append(metrics)

        if self.scaling_mode == ScalingMode.AUTO:
            asyncio.create_task(self._evaluate_scaling())

    async def _evaluate_scaling(self):
        """Evaluate scaling needs based on current metrics."""
        if len(self.metrics_history) < 3:  # Need some history
            return

        # Get recent metrics (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=self.metrics_window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]

        if not recent_metrics:
            return

        # Calculate average values
        avg_metrics = {
            "cpu_utilization": statistics.mean(m.cpu_utilization for m in recent_metrics),
            "memory_utilization": statistics.mean(m.memory_utilization for m in recent_metrics),
            "gpu_utilization": statistics.mean(m.gpu_utilization for m in recent_metrics),
            "queue_length": statistics.mean(m.queue_length for m in recent_metrics),
            "avg_response_time": statistics.mean(m.avg_response_time for m in recent_metrics),
            "concurrent_users": statistics.mean(m.concurrent_users for m in recent_metrics),
            "error_rate": statistics.mean(m.error_rate for m in recent_metrics)
        }

        # Evaluate each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue

            # Check cooldown
            if (rule.last_triggered and
                (datetime.now() - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60):
                continue

            metric_value = avg_metrics.get(rule.metric_name, 0.0)

            # Scale up decision
            if (metric_value > rule.threshold_up and
                self.current_instances < self.max_instances):
                await self._scale_up(rule, metric_value)

            # Scale down decision
            elif (metric_value < rule.threshold_down and
                  self.current_instances > self.min_instances):
                await self._scale_down(rule, metric_value)

    async def _scale_up(self, rule: ScalingRule, metric_value: float):
        """Execute scale up operation."""
        new_count = min(
            self.current_instances + rule.scale_up_count,
            self.max_instances
        )

        if new_count > self.current_instances:
            logger.info(f"Scaling up: {self.current_instances} -> {new_count} "
                       f"(rule: {rule.name}, metric: {metric_value:.2f})")

            if self.scale_up_callback:
                await self.scale_up_callback(new_count - self.current_instances)

            self.current_instances = new_count
            rule.last_triggered = datetime.now()

            self.scaling_events.append({
                "timestamp": datetime.now(),
                "action": "scale_up",
                "rule": rule.name,
                "metric_value": metric_value,
                "threshold": rule.threshold_up,
                "old_count": self.current_instances - (new_count - self.current_instances),
                "new_count": new_count
            })

    async def _scale_down(self, rule: ScalingRule, metric_value: float):
        """Execute scale down operation."""
        new_count = max(
            self.current_instances - rule.scale_down_count,
            self.min_instances
        )

        if new_count < self.current_instances:
            logger.info(f"Scaling down: {self.current_instances} -> {new_count} "
                       f"(rule: {rule.name}, metric: {metric_value:.2f})")

            if self.scale_down_callback:
                await self.scale_down_callback(self.current_instances - new_count)

            self.current_instances = new_count
            rule.last_triggered = datetime.now()

            self.scaling_events.append({
                "timestamp": datetime.now(),
                "action": "scale_down",
                "rule": rule.name,
                "metric_value": metric_value,
                "threshold": rule.threshold_down,
                "old_count": self.current_instances + (self.current_instances - new_count),
                "new_count": new_count
            })

    def manual_scale(self, target_instances: int) -> bool:
        """Manually scale to target instance count."""
        target_instances = max(self.min_instances, min(target_instances, self.max_instances))

        if target_instances == self.current_instances:
            return True

        logger.info(f"Manual scaling: {self.current_instances} -> {target_instances}")

        self.current_instances = target_instances
        self.scaling_events.append({
            "timestamp": datetime.now(),
            "action": "manual_scale",
            "rule": "manual",
            "old_count": self.current_instances,
            "new_count": target_instances
        })

        return True

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scaling_mode": self.scaling_mode.value,
            "rules": [
                {
                    "name": rule.name,
                    "metric": rule.metric_name,
                    "threshold_up": rule.threshold_up,
                    "threshold_down": rule.threshold_down,
                    "enabled": rule.enabled,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule in self.scaling_rules
            ],
            "recent_events": list(self.scaling_events)[-10:]
        }


class ScalableArchitecture:
    """
    Comprehensive scalable architecture manager for Felix Framework.

    Coordinates load balancing, auto-scaling, and resource management
    for high-availability deployments on HuggingFace Pro.
    """

    def __init__(self,
                 initial_instances: List[InstanceConfig],
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
                 enable_auto_scaling: bool = True,
                 min_instances: int = 1,
                 max_instances: int = 10):
        """
        Initialize scalable architecture.

        Args:
            initial_instances: Initial instance configurations
            load_balancing_strategy: Load balancing strategy
            enable_auto_scaling: Enable auto-scaling
            min_instances: Minimum instances for auto-scaling
            max_instances: Maximum instances for auto-scaling
        """
        self.load_balancer = LoadBalancer(initial_instances, load_balancing_strategy)

        if enable_auto_scaling:
            self.auto_scaler = AutoScaler(min_instances, max_instances)
            self.auto_scaler.set_scaling_callbacks(
                scale_up_callback=self._handle_scale_up,
                scale_down_callback=self._handle_scale_down
            )
        else:
            self.auto_scaler = None

        # Request queue for buffering
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.queue_processors: List[asyncio.Task] = []

        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "current_queue_size": 0,
            "peak_queue_size": 0
        }

        logger.info("Scalable architecture initialized")

    async def start(self):
        """Start all architecture components."""
        await self.load_balancer.start()

        # Start queue processors
        processor_count = max(2, len(self.load_balancer.instances) // 2)
        for i in range(processor_count):
            processor = asyncio.create_task(self._queue_processor(f"processor_{i}"))
            self.queue_processors.append(processor)

        logger.info(f"Scalable architecture started with {processor_count} queue processors")

    async def stop(self):
        """Stop all architecture components."""
        await self.load_balancer.stop()

        # Stop queue processors
        for processor in self.queue_processors:
            processor.cancel()

        await asyncio.gather(*self.queue_processors, return_exceptions=True)
        self.queue_processors.clear()

        logger.info("Scalable architecture stopped")

    async def process_request(self, request_func: Callable, *args, **kwargs):
        """Process request through scalable architecture."""
        # Add to queue
        request_item = {
            "func": request_func,
            "args": args,
            "kwargs": kwargs,
            "result_future": asyncio.Future(),
            "timestamp": datetime.now()
        }

        try:
            self.request_queue.put_nowait(request_item)
            self.performance_metrics["current_queue_size"] = self.request_queue.qsize()
            self.performance_metrics["peak_queue_size"] = max(
                self.performance_metrics["peak_queue_size"],
                self.request_queue.qsize()
            )

            # Update auto-scaler metrics
            if self.auto_scaler:
                metrics = ScalingMetrics(
                    timestamp=datetime.now(),
                    concurrent_users=len(self.queue_processors),  # Simplified
                    queue_length=self.request_queue.qsize(),
                    avg_response_time=self.performance_metrics["avg_response_time"],
                    cpu_utilization=60.0,  # Mock data - would be real in production
                    memory_utilization=50.0,
                    gpu_utilization=40.0,
                    request_rate=10.0,
                    error_rate=self.performance_metrics["failed_requests"] /
                              max(1, self.performance_metrics["total_requests"]),
                    cost_per_request=0.05
                )
                self.auto_scaler.add_metrics(metrics)

            return await request_item["result_future"]

        except asyncio.QueueFull:
            raise Exception("Request queue is full - system overloaded")

    async def _queue_processor(self, processor_id: str):
        """Process requests from queue."""
        logger.info(f"Queue processor {processor_id} started")

        while True:
            try:
                # Get request from queue
                request_item = await self.request_queue.get()
                self.performance_metrics["current_queue_size"] = self.request_queue.qsize()

                start_time = time.time()

                try:
                    # Select instance
                    instance = await self.load_balancer.select_instance()
                    if not instance:
                        raise Exception("No healthy instances available")

                    # Execute request
                    result = await self.load_balancer.execute_request(
                        instance,
                        request_item["func"],
                        *request_item["args"],
                        **request_item["kwargs"]
                    )

                    # Update metrics
                    response_time = time.time() - start_time
                    self.performance_metrics["total_requests"] += 1
                    self.performance_metrics["successful_requests"] += 1
                    self._update_avg_response_time(response_time)

                    # Set result
                    request_item["result_future"].set_result(result)

                except Exception as e:
                    self.performance_metrics["total_requests"] += 1
                    self.performance_metrics["failed_requests"] += 1
                    request_item["result_future"].set_exception(e)

                finally:
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor {processor_id} error: {e}")

    def _update_avg_response_time(self, response_time: float):
        """Update average response time."""
        total_requests = self.performance_metrics["successful_requests"]
        if total_requests == 1:
            self.performance_metrics["avg_response_time"] = response_time
        else:
            current_avg = self.performance_metrics["avg_response_time"]
            self.performance_metrics["avg_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )

    async def _handle_scale_up(self, count: int):
        """Handle scale up operation."""
        logger.info(f"Scaling up by {count} instances (mock implementation)")
        # In a real implementation, this would:
        # 1. Launch new HF Spaces instances
        # 2. Add them to the load balancer
        # 3. Wait for health checks to pass

    async def _handle_scale_down(self, count: int):
        """Handle scale down operation."""
        logger.info(f"Scaling down by {count} instances (mock implementation)")
        # In a real implementation, this would:
        # 1. Select instances to terminate
        # 2. Drain their connections
        # 3. Remove from load balancer
        # 4. Terminate instances

    def get_architecture_status(self) -> Dict[str, Any]:
        """Get comprehensive architecture status."""
        status = {
            "load_balancer": self.load_balancer.get_instance_stats(),
            "performance_metrics": self.performance_metrics,
            "queue_size": self.request_queue.qsize(),
            "active_processors": len([p for p in self.queue_processors if not p.done()])
        }

        if self.auto_scaler:
            status["auto_scaler"] = self.auto_scaler.get_scaling_status()

        return status

    def get_recommendations(self) -> List[str]:
        """Get architecture optimization recommendations."""
        recommendations = []

        # Queue analysis
        queue_size = self.request_queue.qsize()
        if queue_size > 50:
            recommendations.append("High queue size detected - consider scaling up")

        # Response time analysis
        avg_response_time = self.performance_metrics["avg_response_time"]
        if avg_response_time > 5.0:
            recommendations.append("High response times - check instance health or scale up")

        # Error rate analysis
        total_requests = self.performance_metrics["total_requests"]
        if total_requests > 0:
            error_rate = self.performance_metrics["failed_requests"] / total_requests
            if error_rate > 0.05:  # 5% error rate
                recommendations.append("High error rate - investigate instance health")

        # Load balancer analysis
        lb_stats = self.load_balancer.get_instance_stats()
        if lb_stats["healthy_instances"] < 2:
            recommendations.append("Low instance count - consider adding redundancy")

        if not recommendations:
            recommendations.append("Architecture performing well - no immediate changes needed")

        return recommendations


# Factory function for easy integration
def create_scalable_architecture(hf_spaces_instances: List[str],
                               enable_auto_scaling: bool = True) -> ScalableArchitecture:
    """
    Create scalable architecture with HF Spaces instances.

    Args:
        hf_spaces_instances: List of HF Spaces URLs
        enable_auto_scaling: Enable auto-scaling

    Returns:
        Configured ScalableArchitecture instance
    """
    # Create instance configurations
    instances = []
    for i, url in enumerate(hf_spaces_instances):
        instances.append(InstanceConfig(
            instance_id=f"hf_space_{i}",
            endpoint_url=url,
            weight=1.0,
            max_concurrent_requests=10,
            health_check_url="/health"
        ))

    return ScalableArchitecture(
        initial_instances=instances,
        load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
        enable_auto_scaling=enable_auto_scaling,
        min_instances=1,
        max_instances=min(10, len(instances) * 3)
    )


# Export main classes
__all__ = [
    'ScalableArchitecture',
    'LoadBalancer',
    'AutoScaler',
    'CircuitBreaker',
    'InstanceConfig',
    'ScalingRule',
    'ScalingMetrics',
    'LoadBalancingStrategy',
    'ScalingMode',
    'create_scalable_architecture'
]