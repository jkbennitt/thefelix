"""
HuggingFace Pro Account Optimization Configuration for Felix Framework

This module provides comprehensive optimization strategies for leveraging HF Pro account
features, ZeroGPU capabilities, and cost-effective deployment while maximizing performance.

Key Features:
- Premium model access with intelligent model selection
- ZeroGPU optimization for cost efficiency
- Advanced caching strategies for reduced compute costs
- Performance monitoring with cost analytics
- Scalable architecture for increased user loads
- Automated resource allocation and optimization

HF Pro Benefits Leveraged:
- Higher concurrent user limits
- Priority access to premium models
- Enhanced ZeroGPU allocation and priority
- Advanced analytics and usage monitoring
- Priority support and faster deployment queues
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers based on HF Pro access and performance."""
    PREMIUM_80B = "premium_80b"  # Qwen3-Next-80B-A3B series
    EFFICIENT_30B = "efficient_30b"  # Specialized models
    FAST_7B = "fast_7b"  # Quick response models
    EDGE_1B = "edge_1b"  # Ultra-fast edge models


class ResourceUsageLevel(Enum):
    """Resource usage levels for cost optimization."""
    MINIMAL = "minimal"  # <10% GPU usage
    MODERATE = "moderate"  # 10-30% GPU usage
    STANDARD = "standard"  # 30-60% GPU usage
    INTENSIVE = "intensive"  # 60-80% GPU usage
    MAXIMUM = "maximum"  # 80%+ GPU usage


@dataclass
class ModelConfig:
    """Configuration for a premium model."""
    model_id: str
    tier: ModelTier
    max_tokens: int = 1024
    temperature: float = 0.7
    cost_per_token: float = 0.0001
    avg_response_time: float = 2.0
    quality_score: float = 0.85
    supports_zerogpu: bool = True
    concurrent_limit: int = 5


@dataclass
class UsageMetrics:
    """Usage and cost metrics tracking."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_users: int = 0
    peak_concurrent: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class HFProOptimizer:
    """
    HuggingFace Pro account optimizer for Felix Framework.

    Provides intelligent model selection, cost optimization, and performance
    monitoring specifically designed for HF Pro account features.
    """

    # Premium model configurations optimized for Felix Framework
    PREMIUM_MODELS = {
        ModelTier.PREMIUM_80B: [
            ModelConfig(
                model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
                tier=ModelTier.PREMIUM_80B,
                max_tokens=2048,
                temperature=0.1,
                cost_per_token=0.0002,
                avg_response_time=4.5,
                quality_score=0.95,
                concurrent_limit=3
            ),
            ModelConfig(
                model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
                tier=ModelTier.PREMIUM_80B,
                max_tokens=1536,
                temperature=0.3,
                cost_per_token=0.00018,
                avg_response_time=3.8,
                quality_score=0.93,
                concurrent_limit=3
            )
        ],
        ModelTier.EFFICIENT_30B: [
            ModelConfig(
                model_id="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
                tier=ModelTier.EFFICIENT_30B,
                max_tokens=1024,
                temperature=0.5,
                cost_per_token=0.00012,
                avg_response_time=2.5,
                quality_score=0.88,
                concurrent_limit=5
            ),
            ModelConfig(
                model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                tier=ModelTier.EFFICIENT_30B,
                max_tokens=1024,
                temperature=0.2,
                cost_per_token=0.0001,
                avg_response_time=2.2,
                quality_score=0.86,
                concurrent_limit=6
            )
        ],
        ModelTier.FAST_7B: [
            ModelConfig(
                model_id="LLM360/K2-Think",
                tier=ModelTier.FAST_7B,
                max_tokens=512,
                temperature=0.7,
                cost_per_token=0.00005,
                avg_response_time=1.2,
                quality_score=0.82,
                concurrent_limit=10
            )
        ],
        ModelTier.EDGE_1B: [
            ModelConfig(
                model_id="facebook/MobileLLM-R1-950M",
                tier=ModelTier.EDGE_1B,
                max_tokens=256,
                temperature=0.8,
                cost_per_token=0.00002,
                avg_response_time=0.5,
                quality_score=0.75,
                concurrent_limit=20
            )
        ]
    }

    # Felix agent type to model tier mapping
    AGENT_MODEL_MAPPING = {
        "synthesis": ModelTier.PREMIUM_80B,  # Highest quality output
        "analysis": ModelTier.EFFICIENT_30B,  # Balanced performance
        "research": ModelTier.FAST_7B,  # Quick exploration
        "critic": ModelTier.EFFICIENT_30B,  # Thorough evaluation
        "general": ModelTier.FAST_7B  # Default fast processing
    }

    def __init__(self,
                 hf_token: Optional[str] = None,
                 monthly_budget: float = 100.0,
                 target_cost_per_request: float = 0.05,
                 enable_advanced_caching: bool = True,
                 enable_cost_alerts: bool = True):
        """
        Initialize HF Pro optimizer.

        Args:
            hf_token: HuggingFace API token with Pro access
            monthly_budget: Monthly budget in USD
            target_cost_per_request: Target cost per Felix request
            enable_advanced_caching: Enable intelligent caching
            enable_cost_alerts: Enable cost monitoring alerts
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.monthly_budget = monthly_budget
        self.target_cost_per_request = target_cost_per_request
        self.enable_advanced_caching = enable_advanced_caching
        self.enable_cost_alerts = enable_cost_alerts

        # Initialize metrics tracking
        self.metrics = UsageMetrics()
        self.hourly_metrics: Dict[str, UsageMetrics] = defaultdict(UsageMetrics)
        self.model_performance: Dict[str, Dict] = defaultdict(dict)

        # Advanced caching system
        self.cache = OrderedDict() if enable_advanced_caching else None
        self.cache_stats = {"hits": 0, "misses": 0, "size": 0}

        # Resource monitoring
        self.resource_usage = ResourceUsageLevel.MINIMAL
        self.concurrent_requests = 0
        self.request_queue = asyncio.Queue()

        logger.info(f"HF Pro Optimizer initialized - Budget: ${monthly_budget}/month")

    def select_optimal_model(self,
                           agent_type: str,
                           task_complexity: str,
                           current_load: int = 0,
                           budget_remaining: float = 1.0) -> ModelConfig:
        """
        Select optimal model based on agent type, complexity, and constraints.

        Args:
            agent_type: Type of Felix agent requesting model
            task_complexity: Complexity level (demo/simple/medium/complex/research)
            current_load: Current system load (0-100)
            budget_remaining: Remaining budget percentage (0.0-1.0)

        Returns:
            Optimal ModelConfig for the request
        """
        # Get base tier for agent type
        base_tier = self.AGENT_MODEL_MAPPING.get(agent_type, ModelTier.FAST_7B)

        # Adjust tier based on complexity and constraints
        if task_complexity in ["research", "complex"] and budget_remaining > 0.3:
            # Use premium models for complex tasks if budget allows
            if base_tier in [ModelTier.EFFICIENT_30B, ModelTier.PREMIUM_80B]:
                target_tier = ModelTier.PREMIUM_80B
            else:
                target_tier = ModelTier.EFFICIENT_30B
        elif current_load > 70 or budget_remaining < 0.2:
            # Use efficient models under high load or low budget
            if base_tier == ModelTier.PREMIUM_80B:
                target_tier = ModelTier.EFFICIENT_30B
            elif base_tier == ModelTier.EFFICIENT_30B:
                target_tier = ModelTier.FAST_7B
            else:
                target_tier = ModelTier.EDGE_1B
        else:
            target_tier = base_tier

        # Select best model from tier
        available_models = self.PREMIUM_MODELS.get(target_tier, [])
        if not available_models:
            # Fallback to fast tier
            available_models = self.PREMIUM_MODELS[ModelTier.FAST_7B]

        # Select model with best performance/cost ratio for current load
        best_model = min(available_models,
                        key=lambda m: self._calculate_selection_score(m, current_load))

        logger.info(f"Selected {best_model.model_id} for {agent_type} agent (complexity: {task_complexity})")
        return best_model

    def _calculate_selection_score(self, model: ModelConfig, current_load: int) -> float:
        """Calculate model selection score (lower is better)."""
        # Base score from cost per token
        score = model.cost_per_token * 1000

        # Adjust for current load (prefer faster models under high load)
        if current_load > 50:
            score += model.avg_response_time * 0.5

        # Prefer models with higher quality
        score -= model.quality_score * 0.2

        # Prefer models with higher concurrent limits under load
        if current_load > 30:
            score -= (model.concurrent_limit / 20) * 0.1

        return score

    @staticmethod
    def create_zerogpu_decorator():
        """Create ZeroGPU decorator for cost-efficient GPU usage."""
        try:
            import spaces
            return spaces.GPU(duration=120)  # 2-minute GPU allocation
        except ImportError:
            logger.warning("ZeroGPU not available - running without GPU optimization")
            return lambda x: x

    def estimate_request_cost(self,
                            agent_count: int,
                            complexity: str,
                            estimated_tokens_per_agent: int = 300) -> Dict[str, Any]:
        """
        Estimate cost for a Felix Framework request.

        Args:
            agent_count: Number of agents in the request
            complexity: Task complexity level
            estimated_tokens_per_agent: Estimated tokens per agent

        Returns:
            Cost estimation with breakdown
        """
        total_cost = 0.0
        model_breakdown = {}

        # Estimate cost for each agent type
        agent_types = ["research", "analysis", "synthesis", "critic"]
        agents_per_type = agent_count // len(agent_types)

        for agent_type in agent_types:
            model = self.select_optimal_model(
                agent_type=agent_type,
                task_complexity=complexity,
                budget_remaining=1.0  # Full budget for estimation
            )

            type_cost = (agents_per_type * estimated_tokens_per_agent *
                        model.cost_per_token)
            total_cost += type_cost

            model_breakdown[agent_type] = {
                "model_id": model.model_id,
                "agents": agents_per_type,
                "estimated_tokens": agents_per_type * estimated_tokens_per_agent,
                "cost": type_cost
            }

        return {
            "total_estimated_cost": total_cost,
            "cost_per_agent": total_cost / agent_count,
            "model_breakdown": model_breakdown,
            "within_target": total_cost <= self.target_cost_per_request,
            "budget_utilization": total_cost / self.target_cost_per_request
        }

    def get_cache_key(self, task_input: str, agent_type: str, complexity: str) -> str:
        """Generate cache key for task input."""
        import hashlib
        content = f"{task_input}_{agent_type}_{complexity}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        if not self.cache:
            return None

        if cache_key in self.cache:
            # Move to end (LRU)
            result = self.cache.pop(cache_key)
            self.cache[cache_key] = result
            self.cache_stats["hits"] += 1
            return result

        self.cache_stats["misses"] += 1
        return None

    def cache_result(self, cache_key: str, result: Dict[str, Any], max_cache_size: int = 1000):
        """Cache a result."""
        if not self.cache:
            return

        # Remove oldest if at capacity
        if len(self.cache) >= max_cache_size and cache_key not in self.cache:
            self.cache.popitem(last=False)

        self.cache[cache_key] = result
        self.cache_stats["size"] = len(self.cache)

    def update_metrics(self,
                      model_id: str,
                      tokens_used: int,
                      response_time: float,
                      success: bool,
                      cost: float):
        """Update usage metrics."""
        # Update global metrics
        self.metrics.total_requests += 1
        self.metrics.total_tokens += tokens_used
        self.metrics.total_cost += cost

        # Update running averages
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) /
            self.metrics.total_requests
        )

        if success:
            success_count = self.metrics.total_requests * self.metrics.success_rate
            self.metrics.success_rate = (success_count + 1) / self.metrics.total_requests
        else:
            success_count = self.metrics.total_requests * self.metrics.success_rate
            self.metrics.success_rate = success_count / self.metrics.total_requests

        # Update hourly metrics
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        hourly = self.hourly_metrics[hour_key]
        hourly.total_requests += 1
        hourly.total_tokens += tokens_used
        hourly.total_cost += cost

        # Update model performance tracking
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {
                "requests": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "total_cost": 0.0
            }

        model_stats = self.model_performance[model_id]
        model_stats["requests"] += 1
        model_stats["avg_response_time"] = (
            (model_stats["avg_response_time"] * (model_stats["requests"] - 1) + response_time) /
            model_stats["requests"]
        )
        model_stats["total_cost"] += cost

        # Check for cost alerts
        if self.enable_cost_alerts:
            self._check_cost_alerts()

    def _check_cost_alerts(self):
        """Check for cost threshold alerts."""
        daily_budget = self.monthly_budget / 30
        current_daily_cost = sum(
            metrics.total_cost for hour, metrics in self.hourly_metrics.items()
            if hour.startswith(datetime.now().strftime("%Y-%m-%d"))
        )

        if current_daily_cost > daily_budget * 0.8:
            logger.warning(f"Daily cost approaching limit: ${current_daily_cost:.2f} / ${daily_budget:.2f}")

        if current_daily_cost > daily_budget:
            logger.error(f"Daily budget exceeded: ${current_daily_cost:.2f} / ${daily_budget:.2f}")

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        cache_hit_rate = (
            self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"])
            if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        )

        return {
            "overview": {
                "total_requests": self.metrics.total_requests,
                "total_cost": self.metrics.total_cost,
                "avg_cost_per_request": (
                    self.metrics.total_cost / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                "success_rate": self.metrics.success_rate,
                "avg_response_time": self.metrics.avg_response_time
            },
            "budget": {
                "monthly_budget": self.monthly_budget,
                "spent_this_month": self.metrics.total_cost,
                "remaining_budget": self.monthly_budget - self.metrics.total_cost,
                "burn_rate": self.metrics.total_cost / max(1, (datetime.now().day)),
                "projected_monthly": self.metrics.total_cost / max(1, (datetime.now().day)) * 30
            },
            "performance": {
                "cache_hit_rate": cache_hit_rate,
                "cache_size": self.cache_stats["size"],
                "concurrent_users": self.metrics.concurrent_users,
                "peak_concurrent": self.metrics.peak_concurrent
            },
            "models": {
                model_id: {
                    "requests": stats["requests"],
                    "avg_response_time": stats["avg_response_time"],
                    "total_cost": stats["total_cost"],
                    "cost_per_request": stats["total_cost"] / max(1, stats["requests"])
                }
                for model_id, stats in self.model_performance.items()
            },
            "optimization_suggestions": self._get_optimization_suggestions()
        }

    def _get_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on usage patterns."""
        suggestions = []

        # Cache efficiency
        cache_hit_rate = (
            self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"])
            if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        )

        if cache_hit_rate < 0.3:
            suggestions.append("Consider increasing cache size or improving cache key strategy")

        # Cost efficiency
        avg_cost = (
            self.metrics.total_cost / self.metrics.total_requests
            if self.metrics.total_requests > 0 else 0
        )

        if avg_cost > self.target_cost_per_request * 1.2:
            suggestions.append("Consider using more efficient models for routine tasks")

        # Performance optimization
        if self.metrics.avg_response_time > 5.0:
            suggestions.append("Consider using faster models or reducing complexity for real-time tasks")

        # Budget management
        if self.metrics.total_cost > self.monthly_budget * 0.8:
            suggestions.append("Approaching monthly budget limit - consider cost controls")

        return suggestions

    async def optimize_request_flow(self,
                                  task_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a batch of Felix Framework requests for cost and performance.

        Args:
            task_requests: List of task request dictionaries

        Returns:
            Optimized request configurations
        """
        optimized_requests = []

        # Sort requests by priority and complexity
        sorted_requests = sorted(task_requests,
                               key=lambda x: (x.get("priority", 5), x.get("complexity", "medium")))

        current_load = len(sorted_requests)
        budget_remaining = (
            (self.monthly_budget - self.metrics.total_cost) / self.monthly_budget
        )

        for i, request in enumerate(sorted_requests):
            # Adjust remaining budget based on position in queue
            adjusted_budget = budget_remaining * (1 - i / len(sorted_requests))

            # Select optimal model configuration
            optimal_model = self.select_optimal_model(
                agent_type=request.get("agent_type", "general"),
                task_complexity=request.get("complexity", "medium"),
                current_load=current_load,
                budget_remaining=adjusted_budget
            )

            # Check cache first
            cache_key = self.get_cache_key(
                request.get("task_input", ""),
                request.get("agent_type", "general"),
                request.get("complexity", "medium")
            )

            cached_result = self.get_cached_result(cache_key)

            optimized_request = {
                **request,
                "model_config": optimal_model,
                "cache_key": cache_key,
                "cached_result": cached_result,
                "estimated_cost": self.estimate_request_cost(
                    agent_count=request.get("agent_count", 8),
                    complexity=request.get("complexity", "medium")
                ),
                "optimization_applied": True
            }

            optimized_requests.append(optimized_request)

        return optimized_requests


# Factory function for easy integration
def create_hf_pro_optimizer(monthly_budget: float = 100.0) -> HFProOptimizer:
    """
    Create HF Pro optimizer with recommended settings.

    Args:
        monthly_budget: Monthly budget in USD

    Returns:
        Configured HFProOptimizer instance
    """
    return HFProOptimizer(
        monthly_budget=monthly_budget,
        target_cost_per_request=0.05,  # 5 cents per Felix request
        enable_advanced_caching=True,
        enable_cost_alerts=True
    )


# Export main classes
__all__ = [
    'HFProOptimizer',
    'ModelTier',
    'ModelConfig',
    'ResourceUsageLevel',
    'UsageMetrics',
    'create_hf_pro_optimizer'
]