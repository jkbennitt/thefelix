"""
Premium Model Configuration for Felix Framework HF Pro Deployment

This module provides intelligent model selection and configuration optimized for
HuggingFace Pro accounts, ZeroGPU capabilities, and cost-effective deployment.

Features:
- Premium model access with Pro account benefits
- Intelligent model routing based on task complexity
- Cost optimization with performance balancing
- ZeroGPU memory management and batch processing
- Fallback chains for high availability
- Performance monitoring and adaptive selection
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from .hf_pro_optimization import ModelTier, ModelConfig, HFProOptimizer

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"       # Quick responses, basic processing
    MODERATE = "moderate"   # Standard analysis and reasoning
    COMPLEX = "complex"     # Deep analysis, multi-step reasoning
    RESEARCH = "research"   # Comprehensive research and synthesis
    CREATIVE = "creative"   # Creative writing and ideation


class ModelPerformanceRating(Enum):
    """Model performance ratings based on benchmarks."""
    EXCELLENT = "excellent"  # 90%+ benchmark scores
    GOOD = "good"           # 80-90% benchmark scores
    MODERATE = "moderate"   # 70-80% benchmark scores
    BASIC = "basic"         # 60-70% benchmark scores


@dataclass
class PremiumModelEntry:
    """Enhanced model configuration with Pro account features."""
    model_id: str
    tier: ModelTier
    performance_rating: ModelPerformanceRating
    max_tokens: int = 2048
    temperature_range: Tuple[float, float] = (0.1, 0.9)
    cost_per_1k_tokens: float = 0.10
    avg_response_time: float = 2.0
    context_window: int = 4096
    supports_zerogpu: bool = True
    supports_batching: bool = True
    concurrent_limit: int = 5
    memory_requirement_gb: float = 8.0
    specialties: List[str] = field(default_factory=list)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    pro_exclusive: bool = False
    fallback_models: List[str] = field(default_factory=list)


class PremiumModelManager:
    """
    Manages premium model access and intelligent selection for Felix Framework.

    Optimized for HuggingFace Pro accounts with advanced model routing,
    cost optimization, and performance monitoring.
    """

    # Premium model catalog with HF Pro exclusive models
    PREMIUM_MODEL_CATALOG = {
        # Ultra-premium 80B+ models (Pro exclusive)
        "qwen3-next-80b-instruct": PremiumModelEntry(
            model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
            tier=ModelTier.PREMIUM_80B,
            performance_rating=ModelPerformanceRating.EXCELLENT,
            max_tokens=4096,
            temperature_range=(0.1, 0.8),
            cost_per_1k_tokens=0.20,
            avg_response_time=5.0,
            context_window=32768,
            memory_requirement_gb=40.0,
            specialties=["reasoning", "analysis", "complex_qa"],
            benchmarks={"mmlu": 0.89, "hellaswag": 0.92, "arc": 0.88},
            pro_exclusive=True,
            fallback_models=["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
        ),

        "qwen3-next-80b-thinking": PremiumModelEntry(
            model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
            tier=ModelTier.PREMIUM_80B,
            performance_rating=ModelPerformanceRating.EXCELLENT,
            max_tokens=3072,
            temperature_range=(0.2, 0.7),
            cost_per_1k_tokens=0.18,
            avg_response_time=4.5,
            context_window=32768,
            memory_requirement_gb=40.0,
            specialties=["reasoning", "step_by_step", "problem_solving"],
            benchmarks={"gsm8k": 0.94, "math": 0.76, "reasoning": 0.91},
            pro_exclusive=True,
            fallback_models=["Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"]
        ),

        # High-performance 30B models
        "tongyi-deepresearch-30b": PremiumModelEntry(
            model_id="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
            tier=ModelTier.EFFICIENT_30B,
            performance_rating=ModelPerformanceRating.GOOD,
            max_tokens=2048,
            temperature_range=(0.1, 0.8),
            cost_per_1k_tokens=0.12,
            avg_response_time=3.0,
            context_window=16384,
            memory_requirement_gb=15.0,
            specialties=["research", "analysis", "synthesis"],
            benchmarks={"mmlu": 0.84, "hellaswag": 0.87, "arc": 0.82},
            fallback_models=["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
        ),

        "qwen3-coder-30b": PremiumModelEntry(
            model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            tier=ModelTier.EFFICIENT_30B,
            performance_rating=ModelPerformanceRating.GOOD,
            max_tokens=2048,
            temperature_range=(0.1, 0.6),
            cost_per_1k_tokens=0.10,
            avg_response_time=2.5,
            context_window=16384,
            memory_requirement_gb=15.0,
            specialties=["coding", "technical_analysis", "structured_output"],
            benchmarks={"humaneval": 0.78, "mbpp": 0.75, "code_quality": 0.85},
            fallback_models=["LLM360/K2-Think"]
        ),

        "ernie-4.5-21b-thinking": PremiumModelEntry(
            model_id="baidu/ERNIE-4.5-21B-A3B-Thinking",
            tier=ModelTier.EFFICIENT_30B,
            performance_rating=ModelPerformanceRating.GOOD,
            max_tokens=1536,
            temperature_range=(0.2, 0.7),
            cost_per_1k_tokens=0.08,
            avg_response_time=2.2,
            context_window=8192,
            memory_requirement_gb=12.0,
            specialties=["reasoning", "multilingual", "thinking"],
            benchmarks={"c_eval": 0.86, "reasoning": 0.83, "multilingual": 0.89},
            fallback_models=["LLM360/K2-Think"]
        ),

        # Efficient 7B-13B models
        "k2-think": PremiumModelEntry(
            model_id="LLM360/K2-Think",
            tier=ModelTier.FAST_7B,
            performance_rating=ModelPerformanceRating.GOOD,
            max_tokens=1024,
            temperature_range=(0.3, 0.8),
            cost_per_1k_tokens=0.05,
            avg_response_time=1.5,
            context_window=8192,
            memory_requirement_gb=7.0,
            specialties=["fast_reasoning", "balanced_performance"],
            benchmarks={"mmlu": 0.78, "hellaswag": 0.82, "speed": 0.95},
            fallback_models=["facebook/MobileLLM-R1-950M"]
        ),

        "llama-3.1-8b-instruct": PremiumModelEntry(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            tier=ModelTier.FAST_7B,
            performance_rating=ModelPerformanceRating.GOOD,
            max_tokens=1024,
            temperature_range=(0.1, 0.9),
            cost_per_1k_tokens=0.06,
            avg_response_time=1.8,
            context_window=8192,
            memory_requirement_gb=8.0,
            specialties=["general_purpose", "instruction_following"],
            benchmarks={"mmlu": 0.82, "instruction_following": 0.88},
            fallback_models=["facebook/MobileLLM-R1-950M"]
        ),

        # Edge models for fast responses
        "mobile-llm-950m": PremiumModelEntry(
            model_id="facebook/MobileLLM-R1-950M",
            tier=ModelTier.EDGE_1B,
            performance_rating=ModelPerformanceRating.MODERATE,
            max_tokens=512,
            temperature_range=(0.5, 0.9),
            cost_per_1k_tokens=0.02,
            avg_response_time=0.8,
            context_window=2048,
            memory_requirement_gb=2.0,
            specialties=["fast_response", "edge_computing", "mobile"],
            benchmarks={"speed": 0.98, "efficiency": 0.95, "basic_qa": 0.72},
            fallback_models=[]
        ),

        "ring-mini-2.0": PremiumModelEntry(
            model_id="inclusionAI/Ring-mini-2.0",
            tier=ModelTier.EDGE_1B,
            performance_rating=ModelPerformanceRating.MODERATE,
            max_tokens=512,
            temperature_range=(0.4, 0.8),
            cost_per_1k_tokens=0.03,
            avg_response_time=1.0,
            context_window=4096,
            memory_requirement_gb=3.0,
            specialties=["multilingual", "fast_processing"],
            benchmarks={"multilingual": 0.78, "speed": 0.90, "basic_reasoning": 0.70},
            fallback_models=["facebook/MobileLLM-R1-950M"]
        )
    }

    # Agent type to model selection strategy
    AGENT_MODEL_STRATEGIES = {
        "research": {
            "preferred_tiers": [ModelTier.FAST_7B, ModelTier.EFFICIENT_30B],
            "preferred_specialties": ["research", "fast_reasoning", "general_purpose"],
            "max_cost_per_request": 0.15,
            "min_performance_rating": ModelPerformanceRating.MODERATE
        },
        "analysis": {
            "preferred_tiers": [ModelTier.EFFICIENT_30B, ModelTier.PREMIUM_80B],
            "preferred_specialties": ["reasoning", "analysis", "step_by_step"],
            "max_cost_per_request": 0.25,
            "min_performance_rating": ModelPerformanceRating.GOOD
        },
        "synthesis": {
            "preferred_tiers": [ModelTier.PREMIUM_80B, ModelTier.EFFICIENT_30B],
            "preferred_specialties": ["synthesis", "reasoning", "complex_qa"],
            "max_cost_per_request": 0.35,
            "min_performance_rating": ModelPerformanceRating.GOOD
        },
        "critic": {
            "preferred_tiers": [ModelTier.EFFICIENT_30B, ModelTier.FAST_7B],
            "preferred_specialties": ["reasoning", "analysis", "thinking"],
            "max_cost_per_request": 0.20,
            "min_performance_rating": ModelPerformanceRating.GOOD
        },
        "general": {
            "preferred_tiers": [ModelTier.FAST_7B, ModelTier.EDGE_1B],
            "preferred_specialties": ["general_purpose", "fast_response", "balanced_performance"],
            "max_cost_per_request": 0.10,
            "min_performance_rating": ModelPerformanceRating.MODERATE
        }
    }

    def __init__(self,
                 hf_pro_optimizer: Optional[HFProOptimizer] = None,
                 enable_adaptive_selection: bool = True,
                 enable_cost_optimization: bool = True,
                 enable_performance_tracking: bool = True):
        """
        Initialize premium model manager.

        Args:
            hf_pro_optimizer: HF Pro optimizer for cost management
            enable_adaptive_selection: Enable adaptive model selection based on performance
            enable_cost_optimization: Enable cost-based model optimization
            enable_performance_tracking: Enable model performance tracking
        """
        self.hf_pro_optimizer = hf_pro_optimizer
        self.enable_adaptive_selection = enable_adaptive_selection
        self.enable_cost_optimization = enable_cost_optimization
        self.enable_performance_tracking = enable_performance_tracking

        # Performance tracking
        self.model_performance_history = {}
        self.selection_history = []
        self.cost_tracking = {}

        # Adaptive selection weights
        self.performance_weights = {
            "response_time": 0.3,
            "quality_score": 0.4,
            "cost_efficiency": 0.2,
            "success_rate": 0.1
        }

        logger.info("Premium Model Manager initialized")

    def select_optimal_model(self,
                           agent_type: str,
                           task_complexity: TaskComplexity,
                           budget_constraint: Optional[float] = None,
                           performance_priority: float = 0.5,
                           speed_priority: float = 0.3,
                           cost_priority: float = 0.2,
                           context_length_needed: int = 2048,
                           gpu_memory_available: float = 16.0) -> PremiumModelEntry:
        """
        Select optimal model based on comprehensive criteria.

        Args:
            agent_type: Type of Felix agent (research, analysis, synthesis, critic, general)
            task_complexity: Complexity level of the task
            budget_constraint: Maximum cost per request
            performance_priority: Weight for performance in selection (0-1)
            speed_priority: Weight for speed in selection (0-1)
            cost_priority: Weight for cost in selection (0-1)
            context_length_needed: Required context window size
            gpu_memory_available: Available GPU memory in GB

        Returns:
            Selected premium model configuration
        """
        # Normalize priorities
        total_priority = performance_priority + speed_priority + cost_priority
        if total_priority > 0:
            performance_priority /= total_priority
            speed_priority /= total_priority
            cost_priority /= total_priority

        # Get agent strategy
        strategy = self.AGENT_MODEL_STRATEGIES.get(agent_type, self.AGENT_MODEL_STRATEGIES["general"])

        # Filter models by constraints
        candidate_models = self._filter_models_by_constraints(
            strategy=strategy,
            task_complexity=task_complexity,
            budget_constraint=budget_constraint,
            context_length_needed=context_length_needed,
            gpu_memory_available=gpu_memory_available
        )

        if not candidate_models:
            # Fallback to basic model
            logger.warning(f"No models match constraints for {agent_type}, using fallback")
            return self.PREMIUM_MODEL_CATALOG["mobile-llm-950m"]

        # Score and rank models
        scored_models = []
        for model in candidate_models:
            score = self._calculate_model_score(
                model=model,
                task_complexity=task_complexity,
                performance_priority=performance_priority,
                speed_priority=speed_priority,
                cost_priority=cost_priority
            )
            scored_models.append((score, model))

        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        selected_model = scored_models[0][1]

        # Track selection
        self._track_selection(agent_type, task_complexity, selected_model, scored_models[0][0])

        logger.info(f"Selected {selected_model.model_id} for {agent_type} agent (score: {scored_models[0][0]:.3f})")
        return selected_model

    def _filter_models_by_constraints(self,
                                     strategy: Dict[str, Any],
                                     task_complexity: TaskComplexity,
                                     budget_constraint: Optional[float],
                                     context_length_needed: int,
                                     gpu_memory_available: float) -> List[PremiumModelEntry]:
        """Filter models by hard constraints."""
        candidates = []

        for model in self.PREMIUM_MODEL_CATALOG.values():
            # Check tier preference
            if model.tier not in strategy["preferred_tiers"]:
                continue

            # Check performance rating
            if model.performance_rating.value < strategy["min_performance_rating"].value:
                continue

            # Check budget constraint
            max_cost = budget_constraint or strategy["max_cost_per_request"]
            estimated_cost = (model.max_tokens / 1000) * model.cost_per_1k_tokens
            if estimated_cost > max_cost:
                continue

            # Check context window
            if model.context_window < context_length_needed:
                continue

            # Check GPU memory requirement
            if model.memory_requirement_gb > gpu_memory_available:
                continue

            # Check complexity alignment
            if task_complexity == TaskComplexity.SIMPLE and model.tier == ModelTier.PREMIUM_80B:
                continue  # Don't use premium models for simple tasks
            elif task_complexity == TaskComplexity.RESEARCH and model.tier == ModelTier.EDGE_1B:
                continue  # Don't use edge models for research tasks

            candidates.append(model)

        return candidates

    def _calculate_model_score(self,
                              model: PremiumModelEntry,
                              task_complexity: TaskComplexity,
                              performance_priority: float,
                              speed_priority: float,
                              cost_priority: float) -> float:
        """Calculate weighted score for model selection."""
        # Performance score (0-1)
        performance_ratings = {
            ModelPerformanceRating.EXCELLENT: 1.0,
            ModelPerformanceRating.GOOD: 0.8,
            ModelPerformanceRating.MODERATE: 0.6,
            ModelPerformanceRating.BASIC: 0.4
        }
        performance_score = performance_ratings[model.performance_rating]

        # Speed score (inverse of response time, normalized)
        max_response_time = 10.0  # Normalize against 10 second max
        speed_score = max(0, (max_response_time - model.avg_response_time) / max_response_time)

        # Cost score (inverse of cost, normalized)
        max_cost = 0.25  # Normalize against $0.25 per 1k tokens
        cost_score = max(0, (max_cost - model.cost_per_1k_tokens) / max_cost)

        # Specialty bonus
        specialty_bonus = 0.0
        if task_complexity == TaskComplexity.RESEARCH and "research" in model.specialties:
            specialty_bonus += 0.1
        elif task_complexity == TaskComplexity.COMPLEX and "reasoning" in model.specialties:
            specialty_bonus += 0.1
        elif task_complexity == TaskComplexity.CREATIVE and "creative" in model.specialties:
            specialty_bonus += 0.1

        # Historical performance bonus
        history_bonus = 0.0
        if self.enable_adaptive_selection and model.model_id in self.model_performance_history:
            history = self.model_performance_history[model.model_id]
            if history.get("success_rate", 0.5) > 0.9:
                history_bonus += 0.05
            if history.get("avg_quality", 0.5) > 0.8:
                history_bonus += 0.05

        # Calculate weighted score
        total_score = (
            performance_score * performance_priority +
            speed_score * speed_priority +
            cost_score * cost_priority +
            specialty_bonus +
            history_bonus
        )

        return total_score

    def _track_selection(self,
                        agent_type: str,
                        task_complexity: TaskComplexity,
                        selected_model: PremiumModelEntry,
                        score: float):
        """Track model selection for adaptive learning."""
        selection_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": agent_type,
            "task_complexity": task_complexity.value,
            "model_id": selected_model.model_id,
            "model_tier": selected_model.tier.value,
            "selection_score": score,
            "estimated_cost": (selected_model.max_tokens / 1000) * selected_model.cost_per_1k_tokens
        }

        self.selection_history.append(selection_record)

        # Keep only last 1000 selections
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]

    def update_model_performance(self,
                               model_id: str,
                               response_time: float,
                               quality_score: float,
                               success: bool,
                               actual_cost: float):
        """Update model performance metrics for adaptive selection."""
        if not self.enable_performance_tracking:
            return

        if model_id not in self.model_performance_history:
            self.model_performance_history[model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_response_time": 0.0,
                "avg_quality": 0.0,
                "total_cost": 0.0,
                "last_updated": datetime.now()
            }

        history = self.model_performance_history[model_id]

        # Update counters
        history["total_requests"] += 1
        if success:
            history["successful_requests"] += 1

        # Update running averages
        n = history["total_requests"]
        history["avg_response_time"] = ((history["avg_response_time"] * (n - 1)) + response_time) / n
        history["avg_quality"] = ((history["avg_quality"] * (n - 1)) + quality_score) / n
        history["total_cost"] += actual_cost
        history["success_rate"] = history["successful_requests"] / history["total_requests"]
        history["last_updated"] = datetime.now()

    def get_model_recommendations(self,
                                agent_types: List[str],
                                task_complexity: TaskComplexity,
                                total_budget: float) -> Dict[str, PremiumModelEntry]:
        """Get model recommendations for multiple agent types within budget."""
        recommendations = {}
        remaining_budget = total_budget

        # Sort agent types by importance (synthesis gets premium models first)
        importance_order = ["synthesis", "analysis", "critic", "research", "general"]
        sorted_agent_types = sorted(agent_types,
                                  key=lambda x: importance_order.index(x) if x in importance_order else 999)

        for agent_type in sorted_agent_types:
            budget_per_agent = remaining_budget / max(1, len(sorted_agent_types))

            selected_model = self.select_optimal_model(
                agent_type=agent_type,
                task_complexity=task_complexity,
                budget_constraint=budget_per_agent,
                performance_priority=0.6 if agent_type in ["synthesis", "analysis"] else 0.4,
                speed_priority=0.2 if agent_type in ["synthesis", "analysis"] else 0.4,
                cost_priority=0.2
            )

            recommendations[agent_type] = selected_model
            estimated_cost = (selected_model.max_tokens / 1000) * selected_model.cost_per_1k_tokens
            remaining_budget -= estimated_cost
            sorted_agent_types.remove(agent_type)

        return recommendations

    def get_fallback_model(self, primary_model_id: str) -> Optional[PremiumModelEntry]:
        """Get fallback model for failed primary model."""
        for model in self.PREMIUM_MODEL_CATALOG.values():
            if model.model_id == primary_model_id and model.fallback_models:
                fallback_id = model.fallback_models[0]
                for fallback_model in self.PREMIUM_MODEL_CATALOG.values():
                    if fallback_model.model_id == fallback_id:
                        return fallback_model

        # Default fallback to edge model
        return self.PREMIUM_MODEL_CATALOG["mobile-llm-950m"]

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data."""
        if not self.selection_history:
            return {"message": "No selection history available"}

        # Model usage statistics
        model_usage = {}
        for selection in self.selection_history:
            model_id = selection["model_id"]
            if model_id not in model_usage:
                model_usage[model_id] = {"count": 0, "total_cost": 0.0}
            model_usage[model_id]["count"] += 1
            model_usage[model_id]["total_cost"] += selection["estimated_cost"]

        # Agent type preferences
        agent_preferences = {}
        for selection in self.selection_history:
            agent_type = selection["agent_type"]
            if agent_type not in agent_preferences:
                agent_preferences[agent_type] = {}

            tier = selection["model_tier"]
            agent_preferences[agent_type][tier] = agent_preferences[agent_type].get(tier, 0) + 1

        # Performance trends
        performance_trends = {}
        for model_id, history in self.model_performance_history.items():
            performance_trends[model_id] = {
                "success_rate": history.get("success_rate", 0),
                "avg_response_time": history.get("avg_response_time", 0),
                "avg_quality": history.get("avg_quality", 0),
                "total_requests": history.get("total_requests", 0),
                "cost_efficiency": history.get("total_cost", 0) / max(1, history.get("total_requests", 1))
            }

        return {
            "model_usage": model_usage,
            "agent_preferences": agent_preferences,
            "performance_trends": performance_trends,
            "total_selections": len(self.selection_history),
            "total_models_used": len(set(s["model_id"] for s in self.selection_history)),
            "avg_selection_score": np.mean([s["selection_score"] for s in self.selection_history]),
            "cost_distribution": {
                tier.value: sum(s["estimated_cost"] for s in self.selection_history
                               if s["model_tier"] == tier.value)
                for tier in ModelTier
            }
        }


# Factory function for easy integration
def create_premium_model_manager(hf_pro_optimizer: Optional[HFProOptimizer] = None) -> PremiumModelManager:
    """
    Create premium model manager with recommended settings.

    Args:
        hf_pro_optimizer: Optional HF Pro optimizer instance

    Returns:
        Configured PremiumModelManager instance
    """
    return PremiumModelManager(
        hf_pro_optimizer=hf_pro_optimizer,
        enable_adaptive_selection=True,
        enable_cost_optimization=True,
        enable_performance_tracking=True
    )


# Export main classes
__all__ = [
    'PremiumModelManager',
    'PremiumModelEntry',
    'TaskComplexity',
    'ModelPerformanceRating',
    'create_premium_model_manager'
]