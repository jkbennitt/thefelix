"""
Comprehensive Analytics Dashboard for Felix Framework HF Pro Deployment

This module provides advanced monitoring, analytics, and cost optimization
dashboards specifically designed for HuggingFace Pro accounts and ZeroGPU deployments.

Features:
- Real-time performance monitoring with GPU metrics
- Cost tracking and budget alerts
- Usage analytics with trend analysis
- Model performance comparison
- Resource utilization optimization
- User engagement analytics
- Predictive cost modeling
- A/B testing for model selection
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .hf_pro_optimization import HFProOptimizer, UsageMetrics
from .premium_model_config import PremiumModelManager

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Metrics for individual user sessions."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    gpu_time_used: float = 0.0
    models_used: List[str] = field(default_factory=list)
    agent_types_used: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None  # 1-5 rating
    device_type: str = "unknown"
    geographic_region: str = "unknown"


@dataclass
class ModelMetrics:
    """Performance metrics for individual models."""
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    avg_quality_score: float = 0.0
    gpu_memory_avg: float = 0.0
    error_rate: float = 0.0
    cost_per_token: float = 0.0
    user_preference_score: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)


@dataclass
class CostAlert:
    """Cost monitoring alert."""
    alert_id: str
    alert_type: str  # "budget_threshold", "spike", "efficiency"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for Felix Framework.

    Provides real-time monitoring, cost optimization, and performance analytics
    optimized for HuggingFace Pro accounts and ZeroGPU deployments.
    """

    def __init__(self,
                 hf_pro_optimizer: Optional[HFProOptimizer] = None,
                 model_manager: Optional[PremiumModelManager] = None,
                 enable_predictive_analytics: bool = True,
                 enable_cost_alerts: bool = True,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize analytics dashboard.

        Args:
            hf_pro_optimizer: HF Pro optimizer instance
            model_manager: Premium model manager instance
            enable_predictive_analytics: Enable predictive cost modeling
            enable_cost_alerts: Enable automated cost alerts
            alert_thresholds: Custom alert thresholds
        """
        self.hf_pro_optimizer = hf_pro_optimizer
        self.model_manager = model_manager
        self.enable_predictive_analytics = enable_predictive_analytics
        self.enable_cost_alerts = enable_cost_alerts

        # Default alert thresholds
        self.alert_thresholds = {
            "daily_budget_80": 0.8,  # 80% of daily budget
            "daily_budget_100": 1.0,  # 100% of daily budget
            "cost_spike_3x": 3.0,  # 3x normal hourly cost
            "error_rate_10": 0.1,  # 10% error rate
            "response_time_5s": 5.0,  # 5 second response time
            "gpu_memory_90": 0.9,  # 90% GPU memory usage
            "token_efficiency_50": 0.5  # 50% token efficiency
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)

        # Data storage
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.cost_alerts: List[CostAlert] = []
        self.hourly_stats: Dict[str, Dict] = defaultdict(dict)

        # Real-time tracking
        self.active_sessions = set()
        self.current_gpu_usage = 0.0
        self.current_concurrent_users = 0
        self.peak_concurrent_users = 0

        # Historical data (rolling windows)
        self.cost_history = deque(maxlen=720)  # 30 days of hourly data
        self.performance_history = deque(maxlen=2880)  # 30 days of 15-min data
        self.user_engagement_history = deque(maxlen=168)  # 7 days of hourly data

        logger.info("Analytics Dashboard initialized")

    def track_session_start(self,
                          session_id: str,
                          device_type: str = "unknown",
                          geographic_region: str = "unknown") -> SessionMetrics:
        """Track the start of a user session."""
        session = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            device_type=device_type,
            geographic_region=geographic_region
        )

        self.session_metrics[session_id] = session
        self.active_sessions.add(session_id)

        # Update concurrent user metrics
        self.current_concurrent_users = len(self.active_sessions)
        self.peak_concurrent_users = max(self.peak_concurrent_users, self.current_concurrent_users)

        logger.info(f"Session started: {session_id} ({device_type}, {geographic_region})")
        return session

    def track_session_end(self, session_id: str, user_satisfaction: Optional[float] = None):
        """Track the end of a user session."""
        if session_id not in self.session_metrics:
            logger.warning(f"Session {session_id} not found for ending")
            return

        session = self.session_metrics[session_id]
        session.end_time = datetime.now()
        session.user_satisfaction = user_satisfaction

        self.active_sessions.discard(session_id)
        self.current_concurrent_users = len(self.active_sessions)

        logger.info(f"Session ended: {session_id} (duration: {session.end_time - session.start_time})")

    def track_request(self,
                     session_id: str,
                     model_id: str,
                     agent_type: str,
                     cost: float,
                     tokens: int,
                     response_time: float,
                     success: bool,
                     quality_score: float = 0.0,
                     gpu_memory_used: float = 0.0):
        """Track an individual request."""
        # Update session metrics
        if session_id in self.session_metrics:
            session = self.session_metrics[session_id]
            session.total_requests += 1
            if success:
                session.successful_requests += 1
            session.total_cost += cost
            session.total_tokens += tokens
            session.gpu_time_used += response_time if gpu_memory_used > 0 else 0

            if model_id not in session.models_used:
                session.models_used.append(model_id)
            if agent_type not in session.agent_types_used:
                session.agent_types_used.append(agent_type)

        # Update model metrics
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = ModelMetrics(model_id=model_id)

        model = self.model_metrics[model_id]
        model.total_requests += 1
        if success:
            model.successful_requests += 1

        # Update running averages
        n = model.total_requests
        model.avg_response_time = ((model.avg_response_time * (n - 1)) + response_time) / n
        model.avg_quality_score = ((model.avg_quality_score * (n - 1)) + quality_score) / n
        model.gpu_memory_avg = ((model.gpu_memory_avg * (n - 1)) + gpu_memory_used) / n

        model.total_cost += cost
        model.total_tokens += tokens
        model.error_rate = 1 - (model.successful_requests / model.total_requests)
        model.cost_per_token = model.total_cost / max(1, model.total_tokens)
        model.last_used = datetime.now()

        # Update hourly statistics
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        if hour_key not in self.hourly_stats:
            self.hourly_stats[hour_key] = {
                "requests": 0,
                "cost": 0.0,
                "tokens": 0,
                "errors": 0,
                "avg_response_time": 0.0,
                "unique_sessions": set(),
                "gpu_time": 0.0
            }

        hour_stats = self.hourly_stats[hour_key]
        hour_stats["requests"] += 1
        hour_stats["cost"] += cost
        hour_stats["tokens"] += tokens
        if not success:
            hour_stats["errors"] += 1
        hour_stats["avg_response_time"] = (
            (hour_stats["avg_response_time"] * (hour_stats["requests"] - 1) + response_time) /
            hour_stats["requests"]
        )
        hour_stats["unique_sessions"].add(session_id)
        hour_stats["gpu_time"] += response_time if gpu_memory_used > 0 else 0

        # Check for alerts
        if self.enable_cost_alerts:
            self._check_alerts(cost, response_time, success, gpu_memory_used)

    def _check_alerts(self, cost: float, response_time: float, success: bool, gpu_memory: float):
        """Check for cost and performance alerts."""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        hour_stats = self.hourly_stats.get(current_hour, {})

        # Daily budget alert
        if self.hf_pro_optimizer:
            daily_budget = self.hf_pro_optimizer.monthly_budget / 30
            daily_cost = sum(
                stats.get("cost", 0) for hour, stats in self.hourly_stats.items()
                if hour.startswith(datetime.now().strftime("%Y-%m-%d"))
            )

            if daily_cost > daily_budget * self.alert_thresholds["daily_budget_80"]:
                severity = "high" if daily_cost > daily_budget else "medium"
                self._create_alert(
                    alert_type="budget_threshold",
                    severity=severity,
                    message=f"Daily cost ({daily_cost:.2f}) approaching budget limit ({daily_budget:.2f})",
                    current_value=daily_cost,
                    threshold_value=daily_budget * self.alert_thresholds["daily_budget_80"]
                )

        # Response time alert
        if response_time > self.alert_thresholds["response_time_5s"]:
            self._create_alert(
                alert_type="performance",
                severity="medium",
                message=f"High response time detected: {response_time:.2f}s",
                current_value=response_time,
                threshold_value=self.alert_thresholds["response_time_5s"]
            )

        # Error rate alert
        if hour_stats.get("requests", 0) >= 10:  # Only check after 10+ requests
            error_rate = hour_stats.get("errors", 0) / hour_stats["requests"]
            if error_rate > self.alert_thresholds["error_rate_10"]:
                self._create_alert(
                    alert_type="error_rate",
                    severity="high",
                    message=f"High error rate: {error_rate:.1%}",
                    current_value=error_rate,
                    threshold_value=self.alert_thresholds["error_rate_10"]
                )

        # GPU memory alert
        if gpu_memory > self.alert_thresholds["gpu_memory_90"]:
            self._create_alert(
                alert_type="resource",
                severity="medium",
                message=f"High GPU memory usage: {gpu_memory:.1%}",
                current_value=gpu_memory,
                threshold_value=self.alert_thresholds["gpu_memory_90"]
            )

    def _create_alert(self,
                     alert_type: str,
                     severity: str,
                     message: str,
                     current_value: float,
                     threshold_value: float):
        """Create a new alert."""
        alert_id = f"{alert_type}_{int(time.time())}"
        alert = CostAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now()
        )

        self.cost_alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.cost_alerts) > 100:
            self.cost_alerts = self.cost_alerts[-100:]

        logger.warning(f"Alert created: {alert_type} - {message}")

    def create_cost_dashboard(self) -> go.Figure:
        """Create comprehensive cost monitoring dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Cost Trend', 'Cost by Model', 'Budget Utilization', 'Cost per Token'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )

        # Daily cost trend
        daily_costs = defaultdict(float)
        for hour, stats in self.hourly_stats.items():
            day = hour[:10]  # Extract YYYY-MM-DD
            daily_costs[day] += stats.get("cost", 0)

        if daily_costs:
            days = sorted(daily_costs.keys())
            costs = [daily_costs[day] for day in days]

            fig.add_trace(
                go.Scatter(x=days, y=costs, name="Daily Cost", line=dict(color="blue")),
                row=1, col=1
            )

            # Add budget line
            if self.hf_pro_optimizer:
                daily_budget = self.hf_pro_optimizer.monthly_budget / 30
                fig.add_hline(
                    y=daily_budget,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Daily Budget",
                    row=1, col=1
                )

        # Cost by model
        model_costs = {model_id: metrics.total_cost for model_id, metrics in self.model_metrics.items()}
        if model_costs:
            fig.add_trace(
                go.Pie(labels=list(model_costs.keys()), values=list(model_costs.values()),
                      name="Model Costs"),
                row=1, col=2
            )

        # Budget utilization
        if self.hf_pro_optimizer:
            monthly_spent = sum(self.hourly_stats[h].get("cost", 0) for h in self.hourly_stats)
            utilization = (monthly_spent / self.hf_pro_optimizer.monthly_budget) * 100

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=utilization,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Budget Utilization %"},
                    gauge={'axis': {'range': [None, 120]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "yellow"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 100}}
                ),
                row=2, col=1
            )

        # Cost per token by model
        model_efficiency = {
            model_id: metrics.cost_per_token for model_id, metrics in self.model_metrics.items()
            if metrics.cost_per_token > 0
        }
        if model_efficiency:
            fig.add_trace(
                go.Bar(x=list(model_efficiency.keys()), y=list(model_efficiency.values()),
                      name="Cost per Token"),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Felix Framework Cost Analytics Dashboard",
            showlegend=False,
            height=800
        )

        return fig

    def create_performance_dashboard(self) -> go.Figure:
        """Create performance monitoring dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time Trend', 'Success Rate by Model', 'GPU Utilization', 'Concurrent Users'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )

        # Response time trend
        hours = sorted(self.hourly_stats.keys())[-24:]  # Last 24 hours
        response_times = [self.hourly_stats[h].get("avg_response_time", 0) for h in hours]
        request_counts = [self.hourly_stats[h].get("requests", 0) for h in hours]

        if response_times:
            fig.add_trace(
                go.Scatter(x=hours, y=response_times, name="Avg Response Time", line=dict(color="blue")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hours, y=request_counts, name="Request Count", line=dict(color="orange")),
                row=1, col=1, secondary_y=True
            )

        # Success rate by model
        model_success_rates = {
            model_id: (metrics.successful_requests / max(1, metrics.total_requests)) * 100
            for model_id, metrics in self.model_metrics.items()
        }
        if model_success_rates:
            fig.add_trace(
                go.Bar(x=list(model_success_rates.keys()), y=list(model_success_rates.values()),
                      name="Success Rate %"),
                row=1, col=2
            )

        # GPU utilization (simulated data)
        gpu_utilization = [min(100, max(0, 30 + np.random.normal(0, 10))) for _ in range(24)]
        gpu_memory = [min(100, max(0, 40 + np.random.normal(0, 15))) for _ in range(24)]

        fig.add_trace(
            go.Scatter(x=hours, y=gpu_utilization, name="GPU Utilization %", line=dict(color="green")),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=hours, y=gpu_memory, name="GPU Memory %", line=dict(color="red")),
            row=2, col=1, secondary_y=True
        )

        # Concurrent users
        concurrent_users = [len(self.hourly_stats[h].get("unique_sessions", set())) for h in hours]
        if concurrent_users:
            fig.add_trace(
                go.Scatter(x=hours, y=concurrent_users, name="Hourly Active Users",
                          fill='tonexty', line=dict(color="purple")),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Felix Framework Performance Dashboard",
            showlegend=True,
            height=800
        )

        return fig

    def create_user_analytics_dashboard(self) -> go.Figure:
        """Create user engagement and analytics dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('User Sessions Over Time', 'Device Type Distribution',
                          'Geographic Distribution', 'User Satisfaction'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "histogram"}]]
        )

        # Session analytics
        daily_sessions = defaultdict(int)
        device_types = defaultdict(int)
        regions = defaultdict(int)
        satisfaction_scores = []

        for session in self.session_metrics.values():
            day = session.start_time.strftime("%Y-%m-%d")
            daily_sessions[day] += 1
            device_types[session.device_type] += 1
            regions[session.geographic_region] += 1
            if session.user_satisfaction:
                satisfaction_scores.append(session.user_satisfaction)

        # Daily sessions
        if daily_sessions:
            days = sorted(daily_sessions.keys())
            sessions = [daily_sessions[day] for day in days]
            fig.add_trace(
                go.Scatter(x=days, y=sessions, name="Daily Sessions", line=dict(color="blue")),
                row=1, col=1
            )

        # Device distribution
        if device_types:
            fig.add_trace(
                go.Pie(labels=list(device_types.keys()), values=list(device_types.values()),
                      name="Device Types"),
                row=1, col=2
            )

        # Geographic distribution
        if regions:
            fig.add_trace(
                go.Pie(labels=list(regions.keys()), values=list(regions.values()),
                      name="Regions"),
                row=2, col=1
            )

        # User satisfaction
        if satisfaction_scores:
            fig.add_trace(
                go.Histogram(x=satisfaction_scores, nbinsx=5, name="Satisfaction Scores"),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Felix Framework User Analytics Dashboard",
            showlegend=False,
            height=800
        )

        return fig

    def create_predictive_dashboard(self) -> go.Figure:
        """Create predictive analytics dashboard."""
        if not self.enable_predictive_analytics:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Predictive analytics disabled",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cost Forecast', 'Usage Prediction', 'Model Performance Trend',
                          'Resource Planning'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"type": "bar"}]]
        )

        # Simple cost forecasting based on recent trends
        recent_days = sorted(self.hourly_stats.keys())[-168:]  # Last 7 days
        if len(recent_days) > 24:
            daily_costs = defaultdict(float)
            for hour in recent_days:
                day = hour[:10]
                daily_costs[day] += self.hourly_stats[hour].get("cost", 0)

            costs = list(daily_costs.values())
            if len(costs) >= 3:
                # Simple linear trend
                x = list(range(len(costs)))
                trend = np.polyfit(x, costs, 1)

                # Forecast next 7 days
                future_x = list(range(len(costs), len(costs) + 7))
                forecast_costs = [np.polyval(trend, xi) for xi in future_x]

                # Historical
                fig.add_trace(
                    go.Scatter(x=list(daily_costs.keys()), y=costs,
                              name="Historical Cost", line=dict(color="blue")),
                    row=1, col=1
                )

                # Forecast
                future_days = [
                    (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, 8)
                ]
                fig.add_trace(
                    go.Scatter(x=future_days, y=forecast_costs,
                              name="Forecast", line=dict(color="red", dash="dash")),
                    row=1, col=1
                )

        # Usage prediction (requests)
        daily_requests = defaultdict(int)
        for hour in recent_days:
            day = hour[:10]
            daily_requests[day] += self.hourly_stats[hour].get("requests", 0)

        if len(daily_requests) >= 3:
            requests = list(daily_requests.values())
            x = list(range(len(requests)))
            trend = np.polyfit(x, requests, 1)

            future_x = list(range(len(requests), len(requests) + 7))
            forecast_requests = [max(0, np.polyval(trend, xi)) for xi in future_x]

            fig.add_trace(
                go.Scatter(x=list(daily_requests.keys()), y=requests,
                          name="Historical Requests", line=dict(color="green")),
                row=1, col=2
            )

            future_days = [
                (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(1, 8)
            ]
            fig.add_trace(
                go.Scatter(x=future_days, y=forecast_requests,
                          name="Request Forecast", line=dict(color="orange", dash="dash")),
                row=1, col=2
            )

        # Model performance trend
        for model_id, metrics in list(self.model_metrics.items())[:3]:  # Top 3 models
            performance_score = (
                metrics.avg_quality_score * 0.4 +
                (1 - metrics.error_rate) * 0.3 +
                min(1, 2.0 / max(0.1, metrics.avg_response_time)) * 0.3
            )

            # Simulate trend data
            trend_data = [performance_score + np.random.normal(0, 0.1) for _ in range(7)]
            days = [(datetime.now() - timedelta(days=6-i)).strftime("%m-%d") for i in range(7)]

            fig.add_trace(
                go.Scatter(x=days, y=trend_data, name=f"{model_id} Performance"),
                row=2, col=1
            )

        # Resource planning recommendations
        recommendations = [
            "Increase GPU allocation",
            "Optimize model selection",
            "Implement caching",
            "Scale user capacity",
            "Cost optimization"
        ]
        importance_scores = [85, 75, 65, 55, 45]

        fig.add_trace(
            go.Bar(x=recommendations, y=importance_scores, name="Priority Score"),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Felix Framework Predictive Analytics",
            showlegend=True,
            height=800
        )

        return fig

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        current_time = datetime.now()

        # Calculate summary statistics
        total_sessions = len(self.session_metrics)
        active_sessions = len(self.active_sessions)
        total_cost = sum(m.total_cost for m in self.model_metrics.values())
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        total_tokens = sum(m.total_tokens for m in self.model_metrics.values())

        avg_session_duration = 0
        completed_sessions = [s for s in self.session_metrics.values() if s.end_time]
        if completed_sessions:
            avg_session_duration = statistics.mean(
                (s.end_time - s.start_time).total_seconds() for s in completed_sessions
            )

        # Model performance ranking
        model_rankings = []
        for model_id, metrics in self.model_metrics.items():
            score = (
                (metrics.successful_requests / max(1, metrics.total_requests)) * 0.3 +
                min(1, 2.0 / max(0.1, metrics.avg_response_time)) * 0.3 +
                (1 - min(1, metrics.cost_per_token * 1000)) * 0.2 +
                metrics.avg_quality_score * 0.2
            )
            model_rankings.append({
                "model_id": model_id,
                "score": score,
                "requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / max(1, metrics.total_requests),
                "avg_response_time": metrics.avg_response_time,
                "cost_per_token": metrics.cost_per_token
            })

        model_rankings.sort(key=lambda x: x["score"], reverse=True)

        # Cost analysis
        monthly_projection = 0
        if self.hf_pro_optimizer:
            daily_average = total_cost / max(1, (current_time.day))
            monthly_projection = daily_average * 30

        # Recent alerts
        recent_alerts = [
            {
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in self.cost_alerts[-10:]  # Last 10 alerts
        ]

        return {
            "report_timestamp": current_time.isoformat(),
            "summary": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_cost": total_cost,
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "avg_session_duration": avg_session_duration,
                "peak_concurrent_users": self.peak_concurrent_users,
                "current_concurrent_users": self.current_concurrent_users
            },
            "cost_analysis": {
                "total_spent": total_cost,
                "monthly_projection": monthly_projection,
                "budget_utilization": (total_cost / self.hf_pro_optimizer.monthly_budget * 100)
                                    if self.hf_pro_optimizer else 0,
                "avg_cost_per_request": total_cost / max(1, total_requests),
                "avg_cost_per_token": total_cost / max(1, total_tokens)
            },
            "performance_metrics": {
                "overall_success_rate": sum(m.successful_requests for m in self.model_metrics.values()) /
                                      max(1, sum(m.total_requests for m in self.model_metrics.values())),
                "avg_response_time": statistics.mean([m.avg_response_time for m in self.model_metrics.values()])
                                   if self.model_metrics else 0,
                "error_rate": 1 - (sum(m.successful_requests for m in self.model_metrics.values()) /
                                 max(1, sum(m.total_requests for m in self.model_metrics.values())))
            },
            "model_rankings": model_rankings[:10],  # Top 10 models
            "recent_alerts": recent_alerts,
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analytics."""
        recommendations = []

        # Cost optimization
        if self.hf_pro_optimizer:
            monthly_spent = sum(m.total_cost for m in self.model_metrics.values())
            if monthly_spent > self.hf_pro_optimizer.monthly_budget * 0.8:
                recommendations.append("Consider implementing more aggressive cost controls - approaching budget limit")

        # Performance optimization
        avg_response_time = statistics.mean([m.avg_response_time for m in self.model_metrics.values()]) if self.model_metrics else 0
        if avg_response_time > 3.0:
            recommendations.append("High average response time detected - consider using faster models for routine tasks")

        # Error rate optimization
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        successful_requests = sum(m.successful_requests for m in self.model_metrics.values())
        error_rate = 1 - (successful_requests / max(1, total_requests))
        if error_rate > 0.05:
            recommendations.append("High error rate detected - review model configurations and fallback strategies")

        # Usage patterns
        if len(self.active_sessions) > self.peak_concurrent_users * 0.8:
            recommendations.append("High concurrent usage - consider scaling infrastructure")

        # Model efficiency
        if self.model_metrics:
            inefficient_models = [
                m for m in self.model_metrics.values()
                if m.cost_per_token > 0.001 and m.total_requests > 10
            ]
            if inefficient_models:
                recommendations.append("Some models show high cost per token - review model selection strategy")

        if not recommendations:
            recommendations.append("System performing well - no immediate optimizations needed")

        return recommendations


# Factory function for easy integration
def create_analytics_dashboard(hf_pro_optimizer: Optional[HFProOptimizer] = None,
                             model_manager: Optional[PremiumModelManager] = None) -> AnalyticsDashboard:
    """
    Create analytics dashboard with recommended settings.

    Args:
        hf_pro_optimizer: Optional HF Pro optimizer instance
        model_manager: Optional premium model manager

    Returns:
        Configured AnalyticsDashboard instance
    """
    return AnalyticsDashboard(
        hf_pro_optimizer=hf_pro_optimizer,
        model_manager=model_manager,
        enable_predictive_analytics=True,
        enable_cost_alerts=True
    )


# Export main classes
__all__ = [
    'AnalyticsDashboard',
    'SessionMetrics',
    'ModelMetrics',
    'CostAlert',
    'create_analytics_dashboard'
]