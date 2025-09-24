"""
Cost Monitoring and Alerting System for Felix Framework HF Pro Deployment

This module provides comprehensive cost monitoring, budget management, and
automated alerting specifically designed for HuggingFace Pro accounts.

Features:
- Real-time cost tracking with per-request granularity
- Budget management with multi-tier alerts
- Predictive cost modeling and forecasting
- Automated cost optimization recommendations
- Usage-based billing analysis
- Cost anomaly detection
- Integration with HF Pro billing APIs
- Slack/email alert integration
- Cost allocation by agent type and user
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import numpy as np

# Optional integrations
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class CostEntry:
    """Individual cost tracking entry."""
    timestamp: datetime
    request_id: str
    session_id: str
    agent_type: str
    model_id: str
    tokens_input: int
    tokens_output: int
    cost_input: float
    cost_output: float
    total_cost: float
    response_time: float
    gpu_time: float = 0.0
    success: bool = True
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetLimit:
    """Budget limit configuration."""
    name: str
    limit_amount: float
    period: str  # "hourly", "daily", "weekly", "monthly"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95, 1.0])
    enabled: bool = True
    reset_day: Optional[int] = None  # For monthly: 1-31
    reset_hour: Optional[int] = None  # For daily/weekly: 0-23


@dataclass
class CostAlert:
    """Cost monitoring alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold_value: float
    budget_name: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostForecast:
    """Cost forecasting result."""
    period: str
    current_spend: float
    projected_spend: float
    confidence_interval: Tuple[float, float]
    trend: str  # "increasing", "decreasing", "stable"
    daily_average: float
    weekly_average: float
    monthly_projection: float
    generated_at: datetime


class CostMonitor:
    """
    Comprehensive cost monitoring system for Felix Framework.

    Provides real-time cost tracking, budget management, and automated
    alerting for HuggingFace Pro account deployments.
    """

    def __init__(self,
                 budgets: Optional[List[BudgetLimit]] = None,
                 alert_channels: Optional[Dict[AlertChannel, Dict[str, Any]]] = None,
                 enable_forecasting: bool = True,
                 enable_anomaly_detection: bool = True,
                 data_retention_days: int = 90):
        """
        Initialize cost monitoring system.

        Args:
            budgets: List of budget limits to monitor
            alert_channels: Configuration for alert delivery channels
            enable_forecasting: Enable cost forecasting
            enable_anomaly_detection: Enable anomaly detection
            data_retention_days: Days to retain cost data
        """
        # Default budgets
        self.budgets = budgets or [
            BudgetLimit(
                name="daily_budget",
                limit_amount=10.0,  # $10/day
                period="daily",
                alert_thresholds=[0.7, 0.85, 0.95, 1.0]
            ),
            BudgetLimit(
                name="monthly_budget",
                limit_amount=200.0,  # $200/month
                period="monthly",
                alert_thresholds=[0.5, 0.8, 0.9, 1.0],
                reset_day=1
            )
        ]

        # Alert channels configuration
        self.alert_channels = alert_channels or {
            AlertChannel.LOG: {"enabled": True}
        }

        self.enable_forecasting = enable_forecasting
        self.enable_anomaly_detection = enable_anomaly_detection
        self.data_retention_days = data_retention_days

        # Cost tracking storage
        self.cost_entries: deque = deque(maxlen=100000)  # ~90 days at high volume
        self.hourly_aggregates: Dict[str, Dict] = {}
        self.daily_aggregates: Dict[str, Dict] = {}

        # Alert management
        self.active_alerts: Dict[str, CostAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # Forecasting data
        self.historical_trends: deque = deque(maxlen=720)  # 30 days hourly
        self.anomaly_baseline: Dict[str, float] = {}

        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "total_cost": 0.0,
            "avg_cost_per_request": 0.0,
            "cost_by_agent": defaultdict(float),
            "cost_by_model": defaultdict(float),
            "cost_by_user": defaultdict(float)
        }

        logger.info("Cost monitoring system initialized")

    def track_cost(self,
                   request_id: str,
                   session_id: str,
                   agent_type: str,
                   model_id: str,
                   tokens_input: int,
                   tokens_output: int,
                   cost_input: float,
                   cost_output: float,
                   response_time: float,
                   gpu_time: float = 0.0,
                   success: bool = True,
                   user_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Track cost for a single request."""
        total_cost = cost_input + cost_output

        entry = CostEntry(
            timestamp=datetime.now(),
            request_id=request_id,
            session_id=session_id,
            agent_type=agent_type,
            model_id=model_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_input=cost_input,
            cost_output=cost_output,
            total_cost=total_cost,
            response_time=response_time,
            gpu_time=gpu_time,
            success=success,
            user_id=user_id,
            metadata=metadata or {}
        )

        self.cost_entries.append(entry)
        self._update_aggregates(entry)
        self._update_stats(entry)

        # Check for budget alerts
        asyncio.create_task(self._check_budget_alerts())

        # Check for anomalies
        if self.enable_anomaly_detection:
            asyncio.create_task(self._check_cost_anomalies(entry))

        logger.debug(f"Tracked cost: ${total_cost:.4f} for {agent_type} agent using {model_id}")

    def _update_aggregates(self, entry: CostEntry):
        """Update hourly and daily cost aggregates."""
        hour_key = entry.timestamp.strftime("%Y-%m-%d-%H")
        day_key = entry.timestamp.strftime("%Y-%m-%d")

        # Hourly aggregates
        if hour_key not in self.hourly_aggregates:
            self.hourly_aggregates[hour_key] = {
                "total_cost": 0.0,
                "total_requests": 0,
                "total_tokens": 0,
                "by_agent": defaultdict(float),
                "by_model": defaultdict(float),
                "avg_response_time": 0.0,
                "total_response_time": 0.0
            }

        hour_agg = self.hourly_aggregates[hour_key]
        hour_agg["total_cost"] += entry.total_cost
        hour_agg["total_requests"] += 1
        hour_agg["total_tokens"] += entry.tokens_input + entry.tokens_output
        hour_agg["by_agent"][entry.agent_type] += entry.total_cost
        hour_agg["by_model"][entry.model_id] += entry.total_cost
        hour_agg["total_response_time"] += entry.response_time
        hour_agg["avg_response_time"] = hour_agg["total_response_time"] / hour_agg["total_requests"]

        # Daily aggregates
        if day_key not in self.daily_aggregates:
            self.daily_aggregates[day_key] = {
                "total_cost": 0.0,
                "total_requests": 0,
                "total_tokens": 0,
                "by_agent": defaultdict(float),
                "by_model": defaultdict(float),
                "by_user": defaultdict(float),
                "unique_sessions": set()
            }

        day_agg = self.daily_aggregates[day_key]
        day_agg["total_cost"] += entry.total_cost
        day_agg["total_requests"] += 1
        day_agg["total_tokens"] += entry.tokens_input + entry.tokens_output
        day_agg["by_agent"][entry.agent_type] += entry.total_cost
        day_agg["by_model"][entry.model_id] += entry.total_cost
        if entry.user_id:
            day_agg["by_user"][entry.user_id] += entry.total_cost
        day_agg["unique_sessions"].add(entry.session_id)

    def _update_stats(self, entry: CostEntry):
        """Update overall performance statistics."""
        self.processing_stats["total_requests"] += 1
        self.processing_stats["total_cost"] += entry.total_cost
        self.processing_stats["avg_cost_per_request"] = (
            self.processing_stats["total_cost"] / self.processing_stats["total_requests"]
        )
        self.processing_stats["cost_by_agent"][entry.agent_type] += entry.total_cost
        self.processing_stats["cost_by_model"][entry.model_id] += entry.total_cost
        if entry.user_id:
            self.processing_stats["cost_by_user"][entry.user_id] += entry.total_cost

    async def _check_budget_alerts(self):
        """Check all budgets for threshold violations."""
        for budget in self.budgets:
            if not budget.enabled:
                continue

            current_spend = self._get_current_spend(budget)
            utilization = current_spend / budget.limit_amount

            # Check each threshold
            for threshold in budget.alert_thresholds:
                if utilization >= threshold:
                    alert_id = f"{budget.name}_{int(threshold * 100)}"

                    # Don't create duplicate alerts
                    if alert_id in self.active_alerts:
                        continue

                    severity = self._determine_alert_severity(threshold)
                    await self._create_budget_alert(budget, current_spend, threshold, severity)

    def _get_current_spend(self, budget: BudgetLimit) -> float:
        """Get current spending for a budget period."""
        now = datetime.now()

        if budget.period == "hourly":
            hour_key = now.strftime("%Y-%m-%d-%H")
            return self.hourly_aggregates.get(hour_key, {}).get("total_cost", 0.0)

        elif budget.period == "daily":
            day_key = now.strftime("%Y-%m-%d")
            return self.daily_aggregates.get(day_key, {}).get("total_cost", 0.0)

        elif budget.period == "weekly":
            # Get spending for current week
            week_start = now - timedelta(days=now.weekday())
            total = 0.0
            for i in range(7):
                day = week_start + timedelta(days=i)
                day_key = day.strftime("%Y-%m-%d")
                total += self.daily_aggregates.get(day_key, {}).get("total_cost", 0.0)
            return total

        elif budget.period == "monthly":
            # Get spending for current month
            if budget.reset_day:
                # Custom reset day
                if now.day >= budget.reset_day:
                    start_date = now.replace(day=budget.reset_day)
                else:
                    # Previous month
                    prev_month = now.replace(day=1) - timedelta(days=1)
                    start_date = prev_month.replace(day=budget.reset_day)
            else:
                # Calendar month
                start_date = now.replace(day=1)

            total = 0.0
            current_date = start_date
            while current_date <= now:
                day_key = current_date.strftime("%Y-%m-%d")
                total += self.daily_aggregates.get(day_key, {}).get("total_cost", 0.0)
                current_date += timedelta(days=1)

            return total

        return 0.0

    def _determine_alert_severity(self, threshold: float) -> AlertSeverity:
        """Determine alert severity based on threshold."""
        if threshold >= 1.0:
            return AlertSeverity.CRITICAL
        elif threshold >= 0.9:
            return AlertSeverity.HIGH
        elif threshold >= 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    async def _create_budget_alert(self, budget: BudgetLimit, current_spend: float,
                                 threshold: float, severity: AlertSeverity):
        """Create and send budget alert."""
        alert_id = f"{budget.name}_{int(threshold * 100)}"
        utilization = current_spend / budget.limit_amount

        alert = CostAlert(
            alert_id=alert_id,
            severity=severity,
            title=f"Budget Alert: {budget.name} ({threshold * 100:.0f}% threshold)",
            message=f"Budget '{budget.name}' has reached {utilization * 100:.1f}% "
                   f"of the ${budget.limit_amount:.2f} {budget.period} limit. "
                   f"Current spend: ${current_spend:.2f}",
            current_value=current_spend,
            threshold_value=budget.limit_amount * threshold,
            budget_name=budget.name,
            triggered_at=datetime.now()
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send alert through configured channels
        await self._send_alert(alert)

        logger.warning(f"Budget alert triggered: {alert.title}")

    async def _check_cost_anomalies(self, entry: CostEntry):
        """Check for cost anomalies."""
        if not self.enable_anomaly_detection:
            return

        # Simple anomaly detection based on cost per request
        key = f"{entry.agent_type}_{entry.model_id}"

        if key not in self.anomaly_baseline:
            # Initialize baseline
            recent_costs = [
                e.total_cost for e in list(self.cost_entries)[-100:]
                if e.agent_type == entry.agent_type and e.model_id == entry.model_id
            ]
            if len(recent_costs) >= 10:
                self.anomaly_baseline[key] = statistics.mean(recent_costs)
            return

        baseline = self.anomaly_baseline[key]
        if entry.total_cost > baseline * 5:  # 5x normal cost
            await self._create_anomaly_alert(entry, baseline)

    async def _create_anomaly_alert(self, entry: CostEntry, baseline: float):
        """Create cost anomaly alert."""
        alert_id = f"anomaly_{entry.request_id}"

        alert = CostAlert(
            alert_id=alert_id,
            severity=AlertSeverity.HIGH,
            title="Cost Anomaly Detected",
            message=f"Unusual cost detected for {entry.agent_type} agent using {entry.model_id}. "
                   f"Cost: ${entry.total_cost:.4f} (baseline: ${baseline:.4f}, "
                   f"{entry.total_cost/baseline:.1f}x normal)",
            current_value=entry.total_cost,
            threshold_value=baseline * 3,
            budget_name="anomaly_detection",
            triggered_at=datetime.now(),
            metadata={
                "agent_type": entry.agent_type,
                "model_id": entry.model_id,
                "baseline_cost": baseline,
                "multiplier": entry.total_cost / baseline
            }
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        await self._send_alert(alert)

    async def _send_alert(self, alert: CostAlert):
        """Send alert through configured channels."""
        for channel, config in self.alert_channels.items():
            if not config.get("enabled", False):
                continue

            try:
                if channel == AlertChannel.EMAIL and EMAIL_AVAILABLE:
                    await self._send_email_alert(alert, config)
                elif channel == AlertChannel.SLACK and REQUESTS_AVAILABLE:
                    await self._send_slack_alert(alert, config)
                elif channel == AlertChannel.WEBHOOK and REQUESTS_AVAILABLE:
                    await self._send_webhook_alert(alert, config)
                elif channel == AlertChannel.LOG:
                    self._send_log_alert(alert)

            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")

    async def _send_email_alert(self, alert: CostAlert, config: Dict[str, Any]):
        """Send email alert."""
        if not EMAIL_AVAILABLE:
            return

        def send_email():
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']
            msg['Subject'] = f"Felix Framework - {alert.title}"

            body = f"""
            Alert Details:
            - Severity: {alert.severity.value.upper()}
            - Message: {alert.message}
            - Current Value: ${alert.current_value:.2f}
            - Threshold: ${alert.threshold_value:.2f}
            - Time: {alert.triggered_at.isoformat()}

            Budget: {alert.budget_name}
            Alert ID: {alert.alert_id}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls', True):
                server.starttls()
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])

            server.sendmail(config['from_email'], config['to_email'], msg.as_string())
            server.quit()

        # Run in thread to avoid blocking
        import threading
        thread = threading.Thread(target=send_email)
        thread.start()

    async def _send_slack_alert(self, alert: CostAlert, config: Dict[str, Any]):
        """Send Slack alert."""
        if not REQUESTS_AVAILABLE:
            return

        color = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }[alert.severity]

        payload = {
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Current Value", "value": f"${alert.current_value:.2f}", "short": True},
                    {"title": "Threshold", "value": f"${alert.threshold_value:.2f}", "short": True},
                    {"title": "Budget", "value": alert.budget_name, "short": True},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True}
                ],
                "ts": alert.triggered_at.timestamp()
            }]
        }

        requests.post(config['webhook_url'], json=payload)

    async def _send_webhook_alert(self, alert: CostAlert, config: Dict[str, Any]):
        """Send webhook alert."""
        if not REQUESTS_AVAILABLE:
            return

        payload = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "budget_name": alert.budget_name,
            "triggered_at": alert.triggered_at.isoformat(),
            "metadata": alert.metadata
        }

        headers = config.get('headers', {})
        requests.post(config['url'], json=payload, headers=headers)

    def _send_log_alert(self, alert: CostAlert):
        """Send log alert."""
        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[alert.severity]

        logger.log(log_level, f"COST ALERT - {alert.title}: {alert.message}")

    def generate_forecast(self, days_ahead: int = 30) -> CostForecast:
        """Generate cost forecast."""
        if not self.enable_forecasting:
            raise ValueError("Forecasting is disabled")

        # Get recent daily costs
        recent_days = sorted(self.daily_aggregates.keys())[-30:]  # Last 30 days
        if len(recent_days) < 7:
            raise ValueError("Insufficient data for forecasting (need at least 7 days)")

        daily_costs = [self.daily_aggregates[day]["total_cost"] for day in recent_days]

        # Simple linear trend forecasting
        x = np.arange(len(daily_costs))
        coeffs = np.polyfit(x, daily_costs, 1)
        trend_slope = coeffs[0]

        # Determine trend
        if trend_slope > 0.01:
            trend = "increasing"
        elif trend_slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"

        # Calculate averages
        daily_average = statistics.mean(daily_costs)
        weekly_average = daily_average * 7

        # Project future
        last_cost = daily_costs[-1]
        projected_daily = max(0, last_cost + (trend_slope * days_ahead))
        monthly_projection = projected_daily * 30

        # Confidence interval (simple approach)
        std_dev = statistics.stdev(daily_costs) if len(daily_costs) > 1 else 0
        confidence_margin = std_dev * 1.96  # 95% confidence
        confidence_interval = (
            max(0, projected_daily - confidence_margin),
            projected_daily + confidence_margin
        )

        return CostForecast(
            period=f"{days_ahead} days",
            current_spend=sum(daily_costs),
            projected_spend=projected_daily * days_ahead,
            confidence_interval=confidence_interval,
            trend=trend,
            daily_average=daily_average,
            weekly_average=weekly_average,
            monthly_projection=monthly_projection,
            generated_at=datetime.now()
        )

    def get_cost_breakdown(self, period: str = "daily") -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        if period == "daily":
            today = datetime.now().strftime("%Y-%m-%d")
            data = self.daily_aggregates.get(today, {})
        elif period == "hourly":
            hour = datetime.now().strftime("%Y-%m-%d-%H")
            data = self.hourly_aggregates.get(hour, {})
        else:
            # Custom period - aggregate multiple days
            data = {"total_cost": 0.0, "by_agent": defaultdict(float), "by_model": defaultdict(float)}
            for day_data in self.daily_aggregates.values():
                data["total_cost"] += day_data.get("total_cost", 0.0)
                for agent, cost in day_data.get("by_agent", {}).items():
                    data["by_agent"][agent] += cost
                for model, cost in day_data.get("by_model", {}).items():
                    data["by_model"][model] += cost

        return {
            "period": period,
            "total_cost": data.get("total_cost", 0.0),
            "by_agent_type": dict(data.get("by_agent", {})),
            "by_model": dict(data.get("by_model", {})),
            "by_user": dict(data.get("by_user", {})) if period == "daily" else {},
            "total_requests": data.get("total_requests", 0),
            "avg_cost_per_request": (
                data.get("total_cost", 0.0) / max(1, data.get("total_requests", 1))
            )
        }

    def get_budget_status(self) -> List[Dict[str, Any]]:
        """Get status of all budgets."""
        status = []

        for budget in self.budgets:
            current_spend = self._get_current_spend(budget)
            utilization = current_spend / budget.limit_amount
            remaining = budget.limit_amount - current_spend

            status.append({
                "name": budget.name,
                "period": budget.period,
                "limit": budget.limit_amount,
                "current_spend": current_spend,
                "remaining": remaining,
                "utilization": utilization,
                "utilization_percent": utilization * 100,
                "status": (
                    "critical" if utilization >= 1.0 else
                    "high" if utilization >= 0.9 else
                    "medium" if utilization >= 0.7 else
                    "low"
                ),
                "enabled": budget.enabled,
                "alert_thresholds": budget.alert_thresholds
            })

        return status

    def get_active_alerts(self) -> List[CostAlert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

    def cleanup_old_data(self):
        """Clean up old cost data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)

        # Clean cost entries
        original_count = len(self.cost_entries)
        self.cost_entries = deque(
            (entry for entry in self.cost_entries if entry.timestamp > cutoff_date),
            maxlen=self.cost_entries.maxlen
        )

        # Clean aggregates
        cutoff_day = cutoff_date.strftime("%Y-%m-%d")
        cutoff_hour = cutoff_date.strftime("%Y-%m-%d-%H")

        old_days = [day for day in self.daily_aggregates.keys() if day < cutoff_day]
        for day in old_days:
            del self.daily_aggregates[day]

        old_hours = [hour for hour in self.hourly_aggregates.keys() if hour < cutoff_hour]
        for hour in old_hours:
            del self.hourly_aggregates[hour]

        cleaned_count = original_count - len(self.cost_entries)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old cost entries")

    def get_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Analyze agent type costs
        agent_costs = self.processing_stats["cost_by_agent"]
        if agent_costs:
            total_cost = sum(agent_costs.values())
            expensive_agents = [
                agent for agent, cost in agent_costs.items()
                if cost > total_cost * 0.4  # More than 40% of total
            ]
            if expensive_agents:
                recommendations.append(
                    f"Consider optimizing {', '.join(expensive_agents)} agents - "
                    f"they account for significant costs"
                )

        # Analyze model efficiency
        model_costs = self.processing_stats["cost_by_model"]
        if model_costs:
            # Find most expensive models
            sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_models) > 1 and sorted_models[0][1] > sorted_models[1][1] * 2:
                recommendations.append(
                    f"Model '{sorted_models[0][0]}' is significantly more expensive - "
                    f"consider using alternatives for routine tasks"
                )

        # Check average cost per request
        avg_cost = self.processing_stats["avg_cost_per_request"]
        if avg_cost > 0.10:  # $0.10 per request
            recommendations.append(
                f"High average cost per request (${avg_cost:.3f}) - "
                f"consider using more efficient models or implementing caching"
            )

        # Check for active high-severity alerts
        high_severity_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        ]
        if high_severity_alerts:
            recommendations.append(
                f"Active high-severity budget alerts detected - "
                f"consider immediate cost reduction measures"
            )

        if not recommendations:
            recommendations.append("No immediate cost optimization needed - usage appears efficient")

        return recommendations


# Factory function for easy integration
def create_cost_monitor(monthly_budget: float = 200.0,
                       daily_budget: float = 10.0,
                       alert_email: Optional[str] = None,
                       slack_webhook: Optional[str] = None) -> CostMonitor:
    """
    Create cost monitor with recommended settings.

    Args:
        monthly_budget: Monthly budget limit in USD
        daily_budget: Daily budget limit in USD
        alert_email: Email address for alerts
        slack_webhook: Slack webhook URL for alerts

    Returns:
        Configured CostMonitor instance
    """
    budgets = [
        BudgetLimit(
            name="daily_budget",
            limit_amount=daily_budget,
            period="daily",
            alert_thresholds=[0.7, 0.85, 0.95, 1.0]
        ),
        BudgetLimit(
            name="monthly_budget",
            limit_amount=monthly_budget,
            period="monthly",
            alert_thresholds=[0.5, 0.8, 0.9, 1.0],
            reset_day=1
        )
    ]

    alert_channels = {AlertChannel.LOG: {"enabled": True}}

    if alert_email and EMAIL_AVAILABLE:
        alert_channels[AlertChannel.EMAIL] = {
            "enabled": True,
            "to_email": alert_email,
            "from_email": os.getenv("SMTP_FROM_EMAIL", "alerts@yourdomain.com"),
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "use_tls": True
        }

    if slack_webhook and REQUESTS_AVAILABLE:
        alert_channels[AlertChannel.SLACK] = {
            "enabled": True,
            "webhook_url": slack_webhook
        }

    return CostMonitor(
        budgets=budgets,
        alert_channels=alert_channels,
        enable_forecasting=True,
        enable_anomaly_detection=True
    )


# Export main classes
__all__ = [
    'CostMonitor',
    'CostEntry',
    'BudgetLimit',
    'CostAlert',
    'CostForecast',
    'AlertSeverity',
    'AlertChannel',
    'create_cost_monitor'
]