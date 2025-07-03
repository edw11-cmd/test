"""
Metrics and Monitoring for HyperCortex-AI
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import structlog

from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class MetricsManager:
    """Manages system metrics and monitoring"""
    
    def __init__(self):
        self.settings = settings
        
        # Metrics storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # System metrics
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        
        # Agent metrics
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_activity": None
        })
        
        # LLM metrics
        self.llm_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0
        }
        
        # Memory metrics
        self.memory_metrics = {
            "total_memories": 0,
            "memory_searches": 0,
            "average_search_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        # Tool metrics
        self.tool_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "usage_count": 0,
            "success_count": 0,
            "error_count": 0,
            "average_execution_time": 0.0
        })
        
        logger.info("Metrics Manager initialized")
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric"""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge metric"""
        self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Record a value in a histogram"""
        self.histograms[name].append(value)
    
    def record_timer(self, name: str, duration: float):
        """Record a timer value"""
        self.timers[name].append(duration)
        # Keep only last 1000 measurements
        if len(self.timers[name]) > 1000:
            self.timers[name] = self.timers[name][-1000:]
    
    def record_request(self, success: bool = True):
        """Record an API request"""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def record_agent_task(
        self,
        agent_name: str,
        success: bool,
        execution_time: float
    ):
        """Record agent task execution"""
        
        metrics = self.agent_metrics[agent_name]
        
        if success:
            metrics["tasks_completed"] += 1
        else:
            metrics["tasks_failed"] += 1
        
        metrics["total_execution_time"] += execution_time
        total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
        
        if total_tasks > 0:
            metrics["average_execution_time"] = metrics["total_execution_time"] / total_tasks
        
        metrics["last_activity"] = datetime.utcnow().isoformat()
    
    def record_llm_request(
        self,
        tokens_used: int,
        response_time: float,
        cost: float = 0.0,
        success: bool = True
    ):
        """Record LLM request metrics"""
        
        self.llm_metrics["total_requests"] += 1
        self.llm_metrics["total_tokens"] += tokens_used
        self.llm_metrics["total_cost"] += cost
        
        # Update average response time
        current_avg = self.llm_metrics["average_response_time"]
        total_requests = self.llm_metrics["total_requests"]
        
        self.llm_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Update error rate
        if not success:
            self.increment_counter("llm_errors")
        
        error_count = self.counters.get("llm_errors", 0)
        self.llm_metrics["error_rate"] = error_count / total_requests
    
    def record_memory_operation(
        self,
        operation_type: str,
        execution_time: float,
        success: bool = True
    ):
        """Record memory operation metrics"""
        
        if operation_type == "search":
            self.memory_metrics["memory_searches"] += 1
            
            # Update average search time
            current_avg = self.memory_metrics["average_search_time"]
            search_count = self.memory_metrics["memory_searches"]
            
            self.memory_metrics["average_search_time"] = (
                (current_avg * (search_count - 1) + execution_time) / search_count
            )
        
        elif operation_type == "store":
            self.memory_metrics["total_memories"] += 1
    
    def record_tool_usage(
        self,
        tool_name: str,
        execution_time: float,
        success: bool = True
    ):
        """Record tool usage metrics"""
        
        metrics = self.tool_metrics[tool_name]
        metrics["usage_count"] += 1
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
        
        # Update average execution time
        current_avg = metrics["average_execution_time"]
        usage_count = metrics["usage_count"]
        
        metrics["average_execution_time"] = (
            (current_avg * (usage_count - 1) + execution_time) / usage_count
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "system": {
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat(),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "requests_per_second": self.request_count / max(uptime, 1)
            },
            "agents": dict(self.agent_metrics),
            "llm": self.llm_metrics,
            "memory": self.memory_metrics,
            "tools": dict(self.tool_metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0
                }
                for name, values in self.histograms.items()
            },
            "timers": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p95": self._percentile(values, 95) if values else 0,
                    "p99": self._percentile(values, 99) if values else 0
                }
                for name, values in self.timers.items()
            }
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        
        metrics = await self.get_metrics()
        
        # Determine health based on metrics
        health_status = "healthy"
        issues = []
        
        # Check error rates
        if metrics["system"]["error_rate"] > 0.1:  # 10% error rate
            health_status = "degraded"
            issues.append("High error rate")
        
        if metrics["llm"]["error_rate"] > 0.05:  # 5% LLM error rate
            health_status = "degraded"
            issues.append("High LLM error rate")
        
        # Check response times
        if metrics["llm"]["average_response_time"] > 10.0:  # 10 seconds
            health_status = "degraded"
            issues.append("Slow LLM response times")
        
        # Check agent activity
        active_agents = 0
        for agent_name, agent_metrics in metrics["agents"].items():
            if agent_metrics["last_activity"]:
                last_activity = datetime.fromisoformat(agent_metrics["last_activity"])
                if (datetime.utcnow() - last_activity).total_seconds() < 3600:  # 1 hour
                    active_agents += 1
        
        if active_agents == 0 and metrics["system"]["uptime_seconds"] > 300:  # 5 minutes
            health_status = "degraded"
            issues.append("No recent agent activity")
        
        return {
            "status": health_status,
            "issues": issues,
            "uptime": metrics["system"]["uptime_seconds"],
            "active_agents": active_agents,
            "total_requests": metrics["system"]["request_count"],
            "error_rate": metrics["system"]["error_rate"]
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
        
        self.request_count = 0
        self.error_count = 0
        self.agent_metrics.clear()
        
        self.llm_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0
        }
        
        self.memory_metrics = {
            "total_memories": 0,
            "memory_searches": 0,
            "average_search_time": 0.0,
            "cache_hit_rate": 0.0
        }
        
        self.tool_metrics.clear()
        
        logger.info("Metrics reset")


class MetricsCollector:
    """Context manager for collecting metrics"""
    
    def __init__(self, metrics_manager: MetricsManager, metric_name: str):
        self.metrics_manager = metrics_manager
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_manager.record_timer(self.metric_name, duration)
            
            if exc_type is not None:
                self.metrics_manager.increment_counter(f"{self.metric_name}_errors")


# Global metrics manager instance
metrics_manager = MetricsManager()


def get_metrics_manager() -> MetricsManager:
    """Get the global metrics manager instance"""
    return metrics_manager


def collect_metrics(metric_name: str) -> MetricsCollector:
    """Create a metrics collector context manager"""
    return MetricsCollector(metrics_manager, metric_name)