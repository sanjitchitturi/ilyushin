from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random

from world.metrics import (
    MetricSnapshot,
    MetricStatus,
    evaluate_metric,
    generate_normal_metrics,
)


class ServiceStatus(Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DOWN     = "down"


@dataclass
class Service:
    name:          str
    status:        ServiceStatus
    metrics:       dict
    dependencies:  list
    restart_count: int           = 0
    scale_level:   int           = 1
    last_action:   Optional[str] = None
    overloaded:    bool          = False


DEPENDENCY_MAP = {
    "web_server":  ["api_gateway", "cache"],
    "api_gateway": ["database", "queue"],
    "database":    [],
    "cache":       [],
    "queue":       [],
}

SERVICE_NAMES = ["web_server", "database", "cache", "queue", "api_gateway"]


class Infrastructure:
    def __init__(self):
        self.services:   dict[str, Service] = {}
        self.tick_count: int = 0
        self.reset()

    def reset(self):
        self.tick_count = 0
        self.services   = {}
        for name in SERVICE_NAMES:
            self.services[name] = Service(
                name=name,
                status=ServiceStatus.HEALTHY,
                metrics=generate_normal_metrics(name),
                dependencies=DEPENDENCY_MAP[name],
                restart_count=0,
                scale_level=1,
                last_action=None,
                overloaded=False,
            )

    def get_service(self, name: str) -> Service:
        if name not in self.services:
            raise ValueError(f"Unknown service: {name}. Available: {SERVICE_NAMES}")
        return self.services[name]

    def get_all_metrics(self) -> dict:
        snapshot = {}
        for name, service in self.services.items():
            snapshot[name] = {
                "status":        service.status.value,
                "restart_count": service.restart_count,
                "scale_level":   service.scale_level,
                "overloaded":    service.overloaded,
                "metrics": {
                    metric_name: {
                        "value":  m.value,
                        "status": m.status.value,
                        "unit":   m.unit,
                    }
                    for metric_name, m in service.metrics.items()
                }
            }
        return snapshot

    def get_health_summary(self) -> dict:
        total   = len(self.services)
        healthy = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
        degraded = sum(1 for s in self.services.values() if s.status == ServiceStatus.DEGRADED)
        down    = sum(1 for s in self.services.values() if s.status == ServiceStatus.DOWN)

        critical_metrics = []
        for name, service in self.services.items():
            for metric_name, m in service.metrics.items():
                if m.status == MetricStatus.CRITICAL:
                    critical_metrics.append(f"{name}.{metric_name}={m.value}{m.unit}")

        return {
            "total_services":  total,
            "healthy":         healthy,
            "degraded":        degraded,
            "down":            down,
            "system_healthy":  down == 0 and degraded == 0,
            "critical_metrics": critical_metrics,
        }

    def apply_action(self, action_type: str, target_service: str) -> dict:
        if target_service not in self.services:
            return {"success": False, "message": f"Unknown service: {target_service}"}

        service = self.services[target_service]

        if action_type == "restart_service":
            if service.overloaded:
                service.restart_count += 1
                service.metrics = generate_normal_metrics(target_service)
                service.status  = ServiceStatus.DEGRADED
                service.last_action = "restart"
                return {
                    "success": True,
                    "message": (
                        f"{target_service} restarted but overload persists — "
                        f"traffic is still too high. Use scale_up to fix this."
                    )
                }
            service.restart_count += 1
            service.status        = ServiceStatus.HEALTHY
            service.metrics       = generate_normal_metrics(target_service)
            service.last_action   = "restart"
            service.overloaded    = False
            return {"success": True, "message": f"{target_service} restarted successfully"}

        elif action_type == "scale_up":
            service.scale_level = min(service.scale_level + 1, 5)
            new_cpu = max(service.metrics["cpu"].value * 0.4, 10.0)
            new_rps = service.metrics["requests_per_second"].value * 0.4
            service.metrics["cpu"] = evaluate_metric("cpu", new_cpu)
            service.metrics["requests_per_second"] = evaluate_metric("requests_per_second", new_rps)
            service.metrics["latency"] = evaluate_metric("latency", random.uniform(10, 150))
            service.status      = ServiceStatus.HEALTHY
            service.overloaded  = False
            service.last_action = "scale_up"
            return {
                "success": True,
                "message": f"{target_service} scaled up to level {service.scale_level} — overload resolved"
            }

        elif action_type == "rollback":
            service.status      = ServiceStatus.HEALTHY
            service.metrics     = generate_normal_metrics(target_service)
            service.scale_level = max(1, service.scale_level - 1)
            service.overloaded  = False
            service.last_action = "rollback"
            return {"success": True, "message": f"{target_service} rolled back successfully"}

        elif action_type == "resolve":
            if service.status == ServiceStatus.HEALTHY:
                return {"success": True, "message": f"{target_service} confirmed resolved"}
            else:
                return {
                    "success": False,
                    "message": f"{target_service} is still {service.status.value}, cannot resolve yet"
                }

        else:
            return {"success": False, "message": f"Unknown action: {action_type}"}

    def tick(self):
        self.tick_count += 1
        self._propagate_cascades()
        self._add_noise()

    def _propagate_cascades(self):
        for name, service in self.services.items():
            if service.status == ServiceStatus.DOWN:
                for other_name, other_service in self.services.items():
                    if name in other_service.dependencies:
                        if other_service.status == ServiceStatus.HEALTHY:
                            other_service.status = ServiceStatus.DEGRADED
                            latency    = other_service.metrics.get("latency")
                            error_rate = other_service.metrics.get("error_rate")
                            if latency:
                                other_service.metrics["latency"] = evaluate_metric(
                                    "latency", latency.value * 2.5
                                )
                            if error_rate:
                                other_service.metrics["error_rate"] = evaluate_metric(
                                    "error_rate", error_rate.value + 8.0
                                )

            elif service.status == ServiceStatus.DEGRADED:
                for other_name, other_service in self.services.items():
                    if name in other_service.dependencies:
                        if other_service.status == ServiceStatus.HEALTHY:
                            latency = other_service.metrics.get("latency")
                            if latency:
                                new_latency = latency.value * 1.4
                                other_service.metrics["latency"] = evaluate_metric(
                                    "latency", new_latency
                                )
                            if other_service.metrics["latency"].status == MetricStatus.CRITICAL:
                                other_service.status = ServiceStatus.DEGRADED

    def _add_noise(self):
        for name, service in self.services.items():
            if service.status == ServiceStatus.HEALTHY and not service.overloaded:
                for metric_name, metric in service.metrics.items():
                    noise     = random.uniform(-2.0, 2.0)
                    new_value = max(0.0, metric.value + noise)
                    service.metrics[metric_name] = evaluate_metric(metric_name, new_value)