from dataclasses import dataclass
from enum import Enum
import random


class MetricStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricConfig:
    name: str
    normal_min: float
    normal_max: float
    warning_threshold: float
    critical_threshold: float
    unit: str


@dataclass
class MetricSnapshot:
    name: str
    value: float
    status: MetricStatus
    unit: str


METRIC_CONFIGS = {
    "cpu": MetricConfig(
        name="cpu",
        normal_min=10.0,
        normal_max=60.0,
        warning_threshold=75.0,
        critical_threshold=90.0,
        unit="percent"
    ),
    "memory": MetricConfig(
        name="memory",
        normal_min=20.0,
        normal_max=65.0,
        warning_threshold=80.0,
        critical_threshold=92.0,
        unit="percent"
    ),
    "latency": MetricConfig(
        name="latency",
        normal_min=10.0,
        normal_max=200.0,
        warning_threshold=500.0,
        critical_threshold=1000.0,
        unit="ms"
    ),
    "error_rate": MetricConfig(
        name="error_rate",
        normal_min=0.0,
        normal_max=1.0,
        warning_threshold=5.0,
        critical_threshold=15.0,
        unit="percent"
    ),
    "requests_per_second": MetricConfig(
        name="requests_per_second",
        normal_min=50.0,
        normal_max=500.0,
        warning_threshold=800.0,
        critical_threshold=1000.0,
        unit="rps"
    ),
}


def evaluate_metric(name: str, value: float) -> MetricSnapshot:
    config = METRIC_CONFIGS[name]

    if name == "requests_per_second":
        if value >= config.critical_threshold:
            status = MetricStatus.CRITICAL
        elif value >= config.warning_threshold:
            status = MetricStatus.WARNING
        else:
            status = MetricStatus.HEALTHY
    else:
        if value >= config.critical_threshold:
            status = MetricStatus.CRITICAL
        elif value >= config.warning_threshold:
            status = MetricStatus.WARNING
        else:
            status = MetricStatus.HEALTHY

    return MetricSnapshot(name=name, value=round(value, 2), status=status, unit=config.unit)


def generate_normal_metrics(service_name: str) -> dict:
    noise_profiles = {
        "web_server":   {"cpu": (20, 45), "memory": (30, 55), "latency": (50, 150), "error_rate": (0, 0.5), "requests_per_second": (100, 400)},
        "database":     {"cpu": (15, 40), "memory": (40, 65), "latency": (10, 80),  "error_rate": (0, 0.2), "requests_per_second": (80, 300)},
        "cache":        {"cpu": (5, 20),  "memory": (20, 45), "latency": (1, 20),   "error_rate": (0, 0.1), "requests_per_second": (200, 500)},
        "queue":        {"cpu": (10, 30), "memory": (25, 50), "latency": (20, 100), "error_rate": (0, 0.3), "requests_per_second": (50, 200)},
        "api_gateway":  {"cpu": (15, 35), "memory": (25, 50), "latency": (30, 120), "error_rate": (0, 0.4), "requests_per_second": (150, 450)},
    }

    profile = noise_profiles.get(service_name, {
        "cpu": (10, 50), "memory": (20, 60), "latency": (20, 200),
        "error_rate": (0, 1.0), "requests_per_second": (50, 400)
    })

    metrics = {}
    for metric_name, (low, high) in profile.items():
        value = random.uniform(low, high)
        metrics[metric_name] = evaluate_metric(metric_name, value)

    return metrics