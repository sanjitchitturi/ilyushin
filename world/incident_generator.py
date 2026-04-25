from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random

from world.metrics import evaluate_metric, MetricStatus
from world.infrastructure import Infrastructure, ServiceStatus, SERVICE_NAMES


class IncidentType(Enum):
    CPU_SPIKE        = "cpu_spike"
    MEMORY_LEAK      = "memory_leak"
    SERVICE_CRASH    = "service_crash"
    HIGH_LATENCY     = "high_latency"
    CASCADE_FAILURE  = "cascade_failure"
    RED_HERRING      = "red_herring"
    OVERLOAD         = "overload"


class Severity(Enum):
    P3 = "p3"
    P2 = "p2"
    P1 = "p1"


@dataclass
class Incident:
    incident_type:   IncidentType
    severity:        Severity
    target_services: list
    decay_rate:      float
    self_healing:    bool
    description:     str
    active:          bool = True
    ticks_active:    int  = 0


class IncidentGenerator:
    def __init__(self, infrastructure: Infrastructure):
        self.infrastructure     = infrastructure
        self.active_incidents:  list[Incident] = []

    def inject(
        self,
        incident_type:   IncidentType,
        target_services: list,
        severity:        Severity = Severity.P2,
    ) -> Incident:
        incident = None

        if incident_type == IncidentType.CPU_SPIKE:
            incident = self._inject_cpu_spike(target_services, severity)
        elif incident_type == IncidentType.MEMORY_LEAK:
            incident = self._inject_memory_leak(target_services, severity)
        elif incident_type == IncidentType.SERVICE_CRASH:
            incident = self._inject_service_crash(target_services, severity)
        elif incident_type == IncidentType.HIGH_LATENCY:
            incident = self._inject_high_latency(target_services, severity)
        elif incident_type == IncidentType.CASCADE_FAILURE:
            incident = self._inject_cascade(target_services, severity)
        elif incident_type == IncidentType.RED_HERRING:
            incident = self._inject_red_herring(target_services, severity)
        elif incident_type == IncidentType.OVERLOAD:
            incident = self._inject_overload(target_services, severity)

        if incident:
            self.active_incidents.append(incident)

        return incident

    def tick(self):
        for incident in self.active_incidents:
            if not incident.active:
                continue

            incident.ticks_active += 1

            if incident.self_healing:
                if incident.incident_type == IncidentType.MEMORY_LEAK:
                    # Memory leak keeps getting WORSE — must restart to fix
                    for service_name in incident.target_services:
                        service = self.infrastructure.get_service(service_name)
                        memory  = service.metrics.get("memory")
                        if memory:
                            # Grows ~3% per tick — will exceed critical within episode
                            new_memory = min(memory.value + 3.0, 99.0)
                            service.metrics["memory"] = evaluate_metric("memory", new_memory)
                            if new_memory >= 92.0:
                                service.status = ServiceStatus.DOWN

                elif incident.incident_type == IncidentType.CPU_SPIKE:
                    # PATCH: CPU spike barely decays — won't self-heal within episode
                    for service_name in incident.target_services:
                        service = self.infrastructure.get_service(service_name)
                        cpu     = service.metrics.get("cpu")
                        if cpu and cpu.value > 30.0:
                            # Was 2.0 * decay_rate (0.3) = 0.6/tick
                            # Now 0.3/tick — takes 100+ ticks to clear
                            new_cpu = cpu.value - (0.3 * incident.decay_rate)
                            service.metrics["cpu"] = evaluate_metric("cpu", new_cpu)
                            # Keep status degraded as long as CPU is critical
                            if service.metrics["cpu"].status == MetricStatus.CRITICAL:
                                service.status = ServiceStatus.DEGRADED

                elif incident.incident_type == IncidentType.OVERLOAD:
                    for service_name in incident.target_services:
                        service = self.infrastructure.get_service(service_name)
                        if service.scale_level <= 1:
                            cpu = service.metrics.get("cpu")
                            rps = service.metrics.get("requests_per_second")
                            if cpu:
                                new_cpu = min(cpu.value + 1.5, 99.0)
                                service.metrics["cpu"] = evaluate_metric("cpu", new_cpu)
                            if rps:
                                new_rps = min(rps.value + 10.0, 1050.0)
                                service.metrics["requests_per_second"] = evaluate_metric(
                                    "requests_per_second", new_rps
                                )
                            if service.metrics["cpu"].status == MetricStatus.CRITICAL:
                                service.status = ServiceStatus.DEGRADED
                                service.overloaded = True   # flag persists

                elif incident.incident_type == IncidentType.RED_HERRING:
                    # PATCH: red herring keeps producing warning metrics but status stays degraded
                    # This forces the agent to ignore it and keep fixing real issues
                    for service_name in incident.target_services:
                        service = self.infrastructure.get_service(service_name)
                        cpu = service.metrics.get("cpu")
                        if cpu:
                            # Keep CPU elevated but not critical — looks suspicious
                            jitter = random.uniform(-1.0, 1.0)
                            new_cpu = max(78.0, min(cpu.value + jitter, 85.0))
                            service.metrics["cpu"] = evaluate_metric("cpu", new_cpu)

            # PATCH: auto-expire increased from 20 → 100 ticks
            # Episode max is 20 steps, so self-healing won't save the agent
            if incident.ticks_active > 100 and incident.self_healing:
                incident.active = False

        self.active_incidents = [i for i in self.active_incidents if i.active]

    def get_active_incidents(self) -> list:
        return [
            {
                "type":        i.incident_type.value,
                "severity":    i.severity.value,
                "targets":     i.target_services,
                "description": i.description,
                "ticks_active": i.ticks_active,
            }
            for i in self.active_incidents
        ]

    def clear(self):
        self.active_incidents = []

    def _inject_cpu_spike(self, targets: list, severity: Severity) -> Incident:
        cpu_values = {Severity.P3: 78.0, Severity.P2: 88.0, Severity.P1: 97.0}
        for service_name in targets:
            service     = self.infrastructure.get_service(service_name)
            spike_value = cpu_values[severity] + random.uniform(0, 3.0)
            service.metrics["cpu"] = evaluate_metric("cpu", spike_value)
            # PATCH: P2 and P1 now mark degraded (was only P1)
            # Random agent can't just wait it out anymore
            if severity in (Severity.P1, Severity.P2):
                service.status = ServiceStatus.DEGRADED

        return Incident(
            incident_type=IncidentType.CPU_SPIKE,
            severity=severity,
            target_services=targets,
            decay_rate=0.05,      # PATCH: was 0.3 — way too fast
            self_healing=True,
            description=f"CPU spike detected on {targets}",
        )

    def _inject_memory_leak(self, targets: list, severity: Severity) -> Incident:
        memory_values = {Severity.P3: 72.0, Severity.P2: 82.0, Severity.P1: 88.0}
        for service_name in targets:
            service     = self.infrastructure.get_service(service_name)
            leak_value  = memory_values[severity] + random.uniform(0, 2.0)
            service.metrics["memory"] = evaluate_metric("memory", leak_value)
            # PATCH: mark degraded immediately
            if severity in (Severity.P1, Severity.P2):
                service.status = ServiceStatus.DEGRADED

        return Incident(
            incident_type=IncidentType.MEMORY_LEAK,
            severity=severity,
            target_services=targets,
            decay_rate=0.05,
            self_healing=True,    # gets worse, not better (see tick())
            description=f"Memory leak detected on {targets} — memory grows every tick",
        )

    def _inject_service_crash(self, targets: list, severity: Severity) -> Incident:
        for service_name in targets:
            service = self.infrastructure.get_service(service_name)
            service.status = ServiceStatus.DOWN
            service.metrics["cpu"]                  = evaluate_metric("cpu", 0.0)
            service.metrics["memory"]               = evaluate_metric("memory", 0.0)
            service.metrics["latency"]              = evaluate_metric("latency", 9999.0)
            service.metrics["error_rate"]           = evaluate_metric("error_rate", 100.0)
            service.metrics["requests_per_second"]  = evaluate_metric("requests_per_second", 0.0)

        return Incident(
            incident_type=IncidentType.SERVICE_CRASH,
            severity=Severity.P1,
            target_services=targets,
            decay_rate=0.0,
            self_healing=False,  # MUST restart to fix
            description=f"Service crash on {targets} — must restart",
        )

    def _inject_high_latency(self, targets: list, severity: Severity) -> Incident:
        latency_values = {Severity.P3: 550.0, Severity.P2: 800.0, Severity.P1: 1200.0}
        for service_name in targets:
            service       = self.infrastructure.get_service(service_name)
            latency_value = latency_values[severity] + random.uniform(0, 100.0)
            service.metrics["latency"] = evaluate_metric("latency", latency_value)
            service.status = ServiceStatus.DEGRADED

        return Incident(
            incident_type=IncidentType.HIGH_LATENCY,
            severity=severity,
            target_services=targets,
            decay_rate=0.1,
            self_healing=False,   # PATCH: was False, keep False. Must act.
            description=f"High latency on {targets}",
        )

    def _inject_cascade(self, targets: list, severity: Severity) -> Incident:
        root_service = targets[0]
        self._inject_service_crash([root_service], Severity.P1)

        for service_name in targets[1:]:
            service = self.infrastructure.get_service(service_name)
            service.status = ServiceStatus.DEGRADED
            service.metrics["latency"]    = evaluate_metric("latency",    900.0 + random.uniform(0, 200))
            service.metrics["error_rate"] = evaluate_metric("error_rate", 12.0  + random.uniform(0, 5))

        return Incident(
            incident_type=IncidentType.CASCADE_FAILURE,
            severity=Severity.P1,
            target_services=targets,
            decay_rate=0.0,
            self_healing=False,
            description=f"Cascading failure starting from {root_service}",
        )

    def _inject_red_herring(self, targets: list, severity: Severity) -> Incident:
        # PATCH: red herring now marks service degraded too — forces agent
        # to waste an action checking it OR correctly ignore it
        for service_name in targets:
            service = self.infrastructure.get_service(service_name)
            service.metrics["cpu"]     = evaluate_metric("cpu",     82.0 + random.uniform(0, 5))
            service.metrics["latency"] = evaluate_metric("latency", 600.0 + random.uniform(0, 100))
            # Red herring is "warning" level — not actually broken
            # but metrics look suspicious. Status stays HEALTHY intentionally
            # so the agent who correctly ignores it wins.

        return Incident(
            incident_type=IncidentType.RED_HERRING,
            severity=Severity.P2,
            target_services=targets,
            decay_rate=0.2,
            self_healing=True,
            description=f"Red herring symptoms on {targets} — not the root cause",
        )

    def _inject_overload(self, targets: list, severity: Severity) -> Incident:
        cpu_values = {Severity.P3: 82.0, Severity.P2: 91.0, Severity.P1: 96.0}
        rps_values = {Severity.P3: 820.0, Severity.P2: 900.0, Severity.P1: 980.0}

        for service_name in targets:
            service   = self.infrastructure.get_service(service_name)
            cpu_value = cpu_values[severity] + random.uniform(0, 3.0)
            rps_value = rps_values[severity] + random.uniform(0, 20.0)
            service.metrics["cpu"]               = evaluate_metric("cpu", cpu_value)
            service.metrics["requests_per_second"] = evaluate_metric("requests_per_second", rps_value)
            service.metrics["latency"]           = evaluate_metric("latency", 600.0 + random.uniform(0, 200))
            service.status = ServiceStatus.DEGRADED
            service.overloaded = True   # PATCH: explicitly flag for agent

        return Incident(
            incident_type=IncidentType.OVERLOAD,
            severity=severity,
            target_services=targets,
            decay_rate=0.0,
            self_healing=True,
            description=f"Traffic overload on {targets} — scale_up required, restart will not fix this",
        )