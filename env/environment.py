import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server import State

from env.state import EnvState
from env.models import IlyushinAction, IlyushinObservation, Reward
from env.reward import compute_reward
from world.infrastructure import Infrastructure, ServiceStatus
from world.incident_generator import IncidentGenerator
from tasks.registry import get_task
from agents.breaker import BreakerAgent

MAX_STEPS = 20

VALID_ACTIONS = [
    "read_logs",
    "check_metrics",
    "restart_service",
    "scale_up",
    "rollback",
    "page_oncall",
    "resolve",
]


class IlyushinEnv(Environment):
    def __init__(self):
        super().__init__()
        self.infrastructure = Infrastructure()
        self.incident_generator = IncidentGenerator(self.infrastructure)
        self.current_state = None
        self.task = None
        self._openenv_state = State(episode_id=str(uuid.uuid4()), step_count=0)

        # Breaker agent — adaptive adversary
        self.breaker_agent = BreakerAgent(self.infrastructure, self.incident_generator)

        # Track performance for Breaker feedback
        self.episode_incidents = []
        self.episode_actions = []

    def reset(self, task_id: str = "easy", seed: int = None, **kwargs) -> IlyushinObservation:
        """Reset environment with Breaker generating incidents."""
        self.infrastructure.reset()
        self.incident_generator.clear()
        self.episode_incidents = []
        self.episode_actions = []

        self.task = get_task(task_id)

        # Let Breaker generate initial incidents based on learning
        breaker_plan = self.breaker_agent.break_system()

        self.current_state = EnvState(
            task_id=task_id,
            step_count=0,
            done=False,
            infrastructure_snapshot=self.infrastructure.get_all_metrics(),
            active_incidents=self.incident_generator.get_active_incidents(),
            actions_taken=[],
            services_resolved=[],
            total_services=len(self.infrastructure.services),
            healthy_services=sum(
                1 for s in self.infrastructure.services.values()
                if s.status == ServiceStatus.HEALTHY
            ),
            oncall_paged=False,
            last_action="none",
            last_action_result="Episode started",
            last_action_success=True,
        )

        self.episode_incidents = self.current_state.active_incidents.copy()

        self._openenv_state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0
        )

        return self._to_observation(self.current_state)

    def step(self, action: IlyushinAction, **kwargs) -> tuple:
        """
        Execute one step in the environment.

        Returns:
            (IlyushinObservation, reward: float, done: bool)
        """
        if self.current_state is None:
            raise RuntimeError("Call reset() before step()")
        if self.current_state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        if isinstance(action, dict):
            action = IlyushinAction(**action)

        if action.type not in VALID_ACTIONS:
            raise ValueError(
                f"Unknown action type: '{action.type}'. "
                f"Valid actions: {VALID_ACTIONS}"
            )

        old_state      = self.current_state.copy()
        action_success = True
        action_result  = ""

        # Track action for Breaker feedback
        self.episode_actions.append(action.type)

        if action.type == "read_logs":
            action_result = self._read_logs()

        elif action.type == "check_metrics":
            action_result = self._check_metrics(action.target_service)

        elif action.type in ("restart_service", "scale_up", "rollback"):
            if not action.target_service:
                raise ValueError(f"Action '{action.type}' requires 'target_service' field.")
            result = self.infrastructure.apply_action(action.type, action.target_service)
            action_success = result["success"]
            action_result  = result["message"]
            if action_success and action.type == "restart_service":
                self._clear_incidents_for_service(action.target_service)

        elif action.type == "page_oncall":
            if not self.current_state.oncall_paged:
                self.current_state.oncall_paged = True
            action_result = "On-call engineer paged."

        elif action.type == "resolve":
            if not action.target_service:
                raise ValueError("Action 'resolve' requires 'target_service' field.")
            result = self.infrastructure.apply_action("resolve", action.target_service)
            action_success = result["success"]
            action_result  = result["message"]
            if action_success and action.target_service not in self.current_state.services_resolved:
                self.current_state.services_resolved.append(action.target_service)

        self.incident_generator.tick()
        self.infrastructure.tick()

        snapshot         = self.infrastructure.get_all_metrics()
        health           = self.infrastructure.get_health_summary()
        active_incidents = self.incident_generator.get_active_incidents()

        self.current_state.infrastructure_snapshot = snapshot
        self.current_state.active_incidents        = active_incidents
        self.current_state.healthy_services        = health["healthy"]
        self.current_state.last_action             = action.type
        self.current_state.last_action_result      = action_result
        self.current_state.last_action_success     = action_success
        self.current_state.actions_taken.append(action.type)
        self.current_state.step_count += 1

        self._openenv_state.step_count = self.current_state.step_count

        if health["system_healthy"]:
            self.current_state.done = True
        elif self.current_state.step_count >= MAX_STEPS:
            self.current_state.done = True

        reward_value, breakdown = compute_reward(
            old_state=old_state,
            new_state=self.current_state,
            action_type=action.type,
            action_success=action_success,
        )

        obs = self._to_observation(self.current_state)

        # Return proper OpenEnv-compliant tuple: (observation, reward, done)
        return obs, reward_value, self.current_state.done

    def state(self) -> IlyushinObservation:
        if self.current_state is None:
            raise RuntimeError("Call reset() first.")
        return self._to_observation(self.current_state)

    def get_breaker_status(self) -> dict:
        """Get Breaker agent status and learning."""
        return self.breaker_agent.get_status()

    def _clear_incidents_for_service(self, service_name: str):
        self.incident_generator.active_incidents = [
            i for i in self.incident_generator.active_incidents
            if service_name not in i.target_services
        ]

    def _to_observation(self, state: EnvState) -> IlyushinObservation:
        return IlyushinObservation(
            task_id=state.task_id,
            step_count=state.step_count,
            done=state.done,
            infrastructure=state.infrastructure_snapshot,
            active_incidents=state.active_incidents,
            healthy_services=state.healthy_services,
            total_services=state.total_services,
            last_action=state.last_action,
            last_action_result=state.last_action_result,
            last_action_success=state.last_action_success,
            oncall_paged=state.oncall_paged,
        )

    def _read_logs(self) -> str:
        issues = []

        for service_name, service in self.infrastructure.services.items():
            if service.status != ServiceStatus.HEALTHY:
                issues.append(f"[ERROR] {service_name} status={service.status.value}")

            for metric_name, metric in service.metrics.items():
                try:
                    if isinstance(metric, dict):
                        status = metric.get("status")
                        value  = metric.get("value")
                        unit   = metric.get("unit", "")
                    else:
                        status = getattr(metric, "status", None)
                        value  = getattr(metric, "value", None)
                        unit   = getattr(metric, "unit", "")

                    if status in ("warning", "critical"):
                        issues.append(
                            f"[{status.upper()}] {service_name}.{metric_name}={value}{unit}"
                        )
                except Exception:
                    pass

        for incident in self.incident_generator.get_active_incidents():
            issues.append(
                f"[INCIDENT] {incident['severity'].upper()} {incident['type'].upper()} "
                f"on {incident['targets']} — {incident['description']}"
            )

        return "\n".join(issues) if issues else "[INFO] All systems operational"

    def _check_metrics(self, target_service: str = None) -> str:
        if target_service:
            service = self.infrastructure.services.get(target_service)
            if not service:
                return f"Service '{target_service}' not found."
            
            # Convert MetricSnapshot objects to plain dicts
            metrics_dict = {
                name: {
                    "value":  m.value,
                    "status": m.status.value,
                    "unit":   m.unit,
                }
                for name, m in service.metrics.items()
            }
            metrics_str = json.dumps(metrics_dict, indent=2)
            return f"Metrics for {target_service}:\n{metrics_str}"
    
        all_metrics = self.infrastructure.get_all_metrics()
        return json.dumps(all_metrics, indent=2)[:500]
