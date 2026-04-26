"""
client.py — IlyushinClient

OpenEnv-compliant WebSocket client for the Ilyushin Incident Response Environment.

Usage (sync):
    from env.client import IlyushinClient

    with IlyushinClient(base_url="ws://localhost:8000").sync() as env:
        result = env.reset(task_id="easy")
        result = env.step({"type": "read_logs"})

Usage (async):
    async with IlyushinClient(base_url="ws://localhost:8000") as env:
        result = await env.reset(task_id="easy")
        result = await env.step({"type": "read_logs"})
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult
from env.models import IlyushinAction, IlyushinObservation, IlyushinState

VALID_SERVICES = ["web_server", "database", "cache", "queue", "api_gateway"]
VALID_ACTIONS  = ["read_logs", "check_metrics", "restart_service",
                  "scale_up", "rollback", "page_oncall", "resolve"]


class IlyushinClient(EnvClient):
    """
    OpenEnv-compliant WebSocket client for the Ilyushin environment.
    Connects to /ws and communicates via the standard OpenEnv protocol.
    """

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IlyushinObservation]:
        """Convert JSON response from env server to StepResult."""
        obs_data = payload.get("observation", payload)
        if not isinstance(obs_data, dict):
            obs_data = {}

        # Ensure required fields have defaults if missing
        obs_data.setdefault("done", False)
        obs_data.setdefault("task_id", "")
        obs_data.setdefault("step_count", 0)
        obs_data.setdefault("infrastructure", {})
        obs_data.setdefault("active_incidents", [])
        obs_data.setdefault("healthy_services", 0)
        obs_data.setdefault("total_services", 5)

        obs = IlyushinObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IlyushinState:
        """Convert JSON response from state endpoint to IlyushinState."""
        data = payload.get("data", payload)
        return IlyushinState(
            episode_id=data.get("episode_id", ""),
            step_count=data.get("step_count", 0),
            task_id=data.get("task_id", ""),
            healthy_services=data.get("healthy_services", 0),
            total_services=data.get("total_services", 5),
            oncall_paged=data.get("oncall_paged", False),
            last_action=data.get("last_action", "none"),
            last_action_result=data.get("last_action_result", ""),
        )

    def _step_payload(self, action) -> Dict[str, Any]:
        """Convert an action to the JSON payload the env server expects."""
        if isinstance(action, dict):
            payload = action.copy()
        elif isinstance(action, IlyushinAction):
            payload = action.model_dump()
        else:
            payload = dict(action)

        if "target_service" not in payload:
            payload["target_service"] = None

        return payload