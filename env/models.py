from typing import Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation


class IlyushinAction(Action):
    type: str = Field(..., description="Action type: read_logs, check_metrics, restart_service, scale_up, rollback, page_oncall, resolve")
    target_service: Optional[str] = Field(default=None, description="Target service name")


class IlyushinObservation(Observation):
    task_id: str = Field(..., description="Current task ID")
    step_count: int = Field(..., description="Number of steps taken")
    done: bool = Field(..., description="Whether the episode has ended")
    infrastructure: dict = Field(..., description="Current state of all services and metrics")
    active_incidents: list = Field(..., description="List of currently active incidents")
    healthy_services: int = Field(..., description="Number of healthy services")
    total_services: int = Field(..., description="Total number of services")
    last_action: str = Field(default="none", description="Last action taken")
    last_action_result: str = Field(default="", description="Result of last action")
    last_action_success: bool = Field(default=True, description="Whether last action succeeded")
    oncall_paged: bool = Field(default=False, description="Whether on-call engineer was paged")


class Reward:
    def __init__(self, value: float, breakdown: dict):
        self.value = value
        self.breakdown = breakdown