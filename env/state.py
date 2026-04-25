from dataclasses import dataclass, field


@dataclass
class EnvState:
    task_id: str
    step_count: int
    done: bool
    infrastructure_snapshot: dict
    active_incidents: list
    actions_taken: list
    services_resolved: list
    total_services: int
    healthy_services: int
    oncall_paged: bool
    last_action: str
    last_action_result: str
    last_action_success: bool

    def copy(self):
        return EnvState(
            task_id=self.task_id,
            step_count=self.step_count,
            done=self.done,
            infrastructure_snapshot=dict(self.infrastructure_snapshot),
            active_incidents=list(self.active_incidents),
            actions_taken=list(self.actions_taken),
            services_resolved=list(self.services_resolved),
            total_services=self.total_services,
            healthy_services=self.healthy_services,
            oncall_paged=self.oncall_paged,
            last_action=self.last_action,
            last_action_result=self.last_action_result,
            last_action_success=self.last_action_success,
        )