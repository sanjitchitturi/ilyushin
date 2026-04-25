from env.state import EnvState

STEP_PENALTY = -0.1
SERVICE_RECOVERED_REWARD = 3.0
SERVICE_DEGRADED_PENALTY = -1.0
WRONG_ACTION_PENALTY = -0.5
ONCALL_PENALTY = -5.0
ALL_RESOLVED_BONUS = 10.0
FAST_RESOLUTION_BONUS = 3.0
CASCADE_PREVENTED_BONUS = 2.0
FAST_RESOLUTION_STEP_THRESHOLD = 10


def compute_reward(old_state: EnvState, new_state: EnvState, action_type: str, action_success: bool) -> tuple:
    """
    Compute reward for a single action.
    
    Returns:
        (reward_value: float, breakdown: dict)
    """
    reward = 0.0
    breakdown = {}

    # Step penalty
    reward += STEP_PENALTY
    breakdown["step_penalty"] = STEP_PENALTY

    # Service recovery reward
    old_healthy = old_state.healthy_services
    new_healthy = new_state.healthy_services
    delta = new_healthy - old_healthy

    if delta > 0:
        service_reward = SERVICE_RECOVERED_REWARD * delta
        reward += service_reward
        breakdown["services_recovered"] = service_reward
    elif delta < 0:
        service_penalty = SERVICE_DEGRADED_PENALTY * abs(delta)
        reward += service_penalty
        breakdown["services_degraded"] = service_penalty

    # Wrong action penalty
    if not action_success and action_type not in ("check_metrics", "read_logs", "page_oncall", "resolve"):
        reward += WRONG_ACTION_PENALTY
        breakdown["wrong_action"] = WRONG_ACTION_PENALTY

    # Oncall penalty
    if action_type == "page_oncall" and not old_state.oncall_paged:
        reward += ONCALL_PENALTY
        breakdown["oncall_paged"] = ONCALL_PENALTY

    # Completion bonuses
    if new_state.healthy_services == new_state.total_services:
        reward += ALL_RESOLVED_BONUS
        breakdown["all_resolved_bonus"] = ALL_RESOLVED_BONUS

        if new_state.step_count <= FAST_RESOLUTION_STEP_THRESHOLD:
            reward += FAST_RESOLUTION_BONUS
            breakdown["fast_resolution_bonus"] = FAST_RESOLUTION_BONUS

    # Cascade prevention
    old_incidents = len(old_state.active_incidents)
    new_incidents = len(new_state.active_incidents)
    if new_incidents < old_incidents:
        cascade_reward = CASCADE_PREVENTED_BONUS * (old_incidents - new_incidents)
        reward += cascade_reward
        breakdown["incidents_cleared"] = cascade_reward

    return round(reward, 4), breakdown