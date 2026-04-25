from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from env.environment import IlyushinEnv
from tasks.registry import get_task

router = APIRouter()


class BaselineRequest(BaseModel):
    task_id: str


@router.post("/")
def run_baseline(req: BaselineRequest):
    try:
        task = get_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    env = IlyushinEnv()
    obs = env.reset(req.task_id)

    trajectory   = []
    total_reward = 0.0

    for action in task["solution_actions"]:
        if obs.done:
            break

        try:
            # Unpack proper OpenEnv tuple: (obs, reward_float, done)
            obs, reward, done = env.step(action)
        except Exception as e:
            trajectory.append({
                "step":             obs.step_count,
                "action":           action,
                "error":            str(e),
                "healthy_services": obs.healthy_services,
                "reward":           0.0,
                "done":             obs.done,
            })
            break

        total_reward += reward  # reward is now a plain float
        trajectory.append({
            "step":               obs.step_count,
            "action":             action,
            "healthy_services":   obs.healthy_services,
            "total_services":     obs.total_services,
            "last_action_result": obs.last_action_result,
            "last_action_success": obs.last_action_success,
            "reward":             round(reward, 4),
            "done":               done,
        })

    total_services   = obs.total_services
    healthy_services = obs.healthy_services
    final_score      = round(healthy_services / total_services, 4) if total_services > 0 else 0.0

    return {
        "task_id":         req.task_id,
        "total_steps":     obs.step_count,
        "total_reward":    round(total_reward, 4),
        "final_score":     final_score,
        "healthy_services": healthy_services,
        "total_services":  total_services,
        "done":            obs.done,
        "oncall_paged":    obs.oncall_paged,
        "trajectory":      trajectory,
    }
