from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from env.environment import IlyushinEnv
from tasks.registry import get_task

router = APIRouter()


class GraderRequest(BaseModel):
    session_id: str
    task_id: str


@router.post("/")
def grade(req: GraderRequest):
    try:
        task = get_task(req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    from api.routes_env import sessions
    env = sessions.get(req.session_id)

    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first."
        )

    try:
        obs = env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_services = obs.total_services
    healthy_services = obs.healthy_services
    score = round(healthy_services / total_services, 4) if total_services > 0 else 0.0

    step_efficiency = 1.0
    if obs.step_count > 0:
        step_efficiency = round(
            max(0.0, 1.0 - (obs.step_count / task["max_steps"]) * 0.2), 4
        )

    oncall_penalty = 0.1 if obs.oncall_paged else 0.0
    efficiency_bonus = round(max(0.0, score * step_efficiency - oncall_penalty), 4)
    final_score = score  # base score on healthy services only 

    return {
    "task_id": req.task_id,
    "session_id": req.session_id,
    "healthy_services": healthy_services,
    "total_services": total_services,
    "score": final_score,           
    "efficiency_score": efficiency_bonus,  # penalized for steps
    "raw_score": score,
    "step_efficiency": step_efficiency,
    "oncall_paged": obs.oncall_paged,
    "oncall_penalty": oncall_penalty,
    "steps_taken": obs.step_count,
    "done": obs.done,
}