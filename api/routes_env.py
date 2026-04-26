from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from env.environment import IlyushinEnv
from env.models import Observation
import traceback

router = APIRouter()

sessions: dict[str, IlyushinEnv] = {}


class ResetRequest(BaseModel):
    session_id: str
    task_id: str


class StepRequest(BaseModel):
    session_id: str
    action: dict


class FeedbackRequest(BaseModel):
    """Send Responder performance to Breaker for learning."""
    session_id: str
    success_rate: float
    avg_steps: int
    avg_reward: float
    failed_on_incidents: list = []
    succeeded_on_incidents: list = []
    unhealthy_services: list = []


def observation_to_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "__dict__"):
        return obs.__dict__
    return {}


@router.post("/reset")
def reset(req: ResetRequest):
    try:
        env = IlyushinEnv()
        obs = env.reset(req.task_id)
        sessions[req.session_id] = env

        return {
            "session_id":     req.session_id,
            "task_id":        req.task_id,
            "breaker_status": env.get_breaker_status(),
            "state":          observation_to_dict(obs),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/step")
def step(req: StepRequest):
    env = sessions.get(req.session_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call /reset first."
        )
    try:
        action_dict = req.action.copy() if isinstance(req.action, dict) else dict(req.action)

        if "target_service" not in action_dict:
            action_dict["target_service"] = None

        if "type" not in action_dict:
            raise ValueError("Action must have 'type' field")

        print(f"[STEP] Session {req.session_id}: action={action_dict}")

        # step() now returns an Observation with reward and done embedded
        obs = env.step(action_dict)
        reward_value = obs.reward if obs.reward is not None else 0.0
        done = obs.done

        return {
            "session_id":     req.session_id,
            "state":          observation_to_dict(obs),
            "reward":         {"value": round(reward_value, 4), "breakdown": {}},
            "done":           done,
            "breaker_status": env.get_breaker_status(),
        }
    except RuntimeError as e:
        print(f"[STEP] RuntimeError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        print(f"[STEP] ValueError: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"[STEP] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/feedback")
def send_feedback(req: FeedbackRequest):
    """Send Responder performance feedback to Breaker."""
    env = sessions.get(req.session_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found."
        )

    try:
        performance_data = {
            "success_rate":           req.success_rate,
            "avg_steps":              req.avg_steps,
            "avg_reward":             req.avg_reward,
            "failed_on_incidents":    req.failed_on_incidents,
            "succeeded_on_incidents": req.succeeded_on_incidents,
            "unhealthy_services":     req.unhealthy_services,
            "healthy_services":       5 - len(req.unhealthy_services),
            "total_services":         5,
        }

        env.breaker_agent.observe_responder_performance(performance_data)
        print(f"[BREAKER] Observed Responder success: {req.success_rate:.1%}")

        return {
            "session_id":         req.session_id,
            "breaker_status":     env.get_breaker_status(),
            "breaker_difficulty": env.breaker_agent.current_difficulty_level,
            "message":            "Breaker has learned and adapted",
        }
    except Exception as e:
        print(f"[FEEDBACK] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{session_id}")
def state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found."
        )
    try:
        obs = env.state()
        return {
            "session_id":     session_id,
            "state":          observation_to_dict(obs),
            "breaker_status": env.get_breaker_status(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/breaker/status/{session_id}")
def breaker_status(session_id: str):
    """Get Breaker agent status."""
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return {
        "session_id":     session_id,
        "breaker_status": env.get_breaker_status(),
    }