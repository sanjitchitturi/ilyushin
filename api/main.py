import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from openenv.core.env_server import HTTPEnvServer

from env.environment import IlyushinEnv
from env.models import IlyushinAction, IlyushinObservation

from api.routes_env import router as env_router
from api.routes_tasks import router as tasks_router
from api.routes_grader import router as grader_router
from api.routes_baseline import router as baseline_router

app = FastAPI(
    title="Ilyushin — Incident Response Environment",
    description="OpenEnv-compatible environment for production incident response.",
    version="1.0.0",
)

# OpenEnv compliant: /ws, /reset, /step, /state, /health
server = HTTPEnvServer(
    env=IlyushinEnv,
    action_cls=IlyushinAction,
    observation_cls=IlyushinObservation,
)
server.register_routes(app)

# Legacy REST routes for training code: /env/reset, /env/step, /env/state
app.include_router(env_router,      prefix="/env",      tags=["Environment (legacy)"])
app.include_router(tasks_router,    prefix="/tasks",    tags=["Tasks"])
app.include_router(grader_router,   prefix="/grader",   tags=["Grader"])
app.include_router(baseline_router, prefix="/baseline", tags=["Baseline"])

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": [
            "/ws", "/reset", "/step", "/state", "/health",
            "/env/reset", "/env/step", "/env/state",
            "/tasks", "/grader", "/baseline"
        ]
    }