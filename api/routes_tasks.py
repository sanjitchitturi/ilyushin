from fastapi import APIRouter, HTTPException
from tasks.registry import list_tasks, get_task

router = APIRouter()


@router.get("/")
def get_tasks():
    return {
        "tasks": list_tasks(),
        "action_schema": {
            "read_logs": {
                "type": "read_logs"
            },
            "check_metrics": {
                "type": "check_metrics",
                "target_service": "<service_name>"
            },
            "restart_service": {
                "type": "restart_service",
                "target_service": "<service_name>"
            },
            "scale_up": {
                "type": "scale_up",
                "target_service": "<service_name>"
            },
            "rollback": {
                "type": "rollback",
                "target_service": "<service_name>"
            },
            "page_oncall": {
                "type": "page_oncall"
            },
            "resolve": {
                "type": "resolve",
                "target_service": "<service_name>"
            }
        },
        "valid_services": [
            "web_server",
            "database",
            "cache",
            "queue",
            "api_gateway"
        ]
    }


@router.get("/{task_id}")
def get_task_by_id(task_id: str):
    try:
        task = get_task(task_id)
        return {
            "task_id": task_id,
            "difficulty": task["difficulty"],
            "description": task["description"],
            "total_services": task["total_services"],
            "target_services": task["target_services"],
            "max_steps": task["max_steps"],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))