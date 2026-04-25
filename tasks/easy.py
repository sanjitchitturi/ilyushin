from world.incident_generator import IncidentType, Severity

EASY_TASK = {
    "task_id": "easy",
    "difficulty": "easy",
    "description": (
        "The cache service has crashed. Identify the failure, restart the service, "
        "and confirm resolution. Single service, no cascading effects."
    ),
    "incidents": [
        {
            "type": IncidentType.SERVICE_CRASH,
            "targets": ["cache"],
            "severity": Severity.P2,
        }
    ],
    "total_services": 5,
    "target_services": ["cache"],
    "max_steps": 20,
    "solution_actions": [
        {"type": "read_logs"},
        {"type": "check_metrics", "target_service": "cache"},
        {"type": "restart_service", "target_service": "cache"},
        {"type": "resolve", "target_service": "cache"},
    ],
}