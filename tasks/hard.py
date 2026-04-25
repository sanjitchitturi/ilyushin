from world.incident_generator import IncidentType, Severity

HARD_TASK = {
    "task_id": "hard",
    "difficulty": "hard",
    "description": (
        "Full infrastructure meltdown. Database is crashed, api_gateway has a memory leak "
        "that worsens every tick, queue is overloaded with traffic and requires scaling not restarting, "
        "and cache shows red herring symptoms. Cascading failures affect web_server. "
        "Agent must diagnose correctly — restarting an overloaded service will not fix it."
    ),
    "incidents": [
        {
            "type":     IncidentType.SERVICE_CRASH,
            "targets":  ["database"],
            "severity": Severity.P1,
        },
        {
            "type":     IncidentType.MEMORY_LEAK,
            "targets":  ["api_gateway"],
            "severity": Severity.P2,
        },
        {
            "type":     IncidentType.OVERLOAD,
            "targets":  ["queue"],
            "severity": Severity.P2,
        },
        {
            "type":     IncidentType.RED_HERRING,
            "targets":  ["cache"],
            "severity": Severity.P2,
        },
    ],
    "total_services": 5,
    "target_services": ["database", "api_gateway", "queue", "web_server"],
    "max_steps": 20,
    "solution_actions": [
        {"type": "read_logs"},
        {"type": "check_metrics", "target_service": "database"},
        {"type": "restart_service", "target_service": "database"},
        {"type": "check_metrics", "target_service": "api_gateway"},
        {"type": "restart_service", "target_service": "api_gateway"},
        {"type": "check_metrics", "target_service": "queue"},
        {"type": "scale_up", "target_service": "queue"},
        {"type": "check_metrics", "target_service": "web_server"},
        {"type": "restart_service", "target_service": "web_server"},
        {"type": "resolve", "target_service": "database"},
        {"type": "resolve", "target_service": "api_gateway"},
        {"type": "resolve", "target_service": "queue"},
        {"type": "resolve", "target_service": "web_server"},
    ],
}