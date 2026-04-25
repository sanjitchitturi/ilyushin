from world.incident_generator import IncidentType, Severity

MEDIUM_TASK = {
    "task_id": "medium",
    "difficulty": "medium",
    "description": (
        "The database has crashed, causing cascading degradation across api_gateway "
        "and web_server. A red herring CPU spike on queue makes diagnosis harder. "
        "Identify the root cause and fix it in the correct order."
    ),
    "incidents": [
        {
            "type": IncidentType.SERVICE_CRASH,
            "targets": ["database"],
            "severity": Severity.P1,
        },
        {
            "type": IncidentType.RED_HERRING,
            "targets": ["queue"],
            "severity": Severity.P2,
        },
    ],
    "total_services": 5,
    "target_services": ["database", "api_gateway", "web_server"],
    "max_steps": 20,
    "solution_actions": [
        {"type": "read_logs"},
        {"type": "check_metrics", "target_service": "database"},
        {"type": "restart_service", "target_service": "database"},
        {"type": "check_metrics", "target_service": "api_gateway"},
        {"type": "check_metrics", "target_service": "web_server"},
        {"type": "resolve", "target_service": "database"},
        {"type": "resolve", "target_service": "api_gateway"},
        {"type": "resolve", "target_service": "web_server"},
    ],
}