"""
dataset.py — Prompt formatting for the Ilyushin Responder Agent.

Key fix: prompt is now much shorter and ends with a clear JSON example
so the model completes valid JSON rather than continuing infrastructure text.
"""

SYSTEM_PROMPT = """You are an incident response agent. Output ONLY valid JSON.
Format: {"type": "ACTION", "target_service": "SERVICE_or_null"}
Actions: read_logs, check_metrics, restart_service, scale_up, rollback, page_oncall, resolve
Services: web_server, database, cache, queue, api_gateway
Rules: restart DOWN services immediately. scale_up OVERLOADED services. read_logs only once."""


def format_prompt(state: dict) -> str:
    """
    Very short prompt — just the critical state info the model needs.
    Ends with a complete example so the model knows exactly what to output.
    Long prompts were getting truncated mid-infrastructure-state causing
    the model to continue that text instead of outputting JSON.
    """
    infrastructure   = state.get("infrastructure", {})
    active_incidents = state.get("active_incidents", [])
    healthy          = state.get("healthy_services", 0)
    total            = state.get("total_services", 5)
    last_result      = state.get("last_action_result", "")

    # Only include critical service info — status and key metrics
    service_lines = []
    for svc, data in infrastructure.items():
        status     = data.get("status", "unknown")
        metrics    = data.get("metrics", {})
        cpu        = metrics.get("cpu", {}).get("value", 0)
        err        = metrics.get("error_rate", {}).get("value", 0)
        overloaded = data.get("overloaded", False)
        service_lines.append(
            f"{svc}: {status} cpu={cpu:.0f}% err={err:.1f}% overloaded={overloaded}"
        )

    # Only include incident types and targets
    incident_lines = []
    for inc in active_incidents:
        incident_lines.append(
            f"{inc.get('severity','?').upper()} {inc.get('type','?')} on {inc.get('targets', [])}"
        )

    down = [s for s, d in infrastructure.items() if d.get("status") == "down"]
    overloaded = [s for s, d in infrastructure.items() if d.get("overloaded") is True]

    prompt = f"""STATE: {healthy}/{total} healthy
DOWN: {down}
OVERLOADED: {overloaded}
INCIDENTS: {incident_lines if incident_lines else 'none'}
SERVICES: {chr(10).join(service_lines)}
LAST: {last_result[:80] if last_result else 'none'}

Output JSON action:
{{"type": "restart_service", "target_service": "web_server"}}

Your action:"""

    return prompt


def build_conversation(state: dict) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_prompt(state)},
    ]