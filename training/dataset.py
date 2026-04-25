"""
dataset.py — Prompt formatting for the Ilyushin Responder Agent.

With PPO, there is no offline dataset generation.
This module only provides:
  - SYSTEM_PROMPT   : the agent's instructions
  - format_prompt() : converts environment state → readable prompt string
  - build_conversation() : wraps the above into chat message format
                           for tokenizer.apply_chat_template()

These are imported by train.py for use during live PPO rollouts.
"""

import json


SYSTEM_PROMPT = """You are a Responder Agent in a production incident response system.
Your job is to resolve infrastructure incidents as fast as possible.

You must return ONLY a JSON object with exactly these fields:
{"type": "<action_type>", "target_service": "<service_name or null>"}

Valid action types: read_logs, check_metrics, restart_service, scale_up, rollback, page_oncall, resolve
Valid services: web_server, database, cache, queue, api_gateway

RULES:
- Fix DOWN services immediately with restart_service
- Use scale_up for OVERLOADED services (high CPU + high RPS, overloaded=true)
- Use restart_service for CRASHED services (cpu=0, error_rate=100%)
- read_logs only once per episode
- Never page_oncall unless absolutely last resort
- Return raw JSON only"""


def format_prompt(state: dict) -> str:
    """
    Convert environment state dict into a human-readable prompt string.
    This is what the model sees as the user message at each step.
    """
    infrastructure   = state.get("infrastructure", {})
    active_incidents = state.get("active_incidents", [])
    step_count       = state.get("step_count", 0)
    healthy          = state.get("healthy_services", 0)
    total            = state.get("total_services", 5)
    last_result      = state.get("last_action_result", "")

    down_services = [
        svc for svc, data in infrastructure.items()
        if data.get("status") == "down"
    ]
    degraded_services = [
        svc for svc, data in infrastructure.items()
        if data.get("status") == "degraded"
    ]
    overloaded_services = [
        svc for svc, data in infrastructure.items()
        if data.get("overloaded") is True
    ]

    service_lines = []
    for svc, data in infrastructure.items():
        status     = data.get("status", "unknown")
        metrics    = data.get("metrics", {})
        cpu        = metrics.get("cpu",     {}).get("value", 0)
        mem        = metrics.get("memory",  {}).get("value", 0)
        lat        = metrics.get("latency", {}).get("value", 0)
        err        = metrics.get("error_rate", {}).get("value", 0)
        rps        = metrics.get("requests_per_second", {}).get("value", 0)
        overloaded = data.get("overloaded", False)
        service_lines.append(
            f"  {svc}: status={status} cpu={cpu:.1f}% mem={mem:.1f}% "
            f"latency={lat:.0f}ms error_rate={err:.2f}% rps={rps:.0f} "
            f"overloaded={overloaded}"
        )

    incident_lines = []
    for inc in active_incidents:
        incident_lines.append(
            f"  [{inc.get('severity','?').upper()}] {inc.get('type','?').upper()} "
            f"on {inc.get('targets', [])} — {inc.get('description', '')}"
        )

    prompt = f"""STEP: {step_count}/20
HEALTHY: {healthy}/{total}
DOWN SERVICES (restart immediately): {down_services}
DEGRADED SERVICES: {degraded_services}
OVERLOADED SERVICES (scale_up required): {overloaded_services}
LAST ACTION RESULT: {last_result}

ACTIVE INCIDENTS:
{chr(10).join(incident_lines) if incident_lines else "  None"}

INFRASTRUCTURE STATE:
{chr(10).join(service_lines)}

What is your next action? Return JSON only.
Respond with ONLY a JSON object. Example: {{"type": "restart_service", "target_service": "web_server"}}"""

    return prompt


def build_conversation(state: dict) -> list:
    """
    Build a chat message list from the current environment state.
    Used with tokenizer.apply_chat_template() during PPO rollouts.

    Returns:
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": <formatted state>}
        ]
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_prompt(state)},
    ]