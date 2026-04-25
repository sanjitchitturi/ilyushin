import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

SYSTEM_PROMPT = """You are a Responder Agent in a production incident response system.
Your job is to resolve infrastructure incidents as fast as possible.

You must return ONLY a JSON object with exactly these fields:
{
    "type": "<action_type>",
    "target_service": "<service_name or null>"
}

Valid action types:
- read_logs (no target, use MAXIMUM once per episode)
- check_metrics (target: service name, use MAXIMUM once per service)
- restart_service (target: service name)
- scale_up (target: service name)
- rollback (target: service name)
- page_oncall (no target, LAST resort only, causes large penalty)
- resolve (target: service name, only when service status is healthy)

Valid services: web_server, database, cache, queue, api_gateway

STRICT RULES:
- read_logs can only be called ONCE. After that you must take fix actions.
- check_metrics can only be called ONCE per service. After that fix or move on.
- ALWAYS fix DOWN services before DEGRADED services.
- database and queue failures are almost always root causes. Fix them first.
- cache showing warning metrics is often a red herring. Ignore unless everything else is fixed.

CRITICAL — HOW TO CHOOSE BETWEEN restart_service AND scale_up:
- If a service is DOWN (cpu=0, error_rate=100%) use restart_service.
- If a service is DEGRADED with CRITICAL cpu AND CRITICAL requests_per_second use scale_up.
- If the last_action_result says "overload persists" or "scale_up required" use scale_up immediately.
- If a service is DEGRADED with high error_rate but normal cpu use restart_service.
- Overloaded services have: cpu > 90%, requests_per_second > 900, status=degraded, overloaded=true.
- Crashed services have: cpu=0, error_rate=100%, status=down.

- After restarting or scaling a service, call resolve on it if it becomes healthy.
- Never repeat the same action on the same service more than twice.
- Never page_oncall unless you have tried restart on every unhealthy service.
- Look at RECENT ACTIONS and never repeat a failing pattern.
- Return raw JSON only. No explanation. No markdown. No extra fields."""


class ResponderAgent:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.action_history = []

    def act(self, observation: dict, monitor_report: dict = None) -> dict:
        infrastructure   = observation.get("infrastructure", {})
        active_incidents = observation.get("active_incidents", [])
        last_result      = observation.get("last_action_result", "")
        step             = observation.get("step_count", 0)
        healthy          = observation.get("healthy_services", 0)
        total            = observation.get("total_services", 5)

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

        monitor_context = ""
        if monitor_report:
            monitor_context = f"""
MONITOR REPORT:
- Severity: {monitor_report.get('severity', 'unknown').upper()}
- Affected services: {monitor_report.get('affected_services', [])}
- Root cause hypothesis: {monitor_report.get('root_cause_hypothesis', 'unknown')}
- Recommended first action: {monitor_report.get('recommended_first_action', 'read_logs')}
"""

        history_context = ""
        if self.action_history:
            recent = self.action_history[-5:]
            history_context = f"\nRECENT ACTIONS (do not repeat failing patterns): {json.dumps(recent, indent=2)}"

        prompt = f"""Current production incident — take the best next action:

STEP: {step} / 20
HEALTHY SERVICES: {healthy}/{total}
DOWN SERVICES (use restart_service): {down_services}
DEGRADED SERVICES: {degraded_services}
OVERLOADED SERVICES (use scale_up, NOT restart): {overloaded_services}
LAST ACTION RESULT: {last_result}
ACTIVE INCIDENTS: {json.dumps(active_incidents, indent=2)}
{monitor_context}
{history_context}

INFRASTRUCTURE STATE:
{json.dumps(infrastructure, indent=2)}

Decision guide:
- DOWN service → restart_service immediately
- OVERLOADED service (overloaded=true, high cpu, high rps) → scale_up
- DEGRADED with high error_rate, normal cpu → restart_service
- last_action_result says overload persists → scale_up on that service now
- cache red herring → ignore unless all others fixed
- Return one action as raw JSON."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = "\n".join(
                    l for l in raw.split("\n")
                    if not l.strip().startswith("```")
                ).strip()
            action = json.loads(raw)
            self.action_history.append(action)
            return action
        except Exception:
            fallback = {"type": "read_logs", "target_service": None}
            self.action_history.append(fallback)
            return fallback

    def reset(self):
        self.action_history = []